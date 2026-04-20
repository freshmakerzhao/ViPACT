from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from constants import DT, get_equipment_model, get_sim_task_config, get_training_config, load_config
from interfaces import BrainOutput, PolicyStepInput
from multimodal_brain import BrainPerceptionFacade
from pipeline.artifact_logger import ArtifactLogger
from policy_runtime import ACTPolicyRuntime
from sim_env import BOX_POSE, make_sim_env
from utils import sample_complex_scene_pose_eval, set_seed
from visualize_episodes import save_videos


EXPECTED_TASK = "sim_lifting_cube_with_complex_scene_scripted"
EXPECTED_EQUIPMENT = "fairino5_single"
EXPECTED_POLICY = "ACT"
DEFAULT_MASK_CAMERA = "cockpit"
DEFAULT_LIFT_THRESHOLD = 0.09


def _safe_name2id(model, name: str, obj_type: str):
    try:
        return model.name2id(name, obj_type)
    except Exception:
        return None


def list_target_geom_names(physics) -> List[str]:
    names = []
    if _safe_name2id(physics.model, "red_box", "geom") is not None:
        names.append("red_box")
    i = 1
    while True:
        name = f"distractor_box_{i}"
        if _safe_name2id(physics.model, name, "geom") is None:
            break
        names.append(name)
        i += 1
    if len(names) == 0:
        raise RuntimeError("No target boxes found in current scene.")
    return names


def geom_to_joint_name(geom_name: str) -> str:
    if geom_name == "red_box":
        return "red_box_joint"
    if geom_name.startswith("distractor_box_"):
        return f"{geom_name}_joint"
    raise ValueError(f"Unsupported geom name: {geom_name}")


def get_object_heights(physics, geom_names: List[str]) -> Dict[str, float]:
    out = {}
    for geom_name in geom_names:
        joint_name = geom_to_joint_name(geom_name)
        joint_id = physics.model.name2id(joint_name, "joint")
        qpos_start = int(physics.model.jnt_qposadr[joint_id])
        out[geom_name] = float(physics.data.qpos[qpos_start + 2])
    return out


def resolve_ckpt_path(ckpt_dir: str) -> str:
    best_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    if os.path.isfile(best_path):
        return best_path
    last_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    if os.path.isfile(last_path):
        return last_path
    epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "policy_epoch_*_seed_*.ckpt"))
    if len(epoch_ckpts) == 0:
        raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
    def _epoch_key(path: str) -> int:
        m = re.search(r"policy_epoch_(\d+)_seed_", os.path.basename(path))
        return int(m.group(1)) if m else -1

    epoch_ckpts.sort(key=_epoch_key)
    return epoch_ckpts[-1]


def load_binary_mask(mask_path: str, height: int, width: int) -> np.ndarray:
    img = Image.open(mask_path).convert("L")
    if img.size != (width, height):
        img = img.resize((width, height), resample=Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = (arr > 127.0).astype(np.float32)
    else:
        arr = (arr > 0.5).astype(np.float32)
    return arr


@dataclass
class BrainToPolicyPipeline:
    config_path: str
    output_dir: str
    mask_camera: str = DEFAULT_MASK_CAMERA

    def __post_init__(self):
        self.yaml_cfg = load_config(self.config_path)
        self.train_cfg = get_training_config(self.config_path)
        self.task_name = self.yaml_cfg.get("task", {}).get("name", "")
        self.equipment_model = get_equipment_model(self.config_path)
        self.policy_class = self.train_cfg.get("policy_class", "")

        if self.task_name != EXPECTED_TASK:
            raise ValueError(f"Only supports task={EXPECTED_TASK}, got {self.task_name}")
        if self.equipment_model != EXPECTED_EQUIPMENT:
            raise ValueError(f"Only supports equipment={EXPECTED_EQUIPMENT}, got {self.equipment_model}")
        if self.policy_class != EXPECTED_POLICY:
            raise ValueError(f"Only supports policy_class={EXPECTED_POLICY}, got {self.policy_class}")
        if not bool(self.train_cfg.get("use_mask_conditioning", False)):
            raise ValueError("This pipeline requires vipact.use_mask_conditioning=true")

        self.task_cfg = get_sim_task_config(self.task_name, self.config_path)
        self.camera_names = list(self.task_cfg["camera_names"])
        if self.mask_camera not in self.camera_names:
            raise ValueError(f"mask_camera={self.mask_camera} not in camera_names={self.camera_names}")

        self.ckpt_dir = self.train_cfg.get("ckpt_dir", "./ckpts")
        self.ckpt_path = resolve_ckpt_path(self.ckpt_dir)
        self.stats_path = os.path.join(self.ckpt_dir, "dataset_stats.pkl")
        if not os.path.isfile(self.stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found: {self.stats_path}")

        self.output_dir_path = Path(self.output_dir).resolve()
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.artifact_logger = ArtifactLogger(str(self.output_dir_path))

        self.policy_config = {
            "lr": float(self.train_cfg.get("lr", 1e-5)),
            "num_queries": self.train_cfg.get("chunk_size", 100),
            "kl_weight": self.train_cfg.get("kl_weight", 10),
            "hidden_dim": self.train_cfg.get("hidden_dim", 512),
            "dim_feedforward": self.train_cfg.get("dim_feedforward", 3200),
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": self.camera_names,
            "equipment_model": self.equipment_model,
            "use_mask_conditioning": bool(self.train_cfg.get("use_mask_conditioning", False)),
            "image_channels": int(self.train_cfg.get("image_channels", 4)),
            "temporal_agg": bool(self.train_cfg.get("temporal_agg", False)),
        }

    def _create_env(self, episode_id: int, init_pose_json: str = ""):
        seed = int(self.train_cfg.get("seed", 1000)) + int(episode_id)
        set_seed(seed)
        if init_pose_json:
            with open(init_pose_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            init_pose = np.asarray(obj["initial_box_pose"], dtype=np.float64).copy()
        else:
            init_pose = np.asarray(sample_complex_scene_pose_eval(), dtype=np.float64).copy()
        BOX_POSE[0] = init_pose.copy()
        env = make_sim_env(self.task_name, equipment_model=self.equipment_model)
        ts = env.reset()
        return env, ts, seed, init_pose

    def _create_runtime(self) -> ACTPolicyRuntime:
        runtime = ACTPolicyRuntime.from_checkpoint(
            ckpt_path=self.ckpt_path,
            stats_path=self.stats_path,
            policy_config=self.policy_config,
            camera_names=self.camera_names,
        )
        runtime.reset_episode()
        return runtime

    def _run_rollout(
        self,
        *,
        env,
        ts,
        runtime: ACTPolicyRuntime,
        run_dir: Path,
        static_mask_dict: Dict[str, np.ndarray],
        episode_id: int,
        seed: int,
        episode_len: int,
        target_id: int = -1,
        save_video: bool = False,
        mask_path: str = "",
        extra_meta: Optional[Dict] = None,
        onscreen_render: bool = False,
        onscreen_cam: Optional[str] = None,
    ) -> Dict:
        target_geom_names = list_target_geom_names(env._physics)
        target_geom_name = None
        if 0 <= int(target_id) < len(target_geom_names):
            target_geom_name = target_geom_names[int(target_id)]
            if hasattr(env, "task"):
                env.task.current_target_id = int(target_id)
                env.task.target_geom_name = target_geom_name

        image_list = []
        rewards = []
        max_z_by_geom = {name: -1e9 for name in target_geom_names}

        plt_img = None
        if onscreen_render:
            cam_name = onscreen_cam if onscreen_cam in self.camera_names else self.mask_camera
            fig = plt.figure("ViPACT Onscreen Render")
            ax = fig.add_subplot(111)
            first_img = env._physics.render(height=480, width=640, camera_id=cam_name)
            plt_img = ax.imshow(first_img)
            ax.set_title(f"camera={cam_name}")
            plt.ion()
            plt.show(block=False)
            plt.pause(0.001)

        for t in range(int(episode_len)):
            image_list.append(ts.observation["images"])
            step_output = runtime.step(
                PolicyStepInput(
                    qpos=np.asarray(ts.observation["qpos"]),
                    images_by_camera=ts.observation["images"],
                    brain_output=BrainOutput(target_id=int(target_id) if target_id >= 0 else 0),
                    static_mask_by_camera=static_mask_dict,
                    step_id=t,
                )
            )
            ts = env.step(step_output.action)
            rewards.append(float(ts.reward))
            if onscreen_render and plt_img is not None:
                cam_name = onscreen_cam if onscreen_cam in self.camera_names else self.mask_camera
                img = env._physics.render(height=480, width=640, camera_id=cam_name)
                plt_img.set_data(img)
                plt.pause(DT)
            heights = get_object_heights(env._physics, target_geom_names)
            for geom_name, z in heights.items():
                max_z_by_geom[geom_name] = max(max_z_by_geom[geom_name], z)

        video_path = ""
        if save_video:
            video_path = str(run_dir / "rollout.mp4")
            save_videos(image_list, DT, video_path=video_path)

        episode_return = float(np.sum(np.asarray(rewards)))
        lifted_flags = {k: (v >= DEFAULT_LIFT_THRESHOLD) for k, v in max_z_by_geom.items()}
        guessed_grab_geom = max(max_z_by_geom.items(), key=lambda kv: kv[1])[0]
        target_lifted = bool(lifted_flags[target_geom_name]) if target_geom_name is not None else None
        strict_success = bool(target_lifted and guessed_grab_geom == target_geom_name) if target_geom_name is not None else None

        summary = {
            "task_name": self.task_name,
            "equipment_model": self.equipment_model,
            "policy_class": EXPECTED_POLICY,
            "seed": int(seed),
            "episode_id": int(episode_id),
            "episode_len": int(episode_len),
            "ckpt_path": self.ckpt_path,
            "stats_path": self.stats_path,
            "camera_names": self.camera_names,
            "mask_camera_names": [self.mask_camera],
            "mask_path": str(mask_path),
            "target_id": int(target_id),
            "target_geom": target_geom_name,
            "target_id_mapping": {str(i): name for i, name in enumerate(target_geom_names)},
            "episode_return": episode_return,
            "lift_threshold": float(DEFAULT_LIFT_THRESHOLD),
            "max_z_by_geom": {k: float(v) for k, v in max_z_by_geom.items()},
            "lifted_flags": {k: bool(v) for k, v in lifted_flags.items()},
            "guessed_grab_geom": guessed_grab_geom,
            "target_lifted": target_lifted,
            "strict_success": strict_success,
            "video_path": video_path,
            "run_dir": str(run_dir),
        }
        if extra_meta:
            summary["extra_meta"] = extra_meta
        if onscreen_render:
            try:
                plt.ioff()
            except Exception:
                pass
        return summary

    def run_with_manual_mask(
        self,
        *,
        mask_path: str,
        episode_id: int = 0,
        target_id: int = -1,
        max_steps: int = -1,
        init_pose_json: str = "",
        save_video: bool = False,
        onscreen_render: bool = False,
        onscreen_cam: Optional[str] = None,
    ) -> Dict:
        env, ts, seed, _ = self._create_env(episode_id=episode_id, init_pose_json=init_pose_json)
        runtime = self._create_runtime()
        episode_len = int(self.task_cfg["episode_len"]) if max_steps <= 0 else int(max_steps)
        run_dir = self.artifact_logger.create_run_dir(f"manual_mask_ep{int(episode_id)}")

        h, w = ts.observation["images"][self.mask_camera].shape[:2]
        static_mask = load_binary_mask(mask_path, height=h, width=w)
        static_mask_dict = {self.mask_camera: static_mask}
        saved_mask_path = self.artifact_logger.save_mask(run_dir / "mask.png", static_mask)

        summary = self._run_rollout(
            env=env,
            ts=ts,
            runtime=runtime,
            run_dir=run_dir,
            static_mask_dict=static_mask_dict,
            episode_id=episode_id,
            seed=seed,
            episode_len=episode_len,
            target_id=target_id,
            save_video=save_video,
            mask_path=str(saved_mask_path),
            onscreen_render=bool(onscreen_render),
            onscreen_cam=onscreen_cam,
        )
        summary["artifact_paths"] = {
            "mask": str(saved_mask_path),
            "summary": str(run_dir / "run_summary.json"),
        }
        out_path = self.artifact_logger.save_json(run_dir / "run_summary.json", summary)
        summary["summary_path"] = str(out_path)
        return summary

    def run_with_instruction(
        self,
        *,
        instruction: str,
        episode_id: int = 0,
        target_id: int = -1,
        max_steps: int = -1,
        init_pose_json: str = "",
        save_video: bool = False,
        llm_api_key: str = "",
        llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_model: str = "qwen-plus",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam2-hiera-small",
        confidence_threshold: float = 0.5,
        tolerance: float = 20.0,
        device: str = "cuda",
        onscreen_render: bool = False,
        onscreen_cam: Optional[str] = None,
    ) -> Dict:
        env, ts, seed, _ = self._create_env(episode_id=episode_id, init_pose_json=init_pose_json)
        runtime = self._create_runtime()
        episode_len = int(self.task_cfg["episode_len"]) if max_steps <= 0 else int(max_steps)
        run_dir = self.artifact_logger.create_run_dir(f"instruction_ep{int(episode_id)}")

        rgb = np.asarray(ts.observation["images"][self.mask_camera], dtype=np.uint8)
        facade = BrainPerceptionFacade.create_default(
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            grounding_model_id=grounding_model_id,
            sam_model_id=sam_model_id,
            device=device,
        )
        brain_result = facade.run_once(
            rgb_image=rgb,
            instruction=instruction,
            confidence_threshold=float(confidence_threshold),
            tolerance=float(tolerance),
        )
        if not brain_result.get("ok", False):
            raise RuntimeError(f"Brain perception failed: {brain_result.get('reason', 'unknown')}")

        mask_u8 = brain_result["mask"].mask
        static_mask_dict = {self.mask_camera: mask_u8.astype(np.float32)}

        mask_path = self.artifact_logger.save_mask(run_dir / "mask.png", mask_u8.astype(np.uint8))
        self.artifact_logger.save_json(run_dir / "parsed_instruction.json", brain_result.get("parsed"))
        self.artifact_logger.save_json(run_dir / "detection.json", brain_result.get("detection"))
        self.artifact_logger.save_json(run_dir / "selection.json", brain_result.get("selection"))

        summary = self._run_rollout(
            env=env,
            ts=ts,
            runtime=runtime,
            run_dir=run_dir,
            static_mask_dict=static_mask_dict,
            episode_id=episode_id,
            seed=seed,
            episode_len=episode_len,
            target_id=target_id,
            save_video=save_video,
            mask_path=str(mask_path),
            extra_meta={
                "instruction": instruction,
                "parsed": getattr(brain_result.get("parsed"), "raw", {}),
                "selection": brain_result.get("selection").selected if brain_result.get("selection") else {},
            },
            onscreen_render=bool(onscreen_render),
            onscreen_cam=onscreen_cam,
        )
        summary["artifact_paths"] = {
            "mask": str(mask_path),
            "parsed_instruction": str(run_dir / "parsed_instruction.json"),
            "detection": str(run_dir / "detection.json"),
            "selection": str(run_dir / "selection.json"),
            "summary": str(run_dir / "run_summary.json"),
        }
        out_path = self.artifact_logger.save_json(run_dir / "run_summary.json", summary)
        summary["summary_path"] = str(out_path)
        return summary
