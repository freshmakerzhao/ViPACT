#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from constants import DT, get_equipment_model, get_sim_task_config, get_training_config, load_config
from policy_runtime import ACTPolicyRuntime
from sim_env import BOX_POSE, make_sim_env
from utils import sample_complex_scene_pose_eval, set_seed
from vipact_interfaces import BrainOutput, PolicyStepInput
from visualize_episodes import save_videos


EXPECTED_TASK = "sim_lifting_cube_with_complex_scene_scripted"
EXPECTED_EQUIPMENT = "fairino5_single"
EXPECTED_POLICY = "ACT"
MASK_CAMERA = "cockpit"
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
        print(f"[WARN] policy_best.ckpt not found, fallback to {last_path}")
        return last_path
    epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "policy_epoch_*_seed_*.ckpt"))
    if len(epoch_ckpts) == 0:
        raise FileNotFoundError(f"No ckpt found in {ckpt_dir}")
    epoch_ckpts.sort(
        key=lambda p: int(re.search(r"policy_epoch_(\d+)_seed_", os.path.basename(p)).group(1))
        if re.search(r"policy_epoch_(\d+)_seed_", os.path.basename(p))
        else -1
    )
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


def main():
    parser = argparse.ArgumentParser(description="Run ACT rollout with manually provided static cockpit mask.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mask-path", type=str, required=True, help="Binary mask image path (preferred from language->mask script)")
    parser.add_argument("--init-pose-json", type=str, default="", help='JSON path containing {"initial_box_pose":[...]}')
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=-1, help="Optional target id for strict success metric")
    parser.add_argument("--output-dir", type=str, default="./notes/act_manual_mask")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    train_cfg = get_training_config(args.config)
    task_name = yaml_cfg.get("task", {}).get("name", "")
    equipment_model = get_equipment_model(args.config)
    policy_class = train_cfg.get("policy_class", "")
    if task_name != EXPECTED_TASK:
        raise ValueError(f"Only supports task={EXPECTED_TASK}, got {task_name}")
    if equipment_model != EXPECTED_EQUIPMENT:
        raise ValueError(f"Only supports equipment={EXPECTED_EQUIPMENT}, got {equipment_model}")
    if policy_class != EXPECTED_POLICY:
        raise ValueError(f"Only supports policy_class={EXPECTED_POLICY}, got {policy_class}")
    if not bool(train_cfg.get("use_mask_conditioning", False)):
        raise ValueError("This test expects vipact.use_mask_conditioning=true")

    task_cfg = get_sim_task_config(task_name, args.config)
    camera_names = task_cfg["camera_names"]
    if MASK_CAMERA not in camera_names:
        raise ValueError(f"camera_names={camera_names} has no '{MASK_CAMERA}'")

    episode_len = int(task_cfg["episode_len"]) if args.max_steps <= 0 else int(args.max_steps)
    ckpt_dir = train_cfg.get("ckpt_dir", "./ckpts")
    ckpt_path = resolve_ckpt_path(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"dataset_stats.pkl not found: {stats_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 1000)) + int(args.episode_id)
    set_seed(seed)

    if args.init_pose_json:
        with open(args.init_pose_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        init_pose = np.asarray(obj["initial_box_pose"], dtype=np.float64).copy()
    else:
        init_pose = np.asarray(sample_complex_scene_pose_eval(), dtype=np.float64).copy()
    BOX_POSE[0] = init_pose.copy()
    env = make_sim_env(task_name, equipment_model=equipment_model)
    ts = env.reset()

    policy_config = {
        "lr": float(train_cfg.get("lr", 1e-5)),
        "num_queries": train_cfg.get("chunk_size", 100),
        "kl_weight": train_cfg.get("kl_weight", 10),
        "hidden_dim": train_cfg.get("hidden_dim", 512),
        "dim_feedforward": train_cfg.get("dim_feedforward", 3200),
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "equipment_model": equipment_model,
        "use_mask_conditioning": bool(train_cfg.get("use_mask_conditioning", False)),
        "image_channels": int(train_cfg.get("image_channels", 4)),
        "temporal_agg": bool(train_cfg.get("temporal_agg", False)),
    }
    runtime = ACTPolicyRuntime.from_checkpoint(
        ckpt_path=ckpt_path,
        stats_path=stats_path,
        policy_config=policy_config,
        camera_names=camera_names,
    )
    runtime.reset_episode()

    h, w = ts.observation["images"][MASK_CAMERA].shape[:2]
    cockpit_mask = load_binary_mask(args.mask_path, height=h, width=w)
    static_mask_dict = {MASK_CAMERA: cockpit_mask}

    target_geom_names = list_target_geom_names(env._physics)
    target_geom_name = None
    if 0 <= int(args.target_id) < len(target_geom_names):
        target_geom_name = target_geom_names[int(args.target_id)]
        if hasattr(env, "task"):
            env.task.current_target_id = int(args.target_id)
            env.task.target_geom_name = target_geom_name

    image_list = []
    rewards = []
    max_z_by_geom = {name: -1e9 for name in target_geom_names}
    action_l2_list = []
    qpos_delta_l2_list = []

    for t in range(episode_len):
        image_list.append(ts.observation["images"])
        step_output = runtime.step(
            PolicyStepInput(
                qpos=np.asarray(ts.observation["qpos"]),
                images_by_camera=ts.observation["images"],
                brain_output=BrainOutput(target_id=int(args.target_id) if args.target_id >= 0 else 0),
                static_mask_by_camera=static_mask_dict,
                step_id=t,
            )
        )
        action_l2_list.append(float(np.linalg.norm(np.asarray(step_output.action))))
        prev_qpos = np.asarray(ts.observation["qpos"], dtype=np.float32).copy()
        ts = env.step(step_output.action)
        curr_qpos = np.asarray(ts.observation["qpos"], dtype=np.float32)
        qpos_delta_l2_list.append(float(np.linalg.norm(curr_qpos - prev_qpos)))
        rewards.append(float(ts.reward))

        heights = get_object_heights(env._physics, target_geom_names)
        for geom_name, z in heights.items():
            max_z_by_geom[geom_name] = max(max_z_by_geom[geom_name], z)

    video_path = ""
    if args.save_video:
        video_path = str(output_dir / f"manual_mask_ep{args.episode_id}.mp4")
        save_videos(image_list, DT, video_path=video_path)

    episode_return = float(np.sum(np.asarray(rewards)))
    lifted_flags = {k: (v >= DEFAULT_LIFT_THRESHOLD) for k, v in max_z_by_geom.items()}
    guessed_grab_geom = max(max_z_by_geom.items(), key=lambda kv: kv[1])[0]
    target_lifted = bool(lifted_flags[target_geom_name]) if target_geom_name is not None else None
    strict_success = bool(target_lifted and guessed_grab_geom == target_geom_name) if target_geom_name is not None else None

    summary = {
        "task_name": task_name,
        "equipment_model": equipment_model,
        "policy_class": EXPECTED_POLICY,
        "seed": int(seed),
        "episode_id": int(args.episode_id),
        "episode_len": int(episode_len),
        "ckpt_path": ckpt_path,
        "stats_path": stats_path,
        "camera_names": camera_names,
        "mask_camera_names": [MASK_CAMERA],
        "mask_path": str(Path(args.mask_path).resolve()),
        "init_pose_json": str(Path(args.init_pose_json).resolve()) if args.init_pose_json else "",
        "target_id": int(args.target_id),
        "target_geom": target_geom_name,
        "target_id_mapping": {str(i): name for i, name in enumerate(target_geom_names)},
        "episode_return": episode_return,
        "lift_threshold": float(DEFAULT_LIFT_THRESHOLD),
        "max_z_by_geom": {k: float(v) for k, v in max_z_by_geom.items()},
        "lifted_flags": {k: bool(v) for k, v in lifted_flags.items()},
        "guessed_grab_geom": guessed_grab_geom,
        "target_lifted": target_lifted,
        "strict_success": strict_success,
        "action_l2_stats": {
            "mean": float(np.mean(action_l2_list)) if action_l2_list else 0.0,
            "min": float(np.min(action_l2_list)) if action_l2_list else 0.0,
            "max": float(np.max(action_l2_list)) if action_l2_list else 0.0,
        },
        "qpos_delta_l2_stats": {
            "mean": float(np.mean(qpos_delta_l2_list)) if qpos_delta_l2_list else 0.0,
            "min": float(np.min(qpos_delta_l2_list)) if qpos_delta_l2_list else 0.0,
            "max": float(np.max(qpos_delta_l2_list)) if qpos_delta_l2_list else 0.0,
        },
        "video_path": video_path,
    }
    summary_path = output_dir / f"summary_manual_mask_ep{args.episode_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {summary_path}")
    if video_path:
        print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()

