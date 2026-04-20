#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
from mujoco import viewer
import numpy as np
from PIL import Image

from constants import DT, XML_DIR
from interfaces import BrainOutput, PolicyStepInput
from multimodal_brain import BrainPerceptionFacade
from multimodal_brain.mask_generator.sam2_box_mask import mask_overlay
from pipeline.brain_to_policy_pipeline import (
    BrainToPolicyPipeline,
    DEFAULT_LIFT_THRESHOLD,
    get_object_heights,
    list_target_geom_names,
)


def _resolve_joint_xml_path(task_name: str, equipment_model: str) -> str:
    if "sim_lifting_cube" in task_name:
        if "fairino5_single" in equipment_model:
            if "with_complex_scene" in task_name:
                xml_filename = "fairino_fr5_lifting_cube_with_complex_scene.xml"
            else:
                xml_filename = "fairino_fr5_lifting_cube.xml"
        elif equipment_model == "excavator_simple":
            xml_filename = "single_viperx_transfer_cube.xml"
        else:
            xml_filename = "single_viperx_transfer_cube.xml"
    elif "sim_transfer_cube" in task_name:
        xml_filename = "bimanual_viperx_transfer_cube.xml"
    elif "sim_insertion" in task_name:
        xml_filename = "bimanual_viperx_insertion.xml"
    else:
        raise ValueError(f"Unsupported task_name={task_name}")
    return str(Path(XML_DIR) / equipment_model / xml_filename)


def _sync_viewer_state(view_model, view_data, env_physics):
    src_qpos = np.asarray(env_physics.data.qpos)
    src_qvel = np.asarray(env_physics.data.qvel)
    src_ctrl = np.asarray(env_physics.data.ctrl)

    nq = min(view_model.nq, src_qpos.shape[0])
    nv = min(view_model.nv, src_qvel.shape[0])
    nu = min(view_model.nu, src_ctrl.shape[0])
    view_data.qpos[:nq] = src_qpos[:nq]
    view_data.qvel[:nv] = src_qvel[:nv]
    if nu > 0:
        view_data.ctrl[:nu] = src_ctrl[:nu]
    mujoco.mj_forward(view_model, view_data)


def _get_keyframe_qpos(model: mujoco.MjModel, key_name: str = "init_pose"):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        return None
    nq = int(model.nq)
    start = key_id * nq
    end = start + nq
    return np.array(model.key_qpos[start:end], dtype=np.float64)


def _apply_init_pose_if_exists(physics, view_model: mujoco.MjModel):
    """
    If XML has keyframe `init_pose`, force env physics to that qpos.
    This guarantees startup pose consistency for demo.
    """
    key_qpos = _get_keyframe_qpos(view_model, "init_pose")
    if key_qpos is None:
        return False
    qpos = np.asarray(physics.data.qpos)
    nq = min(qpos.shape[0], key_qpos.shape[0])
    qpos[:nq] = key_qpos[:nq]
    physics.forward()
    return True


def _parse_cli():
    parser = argparse.ArgumentParser(description="ViPACT passive viewer console demo (IDLE/THINKING/EXECUTING).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fairino5_single_26041508_with_complex_scene_cf_mask/03_eval.yaml",
    )
    parser.add_argument("--output-dir", type=str, default="notes/passive_viewer_runs")
    parser.add_argument("--episode-id-base", type=int, default=0, help="First episode id; increments per command.")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sleep-dt", type=float, default=DT, help="Wall-time sleep per control step for visible execution.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm-base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm-model", type=str, default="qwen-plus")
    parser.add_argument("--grounding-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model-id", type=str, default="facebook/sam2-hiera-small")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = _parse_cli()

    pipeline = BrainToPolicyPipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        mask_camera="cockpit",
    )

    xml_path = _resolve_joint_xml_path(pipeline.task_name, pipeline.equipment_model)
    view_model = mujoco.MjModel.from_xml_path(xml_path)
    view_data = mujoco.MjData(view_model)

    llm_api_key = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("QWEN_API_KEY", "").strip()
    )
    facade = BrainPerceptionFacade.create_default(
        llm_api_key=llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        grounding_model_id=args.grounding_model_id,
        sam_model_id=args.sam_model_id,
        device=args.device,
    )

    print("\n=== ViPACT Passive Viewer Console ===")
    print("State machine: IDLE -> THINKING -> EXECUTING")
    print("Commands: 输入自然语言指令执行；输入 q 退出。\n")
    command_counter = 0

    with viewer.launch_passive(view_model, view_data) as v:
        # Startup scene fixed at launch.
        fixed_episode_id = int(args.episode_id_base)
        fixed_env, fixed_ts, fixed_seed, _ = pipeline._create_env(episode_id=fixed_episode_id, init_pose_json="")
        used_init_pose = _apply_init_pose_if_exists(fixed_env._physics, view_model)
        _sync_viewer_state(view_model, view_data, fixed_env._physics)
        v.sync()
        print(
            f"[INIT] fixed_episode_id={fixed_episode_id}, fixed_seed={fixed_seed}, "
            f"use_init_pose_keyframe={used_init_pose}"
        )

        while v.is_running():
            # IDLE: block on user input. Physics remains static; viewer stays interactive.
            print("\n[IDLE] 等待指令 (输入 q 退出)")
            instruction = input(">>> ").strip()
            if instruction.lower() in {"q", "quit", "exit"}:
                print("退出。")
                break
            if not instruction:
                print("[IDLE] 空指令，继续等待。")
                continue
            command_counter += 1

            # Scene is fixed across commands by reusing same episode_id/seed.
            episode_id = fixed_episode_id

            # New episode for each command.
            env, ts, seed, _ = pipeline._create_env(episode_id=episode_id, init_pose_json="")
            _apply_init_pose_if_exists(env._physics, view_model)
            runtime = pipeline._create_runtime()
            runtime.reset_episode()

            _sync_viewer_state(view_model, view_data, env._physics)
            v.sync()

            # THINKING
            print(f"[THINKING] episode_id={episode_id}, seed={seed}")
            rgb = np.asarray(ts.observation["images"][pipeline.mask_camera], dtype=np.uint8)
            brain_result = facade.run_once(
                rgb_image=rgb,
                instruction=instruction,
                confidence_threshold=float(args.confidence_threshold),
                tolerance=float(args.tolerance),
            )
            if not brain_result.get("ok", False):
                print(f"[THINKING] 失败: {brain_result.get('reason', 'unknown')}")
                continue

            mask_u8 = brain_result["mask"].mask.astype(np.uint8)
            static_mask_dict = {pipeline.mask_camera: mask_u8.astype(np.float32)}
            parsed_raw = getattr(brain_result.get("parsed"), "raw", {})
            selected = getattr(brain_result.get("selection"), "selected", {})
            print(
                f"[THINKING] 完成: parsed={parsed_raw}, selected={selected}, mask_sum={int(mask_u8.sum())}"
            )

            # Save debug artifacts per command for fast diagnosis.
            debug_dir = Path(args.output_dir).resolve() / "brain_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            mask_path = debug_dir / f"cmd_{command_counter:03d}_mask.png"
            overlay_path = debug_dir / f"cmd_{command_counter:03d}_overlay.png"
            meta_path = debug_dir / f"cmd_{command_counter:03d}_meta.txt"
            Image.fromarray((mask_u8 > 0).astype(np.uint8) * 255, mode="L").save(mask_path)
            Image.fromarray(mask_overlay(rgb, mask_u8)).save(overlay_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"instruction={instruction}\n")
                f.write(f"parsed={parsed_raw}\n")
                f.write(f"selected={selected}\n")
                f.write(f"mask_sum={int(mask_u8.sum())}\n")
            print(f"[THINKING] debug saved: {mask_path}, {overlay_path}, {meta_path}")

            # EXECUTING
            print("[EXECUTING] ViPACT 接管执行...")
            episode_len = int(pipeline.task_cfg["episode_len"]) if args.max_steps <= 0 else int(args.max_steps)
            num_queries = int(getattr(runtime, "num_queries", 1))
            expected_chunks = int(np.ceil(float(episode_len) / float(max(1, num_queries))))
            print(
                f"[EXECUTING] episode_len={episode_len}, num_queries={num_queries}, "
                f"expected_policy_calls~{expected_chunks}"
            )
            target_geom_names = list_target_geom_names(env._physics)
            max_z_by_geom = {name: -1e9 for name in target_geom_names}
            rewards = []
            executed_steps = 0

            for t in range(episode_len):
                if not v.is_running():
                    break
                step_output = runtime.step(
                    PolicyStepInput(
                        qpos=np.asarray(ts.observation["qpos"]),
                        images_by_camera=ts.observation["images"],
                        brain_output=BrainOutput(target_id=0),
                        static_mask_by_camera=static_mask_dict,
                        step_id=t,
                    )
                )
                ts = env.step(step_output.action)
                rewards.append(float(ts.reward))

                heights = get_object_heights(env._physics, target_geom_names)
                for geom_name, z in heights.items():
                    max_z_by_geom[geom_name] = max(max_z_by_geom[geom_name], z)

                _sync_viewer_state(view_model, view_data, env._physics)
                v.sync()
                time.sleep(float(args.sleep_dt))
                executed_steps += 1

            episode_return = float(np.sum(np.asarray(rewards))) if rewards else 0.0
            guessed_grab_geom = max(max_z_by_geom.items(), key=lambda kv: kv[1])[0]
            guessed_lift = bool(max_z_by_geom[guessed_grab_geom] >= DEFAULT_LIFT_THRESHOLD)
            print(
                f"[EXECUTING] 完成: executed_steps={executed_steps}/{episode_len}, return={episode_return:.2f}, "
                f"guessed_grab={guessed_grab_geom}, lifted={guessed_lift}"
            )
            print("[IDLE] 返回等待指令。")


if __name__ == "__main__":
    main()
