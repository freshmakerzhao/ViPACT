#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from brain import LanguageToMaskBrain
from constants import get_equipment_model, get_sim_task_config, get_training_config, load_config
from perception import save_mask_debug_images
from sim_env import BOX_POSE, make_sim_env
from utils import sample_complex_scene_pose_eval, set_seed


EXPECTED_TASK = "sim_lifting_cube_with_complex_scene_scripted"
EXPECTED_EQUIPMENT = "fairino5_single"
MASK_CAMERA = "cockpit"


def main():
    parser = argparse.ArgumentParser(
        description="One-shot brain demo: first cockpit frame + language -> binary mask."
    )
    parser.add_argument("--config", type=str, required=True, help="Config path, usually 03_eval.yaml")
    parser.add_argument("--instruction", type=str, required=True, help='Language instruction, e.g. "抓右侧第二个方块"')
    parser.add_argument("--episode-id", type=int, default=0, help="Seed offset for scene sampling")
    parser.add_argument("--output-dir", type=str, default="./notes/language_mask_once")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--grounding-model-id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-id", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    train_cfg = get_training_config(args.config)
    task_name = yaml_cfg.get("task", {}).get("name", "")
    equipment_model = get_equipment_model(args.config)

    if task_name != EXPECTED_TASK:
        raise ValueError(f"Only supports task={EXPECTED_TASK}, got {task_name}")
    if equipment_model != EXPECTED_EQUIPMENT:
        raise ValueError(f"Only supports equipment={EXPECTED_EQUIPMENT}, got {equipment_model}")

    task_cfg = get_sim_task_config(task_name, args.config)
    camera_names = task_cfg["camera_names"]
    if MASK_CAMERA not in camera_names:
        raise ValueError(f"camera_names={camera_names} has no '{MASK_CAMERA}', cannot run this demo")

    seed = int(train_cfg.get("seed", 1000)) + int(args.episode_id)
    set_seed(seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 固定初始布局，便于后续把同一布局喂给 ACT 做手动 mask 测试。
    init_pose = np.asarray(sample_complex_scene_pose_eval(), dtype=np.float64).copy()
    BOX_POSE[0] = init_pose.copy()
    env = make_sim_env(task_name, equipment_model=equipment_model)
    ts = env.reset()
    cockpit_rgb = ts.observation["images"][MASK_CAMERA]

    rgb_path = output_dir / f"ep{args.episode_id}_{MASK_CAMERA}_rgb.png"
    Image.fromarray(cockpit_rgb.astype(np.uint8)).save(rgb_path)

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    brain = LanguageToMaskBrain(
        device=device,
        grounding_model_id=str(args.grounding_model_id),
        sam_model_id=str(args.sam_model_id),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
    )
    mask_u8, mask_debug = brain.predict_mask_with_debug(cockpit_rgb, str(args.instruction))
    mask = mask_u8.astype(np.float32)

    prefix = str(output_dir / f"ep{args.episode_id}_{MASK_CAMERA}_lang")
    debug_paths = save_mask_debug_images(cockpit_rgb, mask, prefix)

    # 另外输出一张明确可复用的单通道掩码图（0/255）
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    mask_path = output_dir / f"ep{args.episode_id}_{MASK_CAMERA}_mask_binary.png"
    Image.fromarray(mask_u8, mode="L").save(mask_path)

    # 记录这次场景的初始布局，给手动 ACT 测试脚本复用。
    init_pose_path = output_dir / f"ep{args.episode_id}_init_pose.json"
    with open(init_pose_path, "w", encoding="utf-8") as f:
        json.dump({"initial_box_pose": init_pose.tolist()}, f, ensure_ascii=False, indent=2)

    summary = {
        "task_name": task_name,
        "equipment_model": equipment_model,
        "episode_id": int(args.episode_id),
        "seed": int(seed),
        "instruction": str(args.instruction),
        "camera_name": MASK_CAMERA,
        "image_shape": [int(cockpit_rgb.shape[0]), int(cockpit_rgb.shape[1])],
        "device": device,
        "grounding_model_id": args.grounding_model_id,
        "sam_model_id": args.sam_model_id,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "mask_debug": mask_debug,
        "paths": {
            "rgb": str(rgb_path),
            "mask_binary": str(mask_path),
            "overlay": debug_paths.get("overlay", ""),
            "init_pose": str(init_pose_path),
        },
    }
    summary_path = output_dir / f"summary_language_to_mask_ep{args.episode_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved RGB: {rgb_path}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {debug_paths.get('overlay', '')}")
    print(f"Saved init pose: {init_pose_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
