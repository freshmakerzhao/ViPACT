#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from brain_pipeline import LanguageToMaskPipeline
from constants import get_equipment_model, get_sim_task_config, get_training_config, load_config
from sim_env import BOX_POSE, make_sim_env
from utils import sample_complex_scene_pose_eval, set_seed


EXPECTED_TASK = "sim_lifting_cube_with_complex_scene_scripted"
EXPECTED_EQUIPMENT = "fairino5_single"


def _draw_boxes(rgb: np.ndarray, candidates: list[dict], selected: dict | None) -> Image.Image:
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for c in candidates:
        x0, y0, x1, y1 = c["box_xyxy"]
        color = (255, 200, 0)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        text = f"#{c['rank']} s={c['score']:.3f} a={c['area_ratio']:.3f}"
        draw.text((x0 + 2, max(0, y0 - 12)), text, fill=color)

    if selected is not None:
        x0, y0, x1, y1 = selected["box_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=4)
        draw.text((x0 + 2, y0 + 2), "SELECTED", fill=(0, 255, 0))
    return img


def _choose_candidate(candidates: list[dict], max_box_area_ratio: float) -> dict | None:
    if len(candidates) == 0:
        return None
    filtered = [c for c in candidates if float(c.get("area_ratio", 1.0)) <= float(max_box_area_ratio)]
    if len(filtered) > 0:
        return filtered[0]
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="GroundingDINO-tiny only debug on first cockpit frame.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--text-prompt", type=str, required=True, help='e.g. "red cube"')
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--camera", type=str, default="cockpit")
    parser.add_argument("--output-dir", type=str, default="./notes/dino_tiny_debug")
    parser.add_argument("--grounding-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.12)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    task_name = yaml_cfg.get("task", {}).get("name", "")
    equipment_model = get_equipment_model(args.config)
    train_cfg = get_training_config(args.config)

    if task_name != EXPECTED_TASK:
        raise ValueError(f"Only supports task={EXPECTED_TASK}, got {task_name}")
    if equipment_model != EXPECTED_EQUIPMENT:
        raise ValueError(f"Only supports equipment={EXPECTED_EQUIPMENT}, got {equipment_model}")

    task_cfg = get_sim_task_config(task_name, args.config)
    camera_names = task_cfg["camera_names"]
    if args.camera not in camera_names:
        raise ValueError(f"--camera={args.camera} not in camera_names={camera_names}")

    seed = int(train_cfg.get("seed", 1000)) + int(args.episode_id)
    set_seed(seed)

    # 与训练评估一致：由采样函数给定布局
    init_pose = np.asarray(sample_complex_scene_pose_eval(), dtype=np.float64).copy()
    BOX_POSE[0] = init_pose.copy()
    env = make_sim_env(task_name, equipment_model=equipment_model)
    ts = env.reset()
    rgb = ts.observation["images"][args.camera]

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = LanguageToMaskPipeline(
        device=device,
        grounding_model_id=str(args.grounding_model_id),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        enable_sam=False,
    )
    det = pipeline.detect_boxes_with_debug(rgb, str(args.text_prompt), top_k=int(args.top_k))

    candidates = det.get("candidates", [])
    selected = _choose_candidate(candidates, max_box_area_ratio=float(args.max_box_area_ratio))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = out_dir / f"dino_tiny_ep{args.episode_id}_{args.camera}_rgb.png"
    overlay_path = out_dir / f"dino_tiny_ep{args.episode_id}_{args.camera}_boxes.png"
    summary_path = out_dir / f"summary_dino_tiny_ep{args.episode_id}.json"

    Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
    overlay_img = _draw_boxes(rgb, candidates, selected)
    overlay_img.save(overlay_path)

    summary = {
        "task_name": task_name,
        "equipment_model": equipment_model,
        "episode_id": int(args.episode_id),
        "seed": int(seed),
        "camera": args.camera,
        "text_prompt": args.text_prompt,
        "grounding_model_id": args.grounding_model_id,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "top_k": int(args.top_k),
        "max_box_area_ratio": float(args.max_box_area_ratio),
        "det_debug": det,
        "selected_candidate": selected,
        "paths": {
            "rgb": str(rgb_path),
            "boxes_overlay": str(overlay_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved rgb: {rgb_path}")
    print(f"Saved boxes overlay: {overlay_path}")
    print(f"Saved summary: {summary_path}")
    if selected is None:
        print("No candidate selected.")
    else:
        print(f"Selected box: {selected['box_xyxy']} score={selected['score']:.4f} area_ratio={selected['area_ratio']:.4f}")


if __name__ == "__main__":
    main()
