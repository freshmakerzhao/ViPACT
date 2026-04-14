import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import load_config, get_equipment_model
from sim_env import make_sim_env, BOX_POSE
from utils import (
    sample_box_pose,
    sample_box_pose_eval,
    sample_box_pose_for_excavator,
    sample_insertion_pose,
    build_oracle_static_mask_dict,
)


def _set_reset_pose(task_name, equipment_model):
    if "sim_transfer_cube" in task_name:
        BOX_POSE[0] = sample_box_pose()
    elif "sim_insertion" in task_name:
        BOX_POSE[0] = np.concatenate(sample_insertion_pose())
    elif "sim_lifting_cube" in task_name:
        if "excavator" in equipment_model:
            BOX_POSE[0] = sample_box_pose_for_excavator()
        else:
            BOX_POSE[0] = sample_box_pose_eval()
    else:
        raise NotImplementedError(f"Unsupported task_name: {task_name}")


def _save_overlay(path, rgb_uint8, mask_float):
    overlay = rgb_uint8.astype(np.float32) / 255.0
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    alpha = 0.45
    overlay = np.where(mask_float[..., None] > 0.5, (1 - alpha) * overlay + alpha * red, overlay)
    plt.imsave(path, overlay)


def main():
    parser = argparse.ArgumentParser(description="Export oracle mask visualization from sim reset frame")
    parser.add_argument("--config", type=str, default="configs/fairino5_single/03_eval.yaml")
    parser.add_argument("--camera", type=str, default="top", help="camera name, e.g. top/angle/vis")
    parser.add_argument("--output_dir", type=str, default="notes/mask_debug")
    args = parser.parse_args()

    yaml_config = load_config(args.config)
    task_name = yaml_config.get("task", {}).get("name", "sim_lifting_cube_scripted")
    equipment_model = get_equipment_model(args.config)
    camera_names = yaml_config.get("task", {}).get("camera_names", ["top"])
    if args.camera not in camera_names:
        raise ValueError(f"Camera '{args.camera}' not in config camera_names={camera_names}")

    _set_reset_pose(task_name, equipment_model)
    env = make_sim_env(task_name, equipment_model)
    ts = env.reset()

    rgb = ts.observation["images"][args.camera]
    oracle_mask = build_oracle_static_mask_dict(env._physics, [args.camera], task_name)[args.camera]

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{task_name}_{equipment_model}_{args.camera}"

    rgb_path = os.path.join(args.output_dir, f"{prefix}_rgb.png")
    oracle_mask_path = os.path.join(args.output_dir, f"{prefix}_oracle_mask.png")
    oracle_overlay_path = os.path.join(args.output_dir, f"{prefix}_oracle_overlay.png")

    plt.imsave(rgb_path, rgb)
    plt.imsave(oracle_mask_path, oracle_mask, cmap="gray", vmin=0, vmax=1)
    _save_overlay(oracle_overlay_path, rgb, oracle_mask)

    print(f"saved_rgb={rgb_path}")
    print(f"saved_oracle_mask={oracle_mask_path}")
    print(f"saved_oracle_overlay={oracle_overlay_path}")
    print(f"oracle_mask_pixels={int((oracle_mask > 0.5).sum())}")


if __name__ == "__main__":
    main()
