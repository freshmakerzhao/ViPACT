import argparse
import glob
import json
import os
import pickle
import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from einops import rearrange

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from constants import DT, get_sim_task_config, get_training_config, load_config
from policy import ACTPolicy
from sim_env import BOX_POSE, make_sim_env
from utils import sample_complex_scene_pose_eval, set_seed
from visualize_episodes import save_videos


EXPECTED_TASK = "sim_lifting_cube_with_complex_scene_scripted"
EXPECTED_EQUIPMENT = "fairino5_single"
EXPECTED_POLICY = "ACT"


def build_act_policy(policy_config: Dict):
    argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0]]
    policy = ACTPolicy(policy_config)
    sys.argv = argv_backup
    return policy


def _safe_name2id(model, name: str, obj_type: str):
    try:
        return model.name2id(name, obj_type)
    except Exception:
        # dm_control may raise wrapper.core.Error for non-existent names.
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


def build_mask_for_single_geom(physics, camera_names: List[str], geom_name: str, height=480, width=640):
    geom_id = physics.model.name2id(geom_name, "geom")
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    static_mask_dict = {}
    for cam_name in camera_names:
        seg = physics.render(height=height, width=width, camera_id=cam_name, segmentation=True)
        objid = seg[..., 0]
        objtype = seg[..., 1]
        mask = ((objid == geom_id) & (objtype == geom_type)).astype(np.float32)
        if mask.sum() < 1:
            raise RuntimeError(
                f"Empty mask for geom={geom_name}, camera={cam_name}. "
                "Try another camera/seed."
            )
        static_mask_dict[cam_name] = mask
    return static_mask_dict


def apply_mask_ablation(static_mask_dict: Dict[str, np.ndarray], mode: str, seed: int):
    if mode == "oracle":
        return static_mask_dict
    if mode == "zero":
        return {k: np.zeros_like(v, dtype=np.float32) for k, v in static_mask_dict.items()}
    if mode == "random":
        rng = np.random.default_rng(seed)
        return {k: (rng.random(v.shape) > 0.5).astype(np.float32) for k, v in static_mask_dict.items()}
    raise ValueError(f"Unsupported mask_ablation={mode}")


def build_policy_image(ts, camera_names: List[str], device: torch.device, static_mask_dict: Dict[str, np.ndarray]):
    curr_images = []
    for cam_name in camera_names:
        rgb = rearrange(ts.observation["images"][cam_name], "h w c -> c h w").astype(np.float32) / 255.0
        mask = static_mask_dict[cam_name].astype(np.float32)[None, ...]
        curr_images.append(np.concatenate([rgb, mask], axis=0))
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image).float().to(device).unsqueeze(0)


def resolve_ckpt_path(ckpt_dir: str, ckpt_arg: str):
    if ckpt_arg.strip():
        return ckpt_arg.strip()

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
    latest = epoch_ckpts[-1]
    print(f"[WARN] policy_best.ckpt not found, fallback to latest epoch ckpt: {latest}")
    return latest


def save_debug_images(
    physics,
    rgb_image: np.ndarray,
    debug_camera: str,
    target_geom_names: List[str],
    selected_target_geom: str,
    target_mask: np.ndarray,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    seg = physics.render(height=rgb_image.shape[0], width=rgb_image.shape[1], camera_id=debug_camera, segmentation=True)
    objid = seg[..., 0]
    objtype = seg[..., 1]
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.axis("off")
    for idx, geom_name in enumerate(target_geom_names):
        geom_id = physics.model.name2id(geom_name, "geom")
        geom_mask = (objid == geom_id) & (objtype == geom_type)
        if geom_mask.sum() < 1:
            continue
        ys, xs = np.where(geom_mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        color = "lime" if geom_name == selected_target_geom else "yellow"
        plt.text(
            cx,
            cy,
            str(idx),
            color=color,
            fontsize=16,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.7, pad=1.5),
        )
    numbered_path = os.path.join(output_dir, f"{debug_camera}_numbered_targets.png")
    fig.tight_layout()
    fig.savefig(numbered_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    mask_path = os.path.join(output_dir, f"{debug_camera}_target_mask.png")
    plt.imsave(mask_path, target_mask, cmap="gray", vmin=0.0, vmax=1.0)

    overlay = rgb_image.astype(np.float32).copy()
    active = target_mask > 0.5
    overlay[active] = 0.65 * overlay[active] + 0.35 * np.array([0, 255, 0], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_path = os.path.join(output_dir, f"{debug_camera}_oracle_overlay.png")
    plt.imsave(overlay_path, overlay)

    return {
        "numbered_image": numbered_path,
        "target_mask": mask_path,
        "overlay_image": overlay_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Single-episode target-by-id demo (fixed: ACT + fairino5_single + complex lifting)."
    )
    parser.add_argument("--config", type=str, required=True, help="Config path, usually 03_eval.yaml")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--target-id", type=int, required=True, help="Target index printed in numbered image")
    parser.add_argument("--episode-id", type=int, default=0, help="Seed offset for reproducible scene")
    parser.add_argument("--mask-ablation", type=str, default="oracle", choices=["oracle", "zero", "random"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--debug-camera", type=str, default="top")
    parser.add_argument("--output-dir", type=str, default="./notes/target_demo")
    parser.add_argument("--lift-threshold", type=float, default=0.09)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    yaml_cfg = load_config(args.config)
    train_cfg = get_training_config(args.config)
    task_name = yaml_cfg.get("task", {}).get("name", "")
    equipment_model = yaml_cfg.get("equipment", {}).get("model", "")
    policy_class = train_cfg.get("policy_class", "")

    if task_name != EXPECTED_TASK:
        raise ValueError(f"Only supports task={EXPECTED_TASK}, got {task_name}")
    if equipment_model != EXPECTED_EQUIPMENT:
        raise ValueError(f"Only supports equipment={EXPECTED_EQUIPMENT}, got {equipment_model}")
    if policy_class != EXPECTED_POLICY:
        raise ValueError(f"Only supports policy_class={EXPECTED_POLICY}, got {policy_class}")
    if not bool(train_cfg.get("use_mask_conditioning", False)):
        raise ValueError("This demo requires vipact.use_mask_conditioning=true")

    task_cfg = get_sim_task_config(task_name, args.config)
    camera_names = task_cfg["camera_names"]
    if args.debug_camera not in camera_names:
        raise ValueError(f"debug-camera must be in camera_names={camera_names}")

    episode_len = int(task_cfg["episode_len"]) if args.max_steps <= 0 else int(args.max_steps)
    state_dim = 7  # fixed for fairino5 single-arm lifting task
    ckpt_dir = train_cfg.get("ckpt_dir", "./ckpts")
    ckpt_path = resolve_ckpt_path(ckpt_dir, args.ckpt)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"dataset_stats.pkl not found: {stats_path}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seed = int(train_cfg.get("seed", 1000)) + int(args.episode_id)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_cfg = {
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
        "use_mask_conditioning": True,
        "image_channels": 4,
    }

    policy = build_act_policy(policy_cfg)
    load_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(load_status)
    policy.to(device)
    policy.eval()
    print(f"Loaded ckpt: {ckpt_path}")
    print(f"Device: {device}")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    BOX_POSE[0] = sample_complex_scene_pose_eval()
    env = make_sim_env(task_name, equipment_model=equipment_model)
    ts = env.reset()

    target_geom_names = list_target_geom_names(env._physics)
    print("Target id mapping:")
    for i, name in enumerate(target_geom_names):
        print(f"  [{i}] {name}")
    if args.target_id < 0 or args.target_id >= len(target_geom_names):
        raise ValueError(f"target-id out of range [0, {len(target_geom_names) - 1}]")
    selected_target_geom = target_geom_names[args.target_id]

    static_mask_dict = build_mask_for_single_geom(env._physics, camera_names, selected_target_geom)
    static_mask_dict = apply_mask_ablation(static_mask_dict, mode=args.mask_ablation, seed=seed)

    debug_paths = {}
    if args.mask_ablation == "oracle":
        debug_paths = save_debug_images(
            physics=env._physics,
            rgb_image=ts.observation["images"][args.debug_camera],
            debug_camera=args.debug_camera,
            target_geom_names=target_geom_names,
            selected_target_geom=selected_target_geom,
            target_mask=static_mask_dict[args.debug_camera],
            output_dir=output_dir,
        )
        print("Saved debug images:")
        for k, v in debug_paths.items():
            print(f"  {k}: {v}")

    query_frequency = int(policy_cfg["num_queries"])
    temporal_agg = bool(train_cfg.get("temporal_agg", False))
    if temporal_agg:
        query_frequency = 1
        num_queries = int(policy_cfg["num_queries"])
        all_time_actions = torch.zeros([episode_len, episode_len + num_queries, state_dim], device=device)

    max_z_by_geom = {name: -1e9 for name in target_geom_names}
    rewards = []
    image_list = []

    if args.render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation["images"][args.debug_camera])
        plt.ion()

    with torch.inference_mode():
        for t in range(episode_len):
            if args.render:
                plt_img.set_data(ts.observation["images"][args.debug_camera])
                plt.pause(DT)

            image_list.append(ts.observation["images"])
            qpos_numpy = np.asarray(ts.observation["qpos"])
            qpos = torch.from_numpy(pre_process(qpos_numpy)).float().to(device).unsqueeze(0)
            curr_image = build_policy_image(ts, camera_names, device, static_mask_dict)

            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)
            if temporal_agg:
                all_time_actions[[t], t : t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]

            action = post_process(raw_action.squeeze(0).cpu().numpy())
            ts = env.step(action)
            rewards.append(float(ts.reward))

            heights = get_object_heights(env._physics, target_geom_names)
            for geom_name, z in heights.items():
                max_z_by_geom[geom_name] = max(max_z_by_geom[geom_name], z)

    if args.render:
        plt.close()

    if args.save_video:
        video_path = os.path.join(
            output_dir,
            f"demo_target_{args.target_id}_{selected_target_geom}_seed_{seed}.mp4",
        )
        save_videos(image_list, DT, video_path=video_path)
        print(f"Saved rollout video: {video_path}")
    else:
        video_path = ""

    episode_return = float(np.sum(np.asarray(rewards)))
    lifted_flags = {k: (v >= args.lift_threshold) for k, v in max_z_by_geom.items()}
    guessed_grab_geom = max(max_z_by_geom.items(), key=lambda kv: kv[1])[0]
    target_lifted = bool(lifted_flags[selected_target_geom])
    lifted_any = any(lifted_flags.values())

    print("\n=== Demo Summary ===")
    print(f"seed: {seed}")
    print(f"target_id: {args.target_id}")
    print(f"target_geom: {selected_target_geom}")
    print(f"mask_ablation: {args.mask_ablation}")
    print(f"episode_return: {episode_return:.2f}")
    print("max_z_by_geom:")
    for geom_name in target_geom_names:
        print(f"  - {geom_name}: {max_z_by_geom[geom_name]:.4f} lifted={lifted_flags[geom_name]}")
    print(f"guessed_grab_geom: {guessed_grab_geom}")
    print(f"target_lifted: {target_lifted}")
    print(f"lifted_any: {lifted_any}")

    summary = {
        "task_name": task_name,
        "equipment_model": equipment_model,
        "policy_class": EXPECTED_POLICY,
        "seed": seed,
        "ckpt_path": ckpt_path,
        "stats_path": stats_path,
        "camera_names": camera_names,
        "debug_camera": args.debug_camera,
        "target_id": int(args.target_id),
        "target_geom": selected_target_geom,
        "target_id_mapping": {str(i): name for i, name in enumerate(target_geom_names)},
        "mask_ablation": args.mask_ablation,
        "episode_len": int(episode_len),
        "episode_return": episode_return,
        "lift_threshold": float(args.lift_threshold),
        "max_z_by_geom": {k: float(v) for k, v in max_z_by_geom.items()},
        "lifted_flags": {k: bool(v) for k, v in lifted_flags.items()},
        "guessed_grab_geom": guessed_grab_geom,
        "target_lifted": bool(target_lifted),
        "lifted_any": bool(lifted_any),
        "video_path": video_path,
        "debug_paths": debug_paths,
    }
    summary_path = os.path.join(
        output_dir,
        f"summary_target_{args.target_id}_{selected_target_geom}_seed_{seed}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
