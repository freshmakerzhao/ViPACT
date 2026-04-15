import os
import csv
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def _overlay_mask(rgb_uint8, mask_binary):
    overlay = rgb_uint8.astype(np.float32) / 255.0
    # Yellow overlay is easy to see on dark/gray tabletop scenes.
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    alpha = 0.35
    m = mask_binary.astype(bool)
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * yellow
    return np.clip(overlay, 0.0, 1.0)


def _safe_percentile(arr, p):
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, p))


def main():
    parser = argparse.ArgumentParser(
        description="Audit /observations/static_masks quality and export RGB/mask/overlay samples."
    )
    parser.add_argument(
        "--episode_dir",
        type=str,
        required=True,
        help="Directory containing episode_*.hdf5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save audit outputs",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="top",
        help="Camera key under /observations/images and /observations/static_masks",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="How many random episode/timestep samples to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sample selection",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    episode_paths = sorted(glob.glob(os.path.join(args.episode_dir, "episode_*.hdf5")))
    if len(episode_paths) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found under: {args.episode_dir}")

    key_mask = f"/observations/static_masks/{args.camera}"
    key_img = f"/observations/images/{args.camera}"

    missing_mask = []
    empty_mask = []
    ratios = []
    valid_episode_paths = []

    for p in episode_paths:
        with h5py.File(p, "r") as f:
            if key_mask not in f:
                missing_mask.append(os.path.basename(p))
                continue
            m = f[key_mask][()]
            r = float((m > 0).mean())
            ratios.append(r)
            valid_episode_paths.append(p)
            if r == 0.0:
                empty_mask.append(os.path.basename(p))

    # Save global statistics.
    summary_path = os.path.join(args.output_dir, "mask_audit_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fw:
        fw.write(f"episode_dir={args.episode_dir}\n")
        fw.write(f"camera={args.camera}\n")
        fw.write(f"episodes_total={len(episode_paths)}\n")
        fw.write(f"episodes_with_mask={len(valid_episode_paths)}\n")
        fw.write(f"missing_mask_count={len(missing_mask)}\n")
        fw.write(f"empty_mask_count={len(empty_mask)}\n")
        if len(ratios) > 0:
            arr = np.array(ratios, dtype=np.float64)
            fw.write(
                "mask_ratio_min_mean_max="
                f"{arr.min():.6f},{arr.mean():.6f},{arr.max():.6f}\n"
            )
            fw.write(
                "mask_ratio_p10_p50_p90="
                f"{_safe_percentile(arr, 10):.6f},"
                f"{_safe_percentile(arr, 50):.6f},"
                f"{_safe_percentile(arr, 90):.6f}\n"
            )
        if len(missing_mask) > 0:
            fw.write("missing_mask_episodes=\n")
            for name in missing_mask:
                fw.write(f"  {name}\n")
        if len(empty_mask) > 0:
            fw.write("empty_mask_episodes=\n")
            for name in empty_mask:
                fw.write(f"  {name}\n")

    # Export random visual samples.
    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(valid_episode_paths))
    chosen_ids = rng.choice(len(valid_episode_paths), size=n, replace=False) if n > 0 else []

    manifest_path = os.path.join(args.output_dir, "sample_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["sample_idx", "episode_file", "timestep", "mask_ratio"],
        )
        writer.writeheader()

        for sample_idx, idx in enumerate(chosen_ids):
            episode_path = valid_episode_paths[int(idx)]
            episode_file = os.path.basename(episode_path)
            with h5py.File(episode_path, "r") as f:
                if key_img not in f:
                    continue
                imgs = f[key_img]
                t = int(rng.integers(0, imgs.shape[0]))
                rgb = imgs[t][()]
                mask = (f[key_mask][()] > 0).astype(np.uint8)
                ratio = float(mask.mean())

            overlay = _overlay_mask(rgb, mask)
            stem = f"{sample_idx:02d}_{episode_file.replace('.hdf5', '')}_t{t}"
            rgb_path = os.path.join(args.output_dir, f"{stem}_rgb.png")
            mask_path = os.path.join(args.output_dir, f"{stem}_mask.png")
            overlay_path = os.path.join(args.output_dir, f"{stem}_overlay.png")
            plt.imsave(rgb_path, rgb)
            plt.imsave(mask_path, mask, cmap="gray", vmin=0, vmax=1)
            plt.imsave(overlay_path, overlay)

            writer.writerow(
                {
                    "sample_idx": sample_idx,
                    "episode_file": episode_file,
                    "timestep": t,
                    "mask_ratio": f"{ratio:.6f}",
                }
            )

    print(f"Saved summary: {summary_path}")
    print(f"Saved sample manifest: {manifest_path}")
    print(f"Saved visual samples to: {args.output_dir}")


if __name__ == "__main__":
    main()
