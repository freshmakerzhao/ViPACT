#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def _load_detections(dino_json_path: str) -> List[Dict]:
    with open(dino_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    detections = obj.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("Invalid dino json: `detections` must be a list.")
    return detections


def _draw_boxes(
    image: Image.Image,
    detections: List[Dict],
    top_k: int = 20,
    min_score: float = 0.0,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    shown = 0
    for det in detections:
        if shown >= int(top_k):
            break
        score = float(det.get("score", 0.0))
        if score < float(min_score):
            continue
        box = det.get("box_xyxy", [0, 0, 0, 0])
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in box]
        label = str(det.get("label", ""))
        rank = int(det.get("rank", -1))

        color = (255, 0, 0) if rank == 0 else (255, 215, 0)
        width = 4 if rank == 0 else 2
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        text = f"#{rank} s={score:.3f} {label}".strip()
        draw.text((x0 + 2, max(0, y0 - 12)), text, fill=color)
        shown += 1
    return image


def main():
    parser = argparse.ArgumentParser(description="Visualize DINO detections on image.")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--dino-json", type=str, required=True, help="DINO result json (with detections)")
    parser.add_argument("--output-image", type=str, default="", help="Output visualization image path")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    detections = _load_detections(args.dino_json)
    vis = _draw_boxes(image=image, detections=detections, top_k=int(args.top_k), min_score=float(args.min_score))

    if args.output_image.strip():
        out_path = Path(args.output_image).resolve()
    else:
        out_path = Path(args.image).resolve().parent / "dino_detection_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

