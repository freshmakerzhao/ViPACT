#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from multimodal_brain.mask_generator.sam2_box_mask import Sam2BoxMaskGenerator, mask_overlay


def _read_box_from_llm_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    sel = obj.get("selection_result", {}).get("selected", None)
    if not isinstance(sel, dict):
        raise ValueError("llm json has no selection_result.selected")
    box = sel.get("box_xyxy", None)
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("selection_result.selected.box_xyxy is invalid")
    return [float(v) for v in box], obj


def _draw_box(rgb: np.ndarray, box_xyxy):
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=3)
    draw.text((x0 + 2, y0 + 2), "BOX", fill=(0, 255, 0))
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser(description="Generate SAM2 mask from llm_parse_and_select_result.json box.")
    parser.add_argument("--llm-json", type=str, required=True, help="Path to llm_parse_and_select_result.json")
    parser.add_argument("--image", type=str, required=True, help="Input RGB image path")
    parser.add_argument("--output-mask", type=str, default="", help="Output binary mask png path")
    parser.add_argument("--output-overlay", type=str, default="", help="Output overlay image path")
    parser.add_argument("--output-box", type=str, default="", help="Output image with selected bbox")
    parser.add_argument("--output-summary", type=str, default="", help="Output summary json path")
    parser.add_argument("--model-id", type=str, default="facebook/sam2-hiera-small")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    box_xyxy, llm_obj = _read_box_from_llm_json(args.llm_json)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))

    generator = Sam2BoxMaskGenerator(model_id=args.model_id, device=args.device)
    mask_u8 = generator.predict_mask(rgb, box_xyxy=box_xyxy).astype(np.uint8)

    overlay = mask_overlay(rgb, mask_u8)
    box_img = _draw_box(rgb, box_xyxy)

    image_path = Path(args.image).resolve()
    out_dir = image_path.parent
    mask_path = Path(args.output_mask).resolve() if args.output_mask else out_dir / "sam2_mask.png"
    overlay_path = Path(args.output_overlay).resolve() if args.output_overlay else out_dir / "sam2_overlay.png"
    box_path = Path(args.output_box).resolve() if args.output_box else out_dir / "selected_box.png"
    summary_path = Path(args.output_summary).resolve() if args.output_summary else out_dir / "sam2_mask_summary.json"

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_u8 * 255).astype(np.uint8), mode="L").save(mask_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(box_img).save(box_path)

    summary = {
        "llm_json": str(Path(args.llm_json).resolve()),
        "image": str(image_path),
        "model_id": str(args.model_id),
        "device": str(args.device),
        "box_xyxy": [float(v) for v in box_xyxy],
        "mask_sum": int(mask_u8.sum()),
        "paths": {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "box": str(box_path),
        },
        "llm_mode": llm_obj.get("llm_mode", ""),
        "parsed_rules": llm_obj.get("parsed_rules", {}),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved box image: {box_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

