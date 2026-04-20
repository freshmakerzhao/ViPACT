#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_brain.dino_detector import GroundingDinoDetector


def main():
    parser = argparse.ArgumentParser(description="Detect boxes with GroundingDINO.")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--query", type=str, required=True, help='Text query, e.g. "a red cube"')
    parser.add_argument("--confidence", type=float, default=0.5, help="Return boxes with score >= confidence")
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default="", help="Optional output path")
    args = parser.parse_args()

    detector = GroundingDinoDetector(
        model_id=str(args.model_id),
        device=str(args.device),
        text_threshold=float(args.text_threshold),
    )
    result = detector.detect_boxes(
        image=str(args.image),
        text_query=str(args.query),
        confidence_threshold=float(args.confidence),
    )

    if args.output_json.strip():
        out_path = Path(args.output_json).resolve()
    else:
        out_path = Path(args.image).resolve().parent / "dino_detection_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
