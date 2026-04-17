#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from multimodal_brain.dino_detector import GroundingDinoDetector
from multimodal_brain.llm_parser import QwenInstructionParser, find_target_location
from multimodal_brain.mask_generator import Sam2BoxMaskGenerator
from pipeline import BrainToPolicyPipeline


def _print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _get_llm_api_key(cli_key: str) -> str:
    return (
        str(cli_key).strip()
        or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("QWEN_API_KEY", "").strip()
    )


def cmd_llm(args):
    parser = QwenInstructionParser(
        api_key=_get_llm_api_key(args.api_key),
        base_url=args.base_url,
        model=args.model,
    )
    out = parser.parse_typed(args.instruction)
    result = {
        "instruction": args.instruction,
        "parsed": out.raw if out.raw else {
            "text_prompt": out.text_prompt,
            "axis": out.axis,
            "reverse": out.reverse,
            "target_rank": out.target_rank,
        },
    }
    _print_json(result)


def cmd_dino(args):
    detector = GroundingDinoDetector(
        model_id=args.model_id,
        device=args.device,
        text_threshold=float(args.text_threshold),
    )
    result = detector.detect_boxes(
        image=args.image,
        text_query=args.query,
        confidence_threshold=float(args.confidence),
    )
    _print_json(result)
    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")


def _load_box(args):
    if args.box:
        vals = [float(x.strip()) for x in args.box.split(",")]
        if len(vals) != 4:
            raise ValueError("--box must be x0,y0,x1,y1")
        return vals
    if args.box_json:
        obj = json.loads(Path(args.box_json).read_text(encoding="utf-8"))
        sel = obj.get("selection_result", {}).get("selected", obj.get("selected", {}))
        box = sel.get("box_xyxy", None) if isinstance(sel, dict) else None
        if not box or len(box) != 4:
            raise ValueError("box_json has no selected.box_xyxy")
        return [float(v) for v in box]
    raise ValueError("provide --box or --box-json")


def cmd_sam(args):
    box = _load_box(args)
    gen = Sam2BoxMaskGenerator(model_id=args.model_id, device=args.device)
    mask_u8 = gen.predict_mask(args.image, box_xyxy=box).astype(np.uint8)
    out_mask = Path(args.output_mask).resolve()
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_u8 > 0).astype(np.uint8) * 255, mode="L").save(out_mask)
    result = {"box_xyxy": box, "mask_sum": int(mask_u8.sum()), "mask_path": str(out_mask)}
    _print_json(result)


def cmd_act(args):
    pipe = BrainToPolicyPipeline(config_path=args.config, output_dir=args.output_dir, mask_camera=args.mask_camera)
    summary = pipe.run_with_manual_mask(
        mask_path=args.mask_path,
        episode_id=int(args.episode_id),
        target_id=int(args.target_id),
        max_steps=int(args.max_steps),
        save_video=bool(args.save_video),
        onscreen_render=bool(args.onscreen_render),
        onscreen_cam=args.onscreen_cam,
    )
    _print_json(summary)


def cmd_full(args):
    pipe = BrainToPolicyPipeline(config_path=args.config, output_dir=args.output_dir, mask_camera=args.mask_camera)
    summary = pipe.run_with_instruction(
        instruction=args.instruction,
        episode_id=int(args.episode_id),
        target_id=int(args.target_id),
        max_steps=int(args.max_steps),
        save_video=bool(args.save_video),
        llm_api_key=_get_llm_api_key(args.api_key),
        llm_base_url=args.base_url,
        llm_model=args.model,
        grounding_model_id=args.grounding_model_id,
        sam_model_id=args.sam_model_id,
        confidence_threshold=float(args.confidence),
        tolerance=float(args.tolerance),
        device=args.device,
        onscreen_render=bool(args.onscreen_render),
        onscreen_cam=args.onscreen_cam,
    )
    _print_json(summary)


def cmd_interactive(args):
    cmd = [
        sys.executable,
        "interactive_app/passive_viewer_console.py",
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--episode-id-base",
        str(args.episode_id_base),
        "--max-steps",
        str(args.max_steps),
        "--sleep-dt",
        str(args.sleep_dt),
        "--device",
        args.device,
        "--llm-base-url",
        args.base_url,
        "--llm-model",
        args.model,
        "--grounding-model-id",
        args.grounding_model_id,
        "--sam-model-id",
        args.sam_model_id,
        "--confidence-threshold",
        str(args.confidence),
        "--tolerance",
        str(args.tolerance),
    ]
    subprocess.run(cmd, check=True)


def build_parser():
    p = argparse.ArgumentParser(description="ViPACT unified console")
    sub = p.add_subparsers(dest="cmd", required=True)

    # llm
    llm = sub.add_parser("llm", help="LLM parse instruction only")
    llm.add_argument("--instruction", required=True)
    llm.add_argument("--api-key", default="")
    llm.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm.add_argument("--model", default="qwen-plus")
    llm.set_defaults(func=cmd_llm)

    # dino
    dino = sub.add_parser("dino", help="GroundingDINO detect")
    dino.add_argument("--image", required=True)
    dino.add_argument("--query", required=True)
    dino.add_argument("--confidence", type=float, default=0.5)
    dino.add_argument("--text-threshold", type=float, default=0.25)
    dino.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    dino.add_argument("--device", default="cuda")
    dino.add_argument("--output-json", default="")
    dino.set_defaults(func=cmd_dino)

    # sam
    sam = sub.add_parser("sam", help="SAM2 mask from box")
    sam.add_argument("--image", required=True)
    sam.add_argument("--box", default="")
    sam.add_argument("--box-json", default="")
    sam.add_argument("--model-id", default="facebook/sam2-hiera-small")
    sam.add_argument("--device", default="cuda")
    sam.add_argument("--output-mask", required=True)
    sam.set_defaults(func=cmd_sam)

    # act
    act = sub.add_parser("act", help="ACT inference with manual mask")
    act.add_argument("--config", required=True)
    act.add_argument("--mask-path", required=True)
    act.add_argument("--output-dir", default="notes/vipact_console_runs")
    act.add_argument("--mask-camera", default="cockpit")
    act.add_argument("--episode-id", type=int, default=0)
    act.add_argument("--target-id", type=int, default=-1)
    act.add_argument("--max-steps", type=int, default=-1)
    act.add_argument("--save-video", action="store_true")
    act.add_argument("--onscreen-render", action="store_true")
    act.add_argument("--onscreen-cam", default="cockpit")
    act.set_defaults(func=cmd_act)

    # full
    full = sub.add_parser("full", help="One-line full pipeline: instruction -> mask -> ACT")
    full.add_argument("--config", required=True)
    full.add_argument("--instruction", required=True)
    full.add_argument("--output-dir", default="notes/vipact_console_runs")
    full.add_argument("--mask-camera", default="cockpit")
    full.add_argument("--episode-id", type=int, default=0)
    full.add_argument("--target-id", type=int, default=-1)
    full.add_argument("--max-steps", type=int, default=-1)
    full.add_argument("--save-video", action="store_true")
    full.add_argument("--onscreen-render", action="store_true")
    full.add_argument("--onscreen-cam", default="cockpit")
    full.add_argument("--api-key", default="")
    full.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    full.add_argument("--model", default="qwen-plus")
    full.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    full.add_argument("--sam-model-id", default="facebook/sam2-hiera-small")
    full.add_argument("--confidence", type=float, default=0.5)
    full.add_argument("--tolerance", type=float, default=20.0)
    full.add_argument("--device", default="cuda")
    full.set_defaults(func=cmd_full)

    # interactive
    inter = sub.add_parser("interactive", help="Interactive full pipeline (input + passive viewer)")
    inter.add_argument("--config", required=True)
    inter.add_argument("--output-dir", default="notes/passive_viewer_runs")
    inter.add_argument("--episode-id-base", type=int, default=0)
    inter.add_argument("--max-steps", type=int, default=-1)
    inter.add_argument("--sleep-dt", type=float, default=0.02)
    inter.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    inter.add_argument("--model", default="qwen-plus")
    inter.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    inter.add_argument("--sam-model-id", default="facebook/sam2-hiera-small")
    inter.add_argument("--confidence", type=float, default=0.5)
    inter.add_argument("--tolerance", type=float, default=20.0)
    inter.add_argument("--device", default="cuda")
    inter.set_defaults(func=cmd_interactive)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

