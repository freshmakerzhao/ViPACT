#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multimodal_brain.llm_parser.qwen_parser import QwenInstructionParser
from multimodal_brain.llm_parser.target_selector import find_target_location


def main():
    parser = argparse.ArgumentParser(description="LLM parse instruction + spatial select target from DINO results.")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--dino-json", type=str, required=True, help="Path to DINO output JSON")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--tolerance", type=float, default=20.0)
    args = parser.parse_args()

    with open(args.dino_json, "r", encoding="utf-8") as f:
        dino_json = json.load(f)

    api_key = (
        str(args.api_key).strip()
        or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("QWEN_API_KEY", "").strip()
    )

    parser_obj = QwenInstructionParser(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
    )
    parsed = parser_obj.parse(args.instruction).to_dict()
    selected = find_target_location(
        dino_json=dino_json,
        rules=parsed,
        tolerance=float(args.tolerance),
    )

    output = {
        "instruction": str(args.instruction),
        "llm_mode": "qwen_api" if bool(api_key) else "heuristic_fallback",
        "parsed_rules": parsed,
        "selection_result": selected,
    }

    if args.output_json.strip():
        out_path = Path(args.output_json).resolve()
    else:
        out_path = Path(args.dino_json).resolve().parent / "llm_parse_and_select_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
