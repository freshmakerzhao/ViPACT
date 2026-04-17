from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from interfaces import DetectionResult, InstructionParseResult, MaskResult, TargetSelectionResult
from multimodal_brain.dino_detector import GroundingDinoDetector
from multimodal_brain.llm_parser.qwen_parser import QwenInstructionParser
from multimodal_brain.llm_parser.target_selector import select_target_typed
from multimodal_brain.mask_generator import Sam2BoxMaskGenerator


@dataclass
class BrainPerceptionFacade:
    """
    Unified high-level entry:
    instruction + image -> parsed rules -> detections -> selected box -> mask.
    """

    llm_parser: QwenInstructionParser
    detector: GroundingDinoDetector
    mask_generator: Sam2BoxMaskGenerator

    @classmethod
    def create_default(
        cls,
        *,
        llm_api_key: str = "",
        llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_model: str = "qwen-plus",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam2-hiera-small",
        device: str = "cuda",
        text_threshold: float = 0.25,
    ) -> "BrainPerceptionFacade":
        return cls(
            llm_parser=QwenInstructionParser(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model=llm_model,
            ),
            detector=GroundingDinoDetector(
                model_id=grounding_model_id,
                device=device,
                text_threshold=float(text_threshold),
            ),
            mask_generator=Sam2BoxMaskGenerator(
                model_id=sam_model_id,
                device=device,
            ),
        )

    def run_once(
        self,
        *,
        rgb_image: np.ndarray,
        instruction: str,
        confidence_threshold: float = 0.5,
        tolerance: float = 20.0,
    ) -> Dict[str, Any]:
        parsed: InstructionParseResult = self.llm_parser.parse_typed(instruction)
        detection: DetectionResult = self.detector.detect_boxes_typed(
            image=rgb_image,
            text_query=parsed.text_prompt,
            confidence_threshold=float(confidence_threshold),
        )

        selection: TargetSelectionResult = select_target_typed(
            dino_json=detection.meta.get("raw", {}),
            rules={
                "text_prompt": parsed.text_prompt,
                "axis": parsed.axis,
                "reverse": parsed.reverse,
                "target_rank": parsed.target_rank,
            },
            tolerance=float(tolerance),
        )

        if not selection.ok:
            return {
                "ok": False,
                "reason": f"selection_failed: {selection.reason}",
                "parsed": parsed,
                "detection": detection,
                "selection": selection,
                "mask": MaskResult(ok=False, mask=None, reason="no_selected_box"),
            }

        box = selection.selected.get("box_xyxy", None)
        if not isinstance(box, list) or len(box) != 4:
            return {
                "ok": False,
                "reason": "selection_has_no_valid_box_xyxy",
                "parsed": parsed,
                "detection": detection,
                "selection": selection,
                "mask": MaskResult(ok=False, mask=None, reason="invalid_box"),
            }

        mask: MaskResult = self.mask_generator.predict_mask_typed(
            image=rgb_image,
            box_xyxy=[float(v) for v in box],
        )

        return {
            "ok": bool(mask.ok),
            "reason": "" if mask.ok else mask.reason,
            "parsed": parsed,
            "detection": detection,
            "selection": selection,
            "mask": mask,
        }

