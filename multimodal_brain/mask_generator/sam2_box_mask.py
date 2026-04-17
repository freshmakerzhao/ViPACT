from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class Sam2BoxMaskGenerator:
    """Generate binary mask from one box using SAM2."""

    model_id: str = "facebook/sam2-hiera-small"
    device: str = "cuda"

    def __post_init__(self):
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as e:
            raise ImportError(
                "Missing dependencies for SAM2. Please install transformers + torch + pillow."
            ) from e

        self._torch = torch
        self._processor = Sam2Processor.from_pretrained(self.model_id)
        self._model = Sam2Model.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    @staticmethod
    def _to_pil(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected image ndarray [H,W,3], got {image.shape}")
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def predict_mask(
        self,
        image: Union[str, np.ndarray, Image.Image],
        box_xyxy: List[float],
    ) -> np.ndarray:
        image_pil = self._to_pil(image)
        w, h = image_pil.size
        x0, y0, x1, y1 = [float(v) for v in box_xyxy]
        input_boxes = [[[x0, y0, x1, y1]]]

        inputs = self._processor(image_pil, input_boxes=input_boxes, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        if "original_sizes" not in inputs:
            raise KeyError(f"SAM2 inputs has no 'original_sizes'. keys={list(inputs.keys())}")

        # SAM2 in current transformers versions uses:
        # post_process_masks(masks, original_sizes, ...)
        # (no reshaped_input_sizes argument)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
        )

        # Usually: masks[0] -> [num_points=1, num_masks=3, H, W]
        mask_block = masks[0]
        if hasattr(mask_block, "numpy"):
            mask_block = mask_block.numpy()
        mask_block = np.asarray(mask_block)

        if mask_block.ndim == 4:
            mask_candidates = mask_block[0]  # [num_masks, H, W]
        elif mask_block.ndim == 3:
            mask_candidates = mask_block
        elif mask_block.ndim == 2:
            mask_candidates = mask_block[None, ...]
        else:
            raise RuntimeError(f"Unexpected SAM2 mask shape: {mask_block.shape}")

        best_idx = 0
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            iou_scores = outputs.iou_scores.detach().cpu().numpy()
            # usually [1,1,num_masks] or [1,num_masks]
            iou_scores = np.asarray(iou_scores).reshape(-1)
            if len(iou_scores) > 0:
                best_idx = int(np.argmax(iou_scores))
                best_idx = min(best_idx, mask_candidates.shape[0] - 1)

        best_mask = (mask_candidates[best_idx] > 0).astype(np.uint8)

        # shape safety
        if best_mask.shape != (h, w):
            best_mask = best_mask[:h, :w]
            pad_h = max(0, h - best_mask.shape[0])
            pad_w = max(0, w - best_mask.shape[1])
            if pad_h > 0 or pad_w > 0:
                best_mask = np.pad(best_mask, ((0, pad_h), (0, pad_w)), mode="constant")
        return best_mask


def mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    active = mask > 0
    overlay[active] = 0.65 * overlay[active] + 0.35 * np.array([255, 0, 0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)
