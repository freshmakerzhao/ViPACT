from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from PIL import Image
from interfaces import DetectionCandidate, DetectionResult


@dataclass
class GroundingDinoDetector:
    """Minimal GroundingDINO detector.

    Input: image + text query + confidence threshold
    Output: all boxes above threshold
    """

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "cuda"
    text_threshold: float = 0.25

    def __post_init__(self):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "Missing dependencies. Please install: pip install transformers torch torchvision pillow"
            ) from e

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    @staticmethod
    def _normalize_text_query(text_query: str) -> str:
        text = str(text_query).strip().lower()
        if text and not text.endswith("."):
            text = f"{text}."
        return text

    @staticmethod
    def _to_pil(image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected ndarray shape [H,W,3], got {image.shape}")
            return Image.fromarray(image.astype(np.uint8))
        raise TypeError(f"Unsupported image type: {type(image)}")

    def detect_boxes(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_query: str,
        confidence_threshold: float = 0.5,
    ) -> Dict:
        """Detect objects by text query and return all boxes above threshold."""
        image_pil = self._to_pil(image)
        w, h = image_pil.size
        query = self._normalize_text_query(text_query)

        inputs = self._processor(images=image_pil, text=query, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        target_sizes = self._torch.tensor([[h, w]], device=self.device)
        post_fn = self._processor.post_process_grounded_object_detection
        post_sig = inspect.signature(post_fn)
        post_kwargs = {
            "outputs": outputs,
            "input_ids": inputs.get("input_ids", None),
            "text_threshold": float(self.text_threshold),
            "target_sizes": target_sizes,
        }

        # Compatibility across transformers versions:
        # - some use `box_threshold`
        # - some use `threshold`
        if "box_threshold" in post_sig.parameters:
            post_kwargs["box_threshold"] = float(confidence_threshold)
        else:
            post_kwargs["threshold"] = float(confidence_threshold)

        results = post_fn(**post_kwargs)
        if len(results) == 0:
            return {
                "query": query,
                "confidence_threshold": float(confidence_threshold),
                "text_threshold": float(self.text_threshold),
                "num_boxes": 0,
                "detections": [],
                "image_size_hw": [int(h), int(w)],
            }

        result0 = results[0]
        boxes = result0["boxes"].detach().cpu().numpy() if "boxes" in result0 else np.zeros((0, 4))
        scores = result0["scores"].detach().cpu().numpy() if "scores" in result0 else np.zeros((0,))
        label_key = "text_labels" if "text_labels" in result0 else "labels"
        labels_raw = result0.get(label_key, [])
        labels = [str(x) for x in labels_raw]

        order = np.argsort(-scores)
        detections: List[Dict] = []
        for rank, idx in enumerate(order):
            i = int(idx)
            x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
            bw = max(0.0, x1 - x0)
            bh = max(0.0, y1 - y0)
            detections.append(
                {
                    "rank": int(rank),
                    "index": i,
                    "score": float(scores[i]),
                    "label": labels[i] if i < len(labels) else "",
                    "box_xyxy": [x0, y0, x1, y1],
                    "area_ratio": float((bw * bh) / max(1.0, float(h * w))),
                }
            )

        return {
            "query": query,
            "confidence_threshold": float(confidence_threshold),
            "text_threshold": float(self.text_threshold),
            "num_boxes": int(len(detections)),
            "detections": detections,
            "image_size_hw": [int(h), int(w)],
        }

    def detect_boxes_typed(
        self,
        image: Union[np.ndarray, Image.Image, str],
        text_query: str,
        confidence_threshold: float = 0.5,
    ) -> DetectionResult:
        raw = self.detect_boxes(
            image=image,
            text_query=text_query,
            confidence_threshold=float(confidence_threshold),
        )
        detections = raw.get("detections", [])
        candidates: List[DetectionCandidate] = []
        for d in detections:
            candidates.append(
                DetectionCandidate(
                    index=int(d.get("index", -1)),
                    score=float(d.get("score", 0.0)),
                    label=str(d.get("label", "")),
                    box_xyxy=[float(v) for v in d.get("box_xyxy", [0, 0, 0, 0])],
                    meta={
                        "rank": int(d.get("rank", -1)),
                        "area_ratio": float(d.get("area_ratio", 0.0)),
                    },
                )
            )

        return DetectionResult(
            ok=bool(raw.get("num_boxes", 0) > 0),
            candidates=candidates,
            reason="" if raw.get("num_boxes", 0) > 0 else "no_boxes",
            meta={
                "query": raw.get("query", ""),
                "confidence_threshold": float(raw.get("confidence_threshold", confidence_threshold)),
                "text_threshold": float(raw.get("text_threshold", self.text_threshold)),
                "num_boxes": int(raw.get("num_boxes", 0)),
                "image_size_hw": raw.get("image_size_hw", []),
                "raw": raw,
            },
        )
