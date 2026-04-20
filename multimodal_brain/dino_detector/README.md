# dino_detector

职责：基于 DINO 的目标候选检测。

## 功能
- 输入：一张图片 + 文本查询 + 置信度阈值
- 输出：所有分数大于阈值的框（`box_xyxy`）

## CLI 示例
```bash
conda run -n act3_env python multimodal_brain/dino_detector/detect_image.py \
  --image notes/dino_tiny_debug_en/dino_tiny_ep0_cockpit_rgb.png \
  --query "a red cube" \
  --confidence 0.5 \
  --output-json notes/dino_tiny_debug_en/dino_detection_result.json
```

```bash
conda run -n act3_env python multimodal_brain/dino_detector/visualize_detections.py \
  --image notes/dino_tiny_debug_en/dino_tiny_ep0_cockpit_rgb.png \
  --dino-json notes/dino_tiny_debug_en/dino_detection_result.json \
  --output-image notes/dino_tiny_debug_en/dino_detection_overlay.png \
  --top-k 20 \
  --min-score 0.5
```
