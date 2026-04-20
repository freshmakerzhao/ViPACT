# mask_generator

职责：将目标候选转换为可供策略使用的二值掩码。

## 已实现
- `sam2_box_mask.py`:
  - `Sam2BoxMaskGenerator`: 使用 SAM2 + box prompt 生成二值 mask
- `mask_from_llm_box.py`:
  - 读取 `llm_parse_and_select_result.json` 中 `selection_result.selected.box_xyxy`
  - 输出：mask 图、overlay 图、box 图、summary json

## CLI 示例
```bash
python multimodal_brain/mask_generator/mask_from_llm_box.py \
  --llm-json notes/dino_tiny_debug_en/llm_parse_and_select_result.json \
  --image notes/dino_tiny_debug_en/dino_tiny_ep0_cockpit_rgb.png \
  --model-id facebook/sam2-hiera-small \
  --device cuda \
  --output-mask notes/dino_tiny_debug_en/sam2_mask.png \
  --output-overlay notes/dino_tiny_debug_en/sam2_overlay.png
```
