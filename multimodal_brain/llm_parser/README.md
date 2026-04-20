# llm_parser

职责：自然语言指令解析与目标选择策略输出。

## 已实现
- `qwen_parser.py`:
  - `QwenInstructionParser`: 指令 -> 结构化 JSON 规则
  - 字段: `text_prompt`, `axis`, `reverse`, `target_rank`
  - Qwen API 失败时自动回退到启发式解析
- `target_selector.py`:
  - 按 `axis/reverse/target_rank` 在 DINO 框中排序+聚类+选目标
- `parse_and_select.py`:
  - CLI 串联：指令 + DINO JSON -> 最终目标框

## CLI 示例
```bash
python multimodal_brain/llm_parser/parse_and_select.py \
  --instruction "抓左侧第二个红色方块" \
  --dino-json notes/dino_tiny_debug_en/dino_detection_result.json \
  --api-key "your_qwen_key" \
  --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --model "qwen-plus" \
  --output-json notes/dino_tiny_debug_en/llm_parse_and_select_result.json
```
