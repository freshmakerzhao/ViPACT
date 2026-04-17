# Multimodal Brain Modules

- `llm_parser/`: 负责语言理解与任务语义解析（指令 -> 结构化目标）。
- `dino_detector/`: 负责目标检测候选生成（图像 -> 候选框/候选目标）。
- `mask_generator/`: 负责掩码生成（候选目标 -> 二值 mask）。

该目录当前仅建立模块边界，具体实现后续按步骤补充。

## 安装（Editable）
在 `multimodal_brain` 目录执行：

```bash
cd multimodal_brain
pip install -e .
```
