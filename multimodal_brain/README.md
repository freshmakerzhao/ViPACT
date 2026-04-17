# Multimodal Brain Modules

- `llm_parser/`: 负责语言理解与任务语义解析（指令 -> 结构化目标）。
- `dino_detector/`: 负责目标检测候选生成（图像 -> 候选框/候选目标）。
- `mask_generator/`: 负责掩码生成（候选目标 -> 二值 mask）。
- `facade.py`: 统一串联入口（instruction + rgb -> parse -> detect -> select -> mask）。

该目录提供模块化实现，支持两种调用方式：
1. 直接调用各子模块。
2. 通过 `BrainPerceptionFacade` 一次完成大脑感知链路。

## 安装（Editable）
在 `multimodal_brain` 目录执行：

```bash
cd multimodal_brain
pip install -e .
```

## 推荐调用（Python）

```python
from multimodal_brain import BrainPerceptionFacade

facade = BrainPerceptionFacade.create_default(device="cuda")
result = facade.run_once(
    rgb_image=rgb_np,
    instruction="抓左侧第二个红色方块",
    confidence_threshold=0.5,
    tolerance=20.0,
)
```
