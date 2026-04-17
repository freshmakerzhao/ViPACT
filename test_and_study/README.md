# test_and_study

该目录用于实验脚本与临时验证，不作为正式生产入口。

## 正式入口（推荐，仅保留三类能力）

- `vipact_console.py`
1. 单步调用子模块：`llm` / `dino` / `sam` / `act`
2. 一行命令跑全流程：`full`
3. 交互式全流程 + MuJoCo 实时窗口：`interactive`

核心编排逻辑在：
- `pipeline/brain_to_policy_pipeline.py`
