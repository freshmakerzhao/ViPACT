# interactive_app

当前采用**同步阻塞交互 + MuJoCo 原生被动渲染器**模式：

1. `IDLE`：终端 `input()` 阻塞等待指令  
2. `THINKING`：Qwen -> DINO -> 规则筛选 -> SAM2  
3. `EXECUTING`：ViPACT (ACT) 执行动作，推进仿真并在 viewer 实时显示

## 运行

```bash
conda run -n act3_env python interactive_app/passive_viewer_console.py \
  --config configs/fairino5_single_26041508_with_complex_scene_cf_mask/03_eval.yaml \
  --output-dir notes/passive_viewer_runs
```

## 使用

- 启动后会弹出 MuJoCo viewer 窗口。
- 终端看到 `>>>` 后输入自然语言指令，例如：
  - `抓最下面的红色方块`
  - `抓左侧第二个红色方块`
- 输入 `q` / `quit` / `exit` 退出。

## 说明

- 该模式用于老师演示，非产品化 GUI。
- 若需 Qwen API，请设置环境变量 `DASHSCOPE_API_KEY` 或 `QWEN_API_KEY`。

