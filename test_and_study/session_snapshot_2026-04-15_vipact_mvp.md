# Session Snapshot - ViPACT MVP (2026-04-15)

## 1) 项目目标（当前一致认知）
- 基于原版 ACT 做 ViPACT MVP。
- 在 MuJoCo 中使用静态 2D binary mask 作为视觉提示（第4通道）。
- 当前阶段目标：打通 RGB+Mask 的训练/评估链路。
- 最终方向：训练出“指哪打哪”的小脑策略，为后续大脑（上层目标指定）对接做准备。

## 2) 已完成改动（高价值部分）

### 2.1 Mask 条件化主链路已打通
- `vipact.use_mask_conditioning` 单开关已接入训练/评估。
- `image_channels` 由开关自动推导（3/4）。
- ACT backbone 首层卷积支持4通道输入。
- 训练与评估都支持 RGB + static mask。

关键文件：
- `constants.py`
- `utils.py`
- `imitate_episodes.py`
- `policy.py`
- `detr/models/backbone.py`

### 2.2 数据录制已改为写入 oracle static mask
- `record_sim_episodes.py` 会在 HDF5 中保存：
  - `/observations/static_masks/<cam>`
- `utils.EpisodicDataset` 在 `use_mask_conditioning=true` 时强制读取 static mask；没有就报错提示重录数据。

关键文件：
- `record_sim_episodes.py`
- `utils.py`

### 2.3 复杂场景环境已建立
- 新任务：`sim_lifting_cube_with_complex_scene_scripted`
- 场景中支持目标方块 + 干扰方块。
- FR5 XML 命名已统一为 `fairino_fr5_*` 路线。

关键文件：
- `ee_sim_env.py`
- `sim_env.py`
- `assets/fairino5_single/fairino_fr5_ee_lifting_cube_with_complex_scene.xml`
- `assets/fairino5_single/fairino_fr5_lifting_cube_with_complex_scene.xml`

### 2.4 实验自动化与调试工具
- 批处理脚本：`test_and_study/run_full_pipeline.sh`
  - 已支持：某个配置失败后跳过并继续下一个。
- mask 审计脚本：`test_and_study/audit_static_masks.py`
- 单回合目标演示脚本：`test_and_study/demo_grasp_target_by_id.py`
  - 已精简为固定场景：`ACT + fairino5_single + sim_lifting_cube_with_complex_scene_scripted`

## 3) 关键实验与结果

### 3.1 当前最佳里程碑结果
- 配置：`configs/fairino5_single_26041502_with_complex_scene`
- 结果文件：`ckpts/26041502_with_complex_scene/result_policy_best.txt`
- 指标：
  - Success rate: **0.92**
  - Average return: **615.1**

说明：
- 这证明“复杂场景 + dual view + mask 通道注入”的链路可用，性能很强。

### 3.2 Demo 发现的重要事实
- 使用 `demo_grasp_target_by_id.py` 切换 `target_id` 时，策略基本仍抓 `red_box(0)`。
- mask 可视化检查正常。

结论：
- 当前模型还没有学会“按 mask 切换目标”，而是学会了固定抓 `red_box`。

## 4) 当前核心瓶颈（统一结论）
- 问题不在推理脚本，而在训练监督分布：
  1. 训练数据中的示教轨迹主要对应 `red_box`。
  2. 当前 mask 语义也主要绑定 `red_box`。
- 因此模型可在不真正理解“目标切换语义”的情况下取得高分。

## 5) 下一阶段计划（最小侵入）

### Step A - 先改数据录制（优先）
- 每个 episode 采样 `target_id`（例如 0~3）。
- 根据 `target_id` 生成对应目标的 oracle mask。
- scripted policy 按 `target_id` 抓对应方块。
- 数据中保存 `target_id`（用于审计与复现）。

### Step B - 奖励绑定目标
- 环境奖励判定不再固定 `red_box`，而是跟随本 episode 的目标方块。

### Step C - 评估验证
- 在同类场景下执行：
  - `oracle / zero / random` mask ablation
  - 多 `target_id` 对照
- 验收标准：`oracle` 显著优于 `zero/random`，且目标切换时抓取对象随之变化。

## 6) 当前建议配置与脚本

### 6.1 批处理脚本
- `test_and_study/run_full_pipeline.sh`
- 当前配置队列：
  - `26041502_with_complex_scene`
  - `26041503_with_complex_scene_dual_no_mask`
  - `26041504_with_complex_scene_top_no_mask`
  - `26041505_with_complex_scene_cockpit_mask`
  - `26041506_with_complex_scene_cockpit_no_mask`

### 6.2 单回合演示脚本
- `test_and_study/demo_grasp_target_by_id.py`
- 示例命令：
```bash
conda run -n act3_env python test_and_study/demo_grasp_target_by_id.py \
  --config configs/fairino5_single_26041502_with_complex_scene/03_eval.yaml \
  --target-id 0 \
  --episode-id 0 \
  --save-video \
  --output-dir notes/target_demo
```

## 7) 注意事项
- 当前仓库存在较多未提交改动；后续改动需继续保持“最小改动 + 不破坏现有0.92链路”。
- `constants.py` 目前会打印完整配置 debug 信息（日志较冗长，后续可再收敛）。

## 8) 一句话状态
- **ViPACT 的 RGB+Mask MVP 链路已成功并拿到高分，但尚未学到“按目标切换抓取”；下一步是把训练监督改成多目标条件化。**
