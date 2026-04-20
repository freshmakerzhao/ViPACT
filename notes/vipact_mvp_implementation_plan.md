# ViPACT MVP 实施计划（基于 ACT，交互式小步迭代）

## 项目目标
在保留原版 ACT 主体结构的前提下，实现 ViPACT 的最小可行原型：
- 在 MuJoCo 抓取场景中支持 mask-conditioned grasping
- 以静态 2D binary mask 作为高层意图提示
- 将 mask 作为第 4 个视觉通道注入 ACT（RGB + Mask）
- 打通训练 / 评估链路
- 当前阶段不接入 VLM、语言和复杂 brain 模块

## 关键原则
1. 最小侵入（Minimal invasive changes）
2. 保持原版 ACT 可用（默认 RGB-only）
3. 通过配置开关控制是否启用 ViPACT
4. 每一步都可运行、可测试、可回退

## 当前代码关键路径（已识别）
1. 训练/评估入口：`imitate_episodes.py`
2. 数据集读取：`utils.py::EpisodicDataset`
3. 策略适配层：`policy.py`
4. ACT 构建入口：`detr/main.py`
5. 模型视觉前向：`detr/models/detr_vae.py`
6. 视觉骨干网络：`detr/models/backbone.py`
7. YAML配置解析：`constants.py`

## 分阶段实施

### Step 0：验收门槛定义（不改代码）
目标：明确 MVP 成功标准，避免无效迭代。
- 功能门槛：RGB+Mask 训练和评估链路完整跑通
- 价值门槛：在目标条件化场景（多目标/干扰物/OOD）优于 RGB-only 基线

验证：形成 checklist，并在后续步骤逐项对照。

---

### Step 1：配置开关与参数透传（零行为变化）
目标：先搭好兼容框架，默认不启用 mask。
涉及文件：
- `constants.py`
- `imitate_episodes.py`
- `configs/fairino5_single/02_train.yaml`
- `configs/fairino5_single/03_eval.yaml`

预期改动：
- 新增配置字段：
  - `vipact.use_mask_conditioning: false`
- 不再要求显式配置 `image_channels`，内部按开关自动推导：
  - `use_mask_conditioning=false -> image_channels=3`
  - `use_mask_conditioning=true -> image_channels=4`
- 训练配置读取并透传到 `policy_config`
- 默认设置下行为与原版 ACT 一致

风险点：
- 老配置缺少 `vipact` 字段时兼容处理

验证：
- 用现有配置启动训练/评估流程，不启用 mask 时行为不变

兼容性：
- 完全兼容 RGB-only 原版 ACT

---

### Step 2：训练与评估输入注入静态 mask
目标：实现 episode 级静态 mask，并拼接到第 4 通道。
涉及文件：
- `utils.py`
- `imitate_episodes.py`

预期改动：
- 训练：Dataset 输出从 `[K,3,H,W]` 可切换为 `[K,4,H,W]`
- 评估：每次 `env.reset()` 生成一次静态 mask，episode 内复用
- 每个时刻将 `rgb` 与固定 `mask` 进行 `cat`

风险点：
- mask 尺寸/类型不一致
- mask 生成失败导致全零
- 训练与评估 mask 逻辑不一致

验证：
- 增加 shape 断言
- 打印/统计 mask 非空比例

兼容性：
- 关闭开关仍走 RGB-only

---

### Step 3：模型最小改造支持 4 通道
目标：仅修改视觉第一层卷积输入通道，保持主干不动。
涉及文件：
- `detr/main.py`
- `detr/models/backbone.py`
- `policy.py`

预期改动：
- 将自动推导的 `image_channels` 传入模型构建参数
- backbone `conv1` 支持输入通道 4
- Normalize 支持 3 或 4 通道

风险点：
- Normalize 通道数与输入不匹配
- conv1 通道不匹配触发 runtime error

验证：
- 单 batch 前向通过
- 单 rollout 推理通过

兼容性：
- `use_mask_conditioning=false` 时与原版一致

---

### Step 4：端到端 smoke test 与 A/B 快速评估
目标：验证 MVP 链路与初步价值。
涉及文件：配置文件为主。

预期改动：
- 运行 RGB-only 与 RGB+Mask 两组短实验

验证：
1. 功能：训练 1-2 epoch + 评估 1 rollout 跑通
2. 结果：在条件化场景比较 success rate/return

兼容性：
- 可随时通过配置切回原版 ACT

## 当前默认假设（后续可调整）
- mask 先采用“静态二值图（episode 固定）”
- 当前不引入语言或 VLM 实时模块
- 优先保障链路稳定和代码可维护

## 实施更新（2026-04-13）
- Step 2 已从纯颜色启发式升级为“仿真 oracle 优先”：
  - 评估阶段：直接用 MuJoCo segmentation 生成目标几何体静态 mask
  - 录制阶段：将每个 camera 的静态 oracle mask 写入 HDF5 `observations/static_masks/<cam>`
  - 训练阶段：仅读取 `static_masks`（oracle-only）；旧数据集没有该字段时直接报错并提示重录
