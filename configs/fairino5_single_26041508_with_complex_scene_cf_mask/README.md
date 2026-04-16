# 26041508_with_complex_scene_cf_mask

## 目标
- complex scene 多目标抓取实验。
- 双视角 RGB（top + cockpit）。
- mask 固定策略：仅 cockpit 生成与使用，top 自动补零 mask 通道。

## 关键设置
- target_conditioning.enabled: true
- target_conditioning.sampling: cycle
- target_conditioning.counterfactual_same_layout: true
- vipact.use_mask_conditioning: true

## 实验意图
反事实数据(同布局多目标) + Mask，作为主实验基线。
