# 26041507_with_complex_scene_multitarget_mask

## 目标
- 在 complex scene 中开启多目标条件化录制（同一任务分布下轮询不同 `target_id`）。
- 使用 `RGB + static mask`（4通道）训练 ACT，验证是否学习到按目标切换抓取。

## 关键设置
- `target_conditioning.enabled: true`
- `target_conditioning.sampling: cycle`
- `target_conditioning.counterfactual_same_layout: true`
- `vipact.use_mask_conditioning: true`
- 相机：`top + cockpit`（双视角 RGB）
- mask 固定策略：仅 `cockpit` 生成/使用 oracle mask，`top` 自动补零 mask 通道

## 当前默认评估
- `03_eval.yaml` 里固定 `eval.target_id: 0`。
- 若要测“指哪打哪”，建议复制该 eval 文件后改 `target_id` 为 `1/2/3` 分别跑。
