# Experiment Note: `fairino5_single_26041500_with_complex_scene`

## 1) 目的
- 验证在 **复杂场景（多方块）** 下，使用 **单视角 `top` + mask conditioning**，当数据量提升到 `200` 条时，ViPACT 是否能获得可用成功率。


## 2) 结果（来自 `ckpts/26041500_with_complex_scene/result_policy_best.txt`）
- Success rate: `0.58`
- Average return: `399.94`
- Reward >= 0: `50/50 = 100.0%`
- Reward >= 1: `46/50 = 92.0%`
- Reward >= 2: `35/50 = 70.0%`
- Reward >= 3: `29/50 = 58.0%`
- Reward >= 4: `29/50 = 58.0%`

## 3) 结论
- 相比早期低数据量复杂场景实验，`200` 条数据显著提升了可学性。
- 当前这组结果可作为后续复杂场景实验基线。

## 4) 下一步
- 用相同设置跑 `mask ablation`（oracle / zero / random）确认 mask 真实贡献。
- 在新实验中测试 `top + cockpit` 双视角。
