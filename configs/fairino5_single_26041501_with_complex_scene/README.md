# Experiment Note: `fairino5_single_26041501_with_complex_scene`

## 1) 目的
- 验证在 **复杂场景（多方块）** 下，使用 **双视角 `top + cockpit` + mask conditioning** 的效果。

## 2) 核心设置
- Task: `sim_lifting_cube_with_complex_scene_scripted`
- Dataset: `./data_sim_episodes/26041501_with_complex_scene`
- Cameras: `['top', 'cockpit']`
- Episodes: `50`
- Policy: `ACT`
- `vipact.use_mask_conditioning: true`
- Train batch size: `16`
- Seed: `1000`
- Ckpt dir: `./ckpts/26041501_with_complex_scene`

## 3) 运行命令
- Record:
```bash
conda run -n act3_env python record_sim_episodes.py --config configs/fairino5_single_26041501_with_complex_scene/01_record.yaml
```
- Train:
```bash
conda run -n act3_env python imitate_episodes.py --config configs/fairino5_single_26041501_with_complex_scene/02_train.yaml
```
- Eval:
```bash
conda run -n act3_env python imitate_episodes.py --config configs/fairino5_single_26041501_with_complex_scene/03_eval.yaml
```

## 4) 结果（来自 `ckpts/26041501_with_complex_scene/result_policy_best.txt`）
- Success rate: `0.16`
- Average return: `121.82`
- Reward >= 0: `50/50 = 100.0%`
- Reward >= 1: `23/50 = 46.0%`
- Reward >= 2: `14/50 = 28.0%`
- Reward >= 3: `8/50 = 16.0%`
- Reward >= 4: `8/50 = 16.0%`

## 5) 结论
- 在当前设置下，双视角并未带来期望提升，成功率显著低于 `26041500` 单视角 200 条数据实验。
- 该结果可作为“视角变更 + 小数据量”对照参考。

## 6) 下一步
- 创建 `26041502` 配置并继续实验（建议提高数据量后再比较双视角）。
- 用 `oracle/zero/random` ablation 验证双视角模型是否真正依赖 mask。
