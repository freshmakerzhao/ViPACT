# Experiment Note: `fairino5_single_26041502_with_complex_scene`

## 1) 目的
- 验证在 **复杂场景（多方块）** 下，使用 **双视角 `top + cockpit` + mask conditioning**，当数据量提升到 `200` 条后，效果是否显著改善。

## 2) 核心设置
- Task: `sim_lifting_cube_with_complex_scene_scripted`
- Dataset: `./data_sim_episodes/26041502_with_complex_scene`
- Cameras: `['top', 'cockpit']`
- Episodes: `200`
- Policy: `ACT`
- `vipact.use_mask_conditioning: true`
- Train batch size: `16`
- Seed: `1000`
- Ckpt dir: `./ckpts/26041502_with_complex_scene`

## 3) 结论
- 待运行。

## 4) 下一步
- 与 `dual_no_mask` 做配对对照，确认 mask 在双视角下是否有效。
