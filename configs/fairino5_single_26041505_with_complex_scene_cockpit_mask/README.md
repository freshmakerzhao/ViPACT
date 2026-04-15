# Experiment Note: `fairino5_single_26041505_with_complex_scene_cockpit_mask`

## 1) 目的
- 验证 **单视角 `cockpit` + mask** 是否比 `top` 更少遮挡、更利于复杂场景抓取。

## 2) 核心设置
- Dataset: `./data_sim_episodes/26041505_with_complex_scene_cockpit_mask`
- Cameras: `['cockpit']`
- Episodes: `200`
- `vipact.use_mask_conditioning: true`
- Train batch size: `32`
- Ckpt dir: `./ckpts/26041505_with_complex_scene_cockpit_mask`

## 3) 结论
- 待运行。
