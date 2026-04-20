# Experiment Note: `fairino5_single_26041503_with_complex_scene_dual_no_mask`

## 1) 目的
- 在 **双视角 `top + cockpit`** 下做 `mask` 消融：验证不使用 mask 时性能变化。

## 2) 核心设置
- Dataset: `./data_sim_episodes/26041503_with_complex_scene_dual_no_mask`
- Cameras: `['top', 'cockpit']`
- Episodes: `200`
- `vipact.use_mask_conditioning: false`
- Train batch size: `16`
- Ckpt dir: `./ckpts/26041503_with_complex_scene_dual_no_mask`

## 3) 结论
- 待运行。
