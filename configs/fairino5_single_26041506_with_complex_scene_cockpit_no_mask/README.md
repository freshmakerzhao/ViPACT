# Experiment Note: `fairino5_single_26041506_with_complex_scene_cockpit_no_mask`

## 1) 目的
- 在 **单视角 `cockpit`** 下做 `mask` 消融，判断 mask 在 cockpit 视角的真实贡献。

## 2) 核心设置
- Dataset: `./data_sim_episodes/26041506_with_complex_scene_cockpit_no_mask`
- Cameras: `['cockpit']`
- Episodes: `200`
- `vipact.use_mask_conditioning: false`
- Train batch size: `32`
- Ckpt dir: `./ckpts/26041506_with_complex_scene_cockpit_no_mask`

## 3) 结论
- 待运行。
