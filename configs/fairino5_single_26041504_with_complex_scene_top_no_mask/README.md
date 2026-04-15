# Experiment Note: `fairino5_single_26041504_with_complex_scene_top_no_mask`

## 1) 目的
- 在 **单视角 `top`** 下做 `mask` 消融：与 `26041500` 形成一一对照。

## 2) 核心设置
- Dataset: `./data_sim_episodes/26041504_with_complex_scene_top_no_mask`
- Cameras: `['top']`
- Episodes: `200`
- `vipact.use_mask_conditioning: false`
- Train batch size: `32`
- Ckpt dir: `./ckpts/26041504_with_complex_scene_top_no_mask`

## 3) 结论
- 待运行。
