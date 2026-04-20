# 26041701_with_complex_scene_cf_mask_longrun

## 目标
- 专攻 ACT 精度提升（不改任务、不改模型结构）
- 重新录制更大规模 counterfactual + mask 数据集
- 通过长时训练提升 `policy_best` 上限

## 关键设置
- 数据集：`./data_sim_episodes/26041701_with_complex_scene_cf_mask_longrun`
- 录制规模：`num_episodes=400`
- `num_epochs: 8000`
- `lr: 5e-6`
- `batch_size: 16`
- `temporal_agg: true`

## 建议执行
```bash
conda run -n act3_env python record_sim_episodes.py --yaml config \
  --config_path configs/fairino5_single_26041701_with_complex_scene_cf_mask_longrun/01_record.yaml
```

录制完成后训练：
```bash
conda run -n act3_env python imitate_episodes.py --yaml config \
  --config_path configs/fairino5_single_26041701_with_complex_scene_cf_mask_longrun/02_train.yaml
```

训练完成后评估：
```bash
conda run -n act3_env python imitate_episodes.py --yaml config \
  --config_path configs/fairino5_single_26041701_with_complex_scene_cf_mask_longrun/03_eval.yaml \
  --eval
```
