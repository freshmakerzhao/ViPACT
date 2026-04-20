# Demo Models (Frozen)

这个目录用于演示留存，不保存整套 epoch checkpoint，只保留每个模型的关键资产：
- `policy_best.ckpt`（软链接到 `ckpts/`）
- `dataset_stats.pkl`（软链接到 `ckpts/`）
- `train_config.yaml`
- `eval_config.yaml`
- `eval_result_policy_best.txt`

## 模型清单

1. `26041701_temporal_agg`
- 特点：训练/评估配置开启 `temporal_agg: true`
- 来源：`ckpts/26041701_with_complex_scene_cf_mask_longrun`

2. `26041702_no_temporal_agg`
- 特点：训练/评估配置关闭 `temporal_agg`
- 来源：`ckpts/26041702_no_temporal_agg`

## 演示命令（推荐）

使用带 `temporal_agg` 的模型：
```bash
conda run -n act3_env python vipact_console.py interactive \
  --config configs/fairino5_single_26041701_with_complex_scene_cf_mask_longrun/03_eval.yaml \
  --output-dir notes/passive_viewer_runs_26041701 \
  --episode-id-base 0
```

使用不带 `temporal_agg` 的模型：
```bash
conda run -n act3_env python vipact_console.py interactive \
  --config configs/fairino5_single_26041702_with_complex_scene_cf_mask_longrun/03_eval.yaml \
  --output-dir notes/passive_viewer_runs_26041702 \
  --episode-id-base 0
```

## 说明

- 为节省磁盘，这里使用软链接，不复制大文件。

