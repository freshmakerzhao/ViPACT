#!/usr/bin/env bash
set -uo pipefail

# ==========================================================
# Simple full pipeline runner (record -> train -> eval)
# Usage:
#   1) Edit CONFIG_DIRS below (absolute paths recommended)
#   2) Run:
#      bash test_and_study/run_full_pipeline.sh
# ==========================================================

# Conda environment that has all required dependencies.
CONDA_ENV="act3_env"

# Python entry scripts (relative to repo root).
RECORD_SCRIPT="record_sim_episodes.py"
TRAIN_EVAL_SCRIPT="imitate_episodes.py"

# Fixed config file names in each config directory.
RECORD_CFG_NAME="01_record.yaml"
TRAIN_CFG_NAME="02_train.yaml"
EVAL_CFG_NAME="03_eval.yaml"

# Put your config directories here.
CONFIG_DIRS=(
  "/home/zhaoshuai/workspace_act/ViPACT/configs/fairino5_single_26041701_with_complex_scene_cf_mask_longrun"
  "/home/zhaoshuai/workspace_act/ViPACT/configs/fairino5_single_26041702_with_complex_scene_cf_mask_longrun"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$REPO_ROOT/notes/pipeline_logs/$(date +%y%m%d_%H%M%S)_bash"
mkdir -p "$LOG_ROOT"

echo "Repo root: $REPO_ROOT"
echo "Log root : $LOG_ROOT"
echo "Conda env: $CONDA_ENV"
echo

run_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2
  echo "[$(date '+%F %T')] RUN [$step_name]"
  echo "CMD: $*"
  echo "LOG: $log_file"
  conda run -n "$CONDA_ENV" "$@" 2>&1 | tee "$log_file"
  echo "[$(date '+%F %T')] DONE [$step_name]"
  echo
}

SUCCESS_CONFIGS=()
FAILED_CONFIGS=()

for cfg_dir in "${CONFIG_DIRS[@]}"; do
  echo "=========================================================="
  echo "Config dir: $cfg_dir"
  echo "=========================================================="

  if [[ ! -d "$cfg_dir" ]]; then
    echo "ERROR: config dir not found: $cfg_dir" >&2
    exit 1
  fi

  record_cfg="$cfg_dir/$RECORD_CFG_NAME"
  train_cfg="$cfg_dir/$TRAIN_CFG_NAME"
  eval_cfg="$cfg_dir/$EVAL_CFG_NAME"

  for f in "$record_cfg" "$train_cfg" "$eval_cfg"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: config file not found: $f" >&2
      exit 1
    fi
  done

  cfg_name="$(basename "$cfg_dir")"

  if ! run_step "record:$cfg_name" \
    "$LOG_ROOT/${cfg_name}_01_record.log" \
    python "$REPO_ROOT/$RECORD_SCRIPT" --config "$record_cfg"; then
    echo "[$(date '+%F %T')] ERROR [$cfg_name] record failed, skip this config."
    echo
    FAILED_CONFIGS+=("${cfg_name}:record")
    continue
  fi

  if ! run_step "train:$cfg_name" \
    "$LOG_ROOT/${cfg_name}_02_train.log" \
    python "$REPO_ROOT/$TRAIN_EVAL_SCRIPT" --config "$train_cfg"; then
    echo "[$(date '+%F %T')] ERROR [$cfg_name] train failed, skip eval."
    echo
    FAILED_CONFIGS+=("${cfg_name}:train")
    continue
  fi

  if ! run_step "eval:$cfg_name" \
    "$LOG_ROOT/${cfg_name}_03_eval.log" \
    python "$REPO_ROOT/$TRAIN_EVAL_SCRIPT" --config "$eval_cfg"; then
    echo "[$(date '+%F %T')] ERROR [$cfg_name] eval failed."
    echo
    FAILED_CONFIGS+=("${cfg_name}:eval")
    continue
  fi

  SUCCESS_CONFIGS+=("$cfg_name")
done

echo "All pipelines completed."
echo "Logs saved to: $LOG_ROOT"
echo
echo "Succeeded (${#SUCCESS_CONFIGS[@]}):"
for name in "${SUCCESS_CONFIGS[@]}"; do
  echo "  - $name"
done
echo
echo "Failed (${#FAILED_CONFIGS[@]}):"
for item in "${FAILED_CONFIGS[@]}"; do
  echo "  - $item"
done
