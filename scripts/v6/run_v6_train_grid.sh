#!/usr/bin/env bash
# v6.0 — 主线三 family 全网格后台训练 (P3)
# family × endpoint × budget × seed = 3 × 2 × 4 × 5 = 120 runs
# 串行执行，日志追加到 experiments/v6_20260830/train_grid.log
# 幂等：仅当 evaluation.json 与 test_predictions.csv 同时存在才视为完成
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="$ROOT/experiments/v6_20260830"
LOG="$OUT_ROOT/train_grid.log"
mkdir -p "$OUT_ROOT"
DEVICE="${DEVICE:-cuda}"
PRETRAIN_EP="${PRETRAIN_EP:-40}"
UPDATE_EP="${UPDATE_EP:-20}"

FAMILIES=(target_only source_only supervised_update)
ENDPOINTS=(idh ih)
BUDGETS=(0.10 0.25 0.50 1.00)
SEEDS=(401 402 433 202 107)

echo "=== v6 train grid start $(date) device=$DEVICE preview=$(date +%s) ===" | tee -a "$LOG"
total=0
for fam in "${FAMILIES[@]}"; do
  for ep in "${ENDPOINTS[@]}"; do
    for b in "${BUDGETS[@]}"; do
      for s in "${SEEDS[@]}"; do
        total=$((total+1))
        out="$OUT_ROOT/${fam}__${ep}__${b}__seed${s}"
        # 完成判定：须同时具备聚合评估与 session 级预测（bootstrap 前提）
        if [ -f "$out/evaluation.json" ] && [ -f "$out/test_predictions.csv" ]; then
          echo "[skip] $out" | tee -a "$LOG"
          continue
        fi
        echo "[run $total] $fam $ep $b seed$s $(date +%H:%M:%S)" | tee -a "$LOG"
        python3 "$ROOT/scripts/v6/run_v6_train.py" \
          --family "$fam" --endpoint "$ep" --budget "$b" --seed "$s" \
          --device "$DEVICE" \
          --out-root "$OUT_ROOT" \
          --max-pretrain-epochs "$PRETRAIN_EP" --max-update-epochs "$UPDATE_EP" \
          >> "$LOG" 2>&1
        if [ $? -ne 0 ]; then
          echo "[ERROR] failed $fam $ep $b seed$s - continue" | tee -a "$LOG"
        fi
      done
    done
  done
done
echo "=== v6 train grid done $(date) attempted=$total ===" | tee -a "$LOG"