#!/usr/bin/env bash
# No-GAN baseline: same transformer backbones, same last-2-layers freezing, and
# the same per-dataset batch/seq-length as the GAN-BERT runs, but trained with
# plain supervised cross-entropy (no generator/discriminator, no unlabeled
# stream). Used to isolate whether GAN-BERT's semi-supervised training helps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
RESULTS_ROOT="$REPO_ROOT/results/baseline"
LOG_DIR="$RESULTS_ROOT/logs"

mkdir -p "$LOG_DIR"

declare -A MODEL_NAMES=(
  [specter2]="allenai/specter2_base"
  [scibert]="allenai/scibert_scivocab_uncased"
  [xlnet]="xlnet-base-cased"
)

declare -A DATASET_LABELS=(
  [scicite]="background method result"
  [acl-arc]="background uses compareorcontrast motivation extends future"
  [3C]="background comparecontrast extension future motivation uses"
)

declare -A DATASET_DIRS=( [scicite]="scicite" [acl-arc]="acl-arc" [3C]="3C" )

declare -A DATASET_MAX_SEQ=( [scicite]=160 [acl-arc]=64 [3C]=64 )
declare -A DATASET_BATCH=(   [scicite]=32  [acl-arc]=16 [3C]=16 )
declare -A DATASET_EPOCHS=(  [scicite]=20  [acl-arc]=30 [3C]=30 )

run_one() {
  local dataset="$1" model_key="$2" gpu="$3"
  local model_name="${MODEL_NAMES[$model_key]}"
  local labels="${DATASET_LABELS[$dataset]}"
  local data_dir="$REPO_ROOT/data/${DATASET_DIRS[$dataset]}/gan_bert"
  local out_dir="$RESULTS_ROOT/$dataset/$model_key"
  local log_file="$LOG_DIR/${dataset}_${model_key}.log"

  mkdir -p "$out_dir"
  echo "-> [baseline] $dataset / $model_key on GPU $gpu (log: $log_file)"

  (
    cd "$GAN_BERT_SRC"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --project "$REPO_ROOT" python cli_train_baseline.py \
      --labeled_csv "$data_dir/labeled_train.csv" \
      --val_csv "$data_dir/val.csv" \
      --test_csv "$data_dir/test.csv" \
      --labels $labels \
      --model_name "$model_name" \
      --output_dir "$out_dir" \
      --epochs "${DATASET_EPOCHS[$dataset]}" \
      --max_seq_length "${DATASET_MAX_SEQ[$dataset]}" \
      --batch_size "${DATASET_BATCH[$dataset]}" \
      --lr 2e-5 \
      --num_trainable_layers 2 \
      --dataset_name "$dataset" \
      --seed 42
  ) > "$log_file" 2>&1
}

gpu_toggle=0
for dataset in scicite acl-arc 3C; do
  for model_key in specter2 scibert xlnet; do
    gpu=$gpu_toggle
    gpu_toggle=$(((gpu_toggle + 1) % 2))
    while [ "$(jobs -r -p | wc -l)" -ge 2 ]; do
      wait -n
    done
    run_one "$dataset" "$model_key" "$gpu" &
  done
done
wait
echo "== Baseline runs finished =="
