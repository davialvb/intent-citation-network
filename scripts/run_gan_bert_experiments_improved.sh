#!/usr/bin/env bash
# Same 3x3 GAN-BERT matrix and per-dataset hyperparameters as
# run_gan_bert_experiments.sh, but with the "improved" semi-supervised loss
# enabled: label smoothing on the supervised term (reduces overconfidence /
# mode-collapse risk) + Pi-model consistency regularization on the unlabeled
# stream (two stochastic discriminator forward passes must agree). Everything
# else (data, seed, architecture, epochs) is identical to the vanilla runs in
# results/gan_bert/, so any metric delta is attributable to the loss change.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
RESULTS_ROOT="$REPO_ROOT/results/gan_bert_improved"
LOG_DIR="$RESULTS_ROOT/logs"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"

mkdir -p "$LOG_DIR"

declare -A MODEL_NAMES=(
  [specter2]="allenai/specter2_base"
  [scibert]="allenai/scibert_scivocab_uncased"
  [xlnet]="xlnet-base-cased"
)

declare -A DATASET_LABELS=(
  [scicite]="background method result unknown"
  [acl-arc]="background uses compareorcontrast motivation extends future unknown"
  [3C]="background comparecontrast extension future motivation uses unknown"
)

declare -A DATASET_DIRS=( [scicite]="scicite" [acl-arc]="acl-arc" [3C]="3C" )

declare -A DATASET_MAX_SEQ=( [scicite]=160 [acl-arc]=64  [3C]=64  )
declare -A DATASET_BATCH=(   [scicite]=32  [acl-arc]=16  [3C]=16  )
declare -A DATASET_HG=(      [scicite]=1   [acl-arc]=2   [3C]=2   )
declare -A DATASET_HD=(      [scicite]=1   [acl-arc]=1   [3C]=1   )
declare -A DATASET_NOISE=(   [scicite]=768 [acl-arc]=100 [3C]=100 )
declare -A DATASET_DROPOUT=( [scicite]=0.20 [acl-arc]=0.10 [3C]=0.10 )
declare -A DATASET_LR_D=(    [scicite]=2e-7 [acl-arc]=5e-5 [3C]=5e-5 )
declare -A DATASET_LR_G=(    [scicite]=2e-7 [acl-arc]=5e-4 [3C]=5e-4 )
declare -A DATASET_EPOCHS=(  [scicite]=20  [acl-arc]=30  [3C]=30  )

run_one() {
  local dataset="$1" model_key="$2" gpu="$3"
  local model_name="${MODEL_NAMES[$model_key]}"
  local labels="${DATASET_LABELS[$dataset]}"
  local data_dir="$REPO_ROOT/data/${DATASET_DIRS[$dataset]}/gan_bert"
  local out_dir="$RESULTS_ROOT/$dataset/$model_key"
  local log_file="$LOG_DIR/${dataset}_${model_key}.log"

  mkdir -p "$out_dir"
  echo "-> [improved] $dataset / $model_key on GPU $gpu (log: $log_file)"

  (
    cd "$GAN_BERT_SRC"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --project "$REPO_ROOT" python cli_train.py \
      --labeled_csv "$data_dir/labeled_train.csv" \
      --unlabeled_csv "$data_dir/unsupervised.csv" \
      --val_csv "$data_dir/val.csv" \
      --test_csv "$data_dir/test.csv" \
      --labels $labels \
      --model_name "$model_name" \
      --output_dir "$out_dir" \
      --epochs "${DATASET_EPOCHS[$dataset]}" \
      --max_seq_length "${DATASET_MAX_SEQ[$dataset]}" \
      --batch_size "${DATASET_BATCH[$dataset]}" \
      --hidden_layers_g "${DATASET_HG[$dataset]}" \
      --hidden_layers_d "${DATASET_HD[$dataset]}" \
      --noise_size "${DATASET_NOISE[$dataset]}" \
      --dropout "${DATASET_DROPOUT[$dataset]}" \
      --lr_d "${DATASET_LR_D[$dataset]}" \
      --lr_g "${DATASET_LR_G[$dataset]}" \
      --epsilon 2e-7 \
      --num_trainable_layers 2 \
      --label_smoothing "$LABEL_SMOOTHING" \
      --consistency_weight "$CONSISTENCY_WEIGHT" \
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
echo "== Improved-GAN runs finished =="
