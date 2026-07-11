#!/usr/bin/env bash
# Runs the full 3 model x 3 dataset GAN-BERT experiment matrix for the thesis.
#
# Models:   SPECTER2 (allenai/specter2_base), SciBERT (allenai/scibert_scivocab_uncased), XLNet (xlnet-base-cased)
# Datasets: SciCite, ACL-ARC, 3C
#
# Each run fine-tunes only the last 2 transformer encoder layers (+ pooler,
# where present); the rest of the transformer is frozen. Up to 2 runs execute
# concurrently, one per GPU.
#
# Per-dataset hyperparameters below come from the prior thesis experiments
# (cGAN-SciBERT, full fine-tuning) and are re-used here as the tuned starting
# point for the partial-fine-tuning ablation across all 3 models:
#   SciCite:         L_max=160, batch=32, G=1, D=1, noise=768, dropout=0.20, lr_g=lr_d=2e-7, epochs=20
#   ACL-ARC and 3C:  L_max=64,  batch=16, G=2, D=1, noise=100, dropout=0.10, lr_d=5e-5, lr_g=5e-4, epochs=30
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
RESULTS_ROOT="$REPO_ROOT/results/gan_bert"
LOG_DIR="$RESULTS_ROOT/logs"

mkdir -p "$LOG_DIR"

echo "== Building dataset splits =="
uv run --project "$REPO_ROOT" python "$REPO_ROOT/src/utils/gan_bert_preprocessing.py" --seed 42 --unsup_frac 0.1

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

declare -A DATASET_DIRS=(
  [scicite]="scicite"
  [acl-arc]="acl-arc"
  [3C]="3C"
)

# Per-dataset hyperparameters (see header comment).
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
  echo "-> $dataset / $model_key on GPU $gpu (log: $log_file)"

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
      --dataset_name "$dataset" \
      --seed 42
  ) > "$log_file" 2>&1
}

# Build the full job matrix, dispatch round-robin across 2 GPUs, cap concurrency at 2.
jobs_pids=()
gpu_toggle=0
for dataset in scicite acl-arc 3C; do
  for model_key in specter2 scibert xlnet; do
    gpu=$gpu_toggle
    gpu_toggle=$(((gpu_toggle + 1) % 2))

    # Throttle to at most 2 concurrent background jobs.
    while [ "$(jobs -r -p | wc -l)" -ge 2 ]; do
      wait -n
    done

    run_one "$dataset" "$model_key" "$gpu" &
    jobs_pids+=($!)
  done
done

wait

echo "== All runs finished. Aggregating summary =="
uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/aggregate_gan_bert_results.py"
