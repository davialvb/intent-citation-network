#!/usr/bin/env bash
# Verification runs for the methodology audit (see methodology_findings.md).
#
# All runs are SciCite / SciBERT / FULL fine-tuning (12 layers), 20 epochs,
# batch 32, max_seq_length 160 -- i.e. the thesis' flagship cell -- so that the
# only thing varying is the knob named in each run's output directory.
#
#   B  : no-GAN baseline at the SS-cGAN's learning rate (2e-7) and, conversely,
#        SS-cGAN at the baseline's learning rate (2e-5). Separates "adversarial
#        regularization" from "100x smaller step size".
#   C  : seeds 1/2/3 (plus a seed-42 replication) for both arms at their
#        original learning rates, for mean +/- std and rebound consistency.
#   D  : no-GAN + a simple regularizer (label smoothing / weight decay) as a
#        cheap non-adversarial comparison.
#
# Two jobs run concurrently, one per GPU. Non-best checkpoints are pruned after
# each run to keep the disk footprint bounded.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
OUT_ROOT="$REPO_ROOT/results/verify"
DATA_DIR="$REPO_ROOT/data/scicite/gan_bert"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

MODEL="allenai/scibert_scivocab_uncased"
COMMON="--epochs 20 --max_seq_length 160 --batch_size 32 --num_trainable_layers 12 --dataset_name scicite"

# Keep only the highest-F1 checkpoint in each of a run's weight directories.
prune_checkpoints() {
  local run_dir="$1"
  for sub in transformer discriminator generator; do
    local d="$run_dir/$sub"
    [ -d "$d" ] || continue
    # Checkpoint filenames end in _f1_<score>.pth; sort by that score, drop the top one.
    ls "$d"/*.pth 2>/dev/null \
      | sed -E 's/.*_f1_([0-9.]+)\.pth$/\1 &/' \
      | sort -k1,1g | head -n -1 | cut -d' ' -f2- \
      | xargs -r rm -f
  done
}

run_baseline() {
  local name="$1" gpu="$2" lr="$3" seed="$4"; shift 4
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  ( cd "$GAN_BERT_SRC"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train_baseline.py \
      --labeled_csv "$DATA_DIR/labeled_train.csv" \
      --val_csv "$DATA_DIR/val.csv" \
      --test_csv "$DATA_DIR/test.csv" \
      --labels background method result \
      --model_name "$MODEL" --output_dir "$out" \
      $COMMON --lr "$lr" --seed "$seed" "$@"
  ) > "$LOG_DIR/$name.log" 2>&1
  prune_checkpoints "$out"
  echo "   done: $name"
}

run_gan() {
  local name="$1" gpu="$2" lr="$3" seed="$4"; shift 4
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  ( cd "$GAN_BERT_SRC"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train.py \
      --labeled_csv "$DATA_DIR/labeled_train.csv" \
      --unlabeled_csv "$DATA_DIR/unsupervised.csv" \
      --val_csv "$DATA_DIR/val.csv" \
      --test_csv "$DATA_DIR/test.csv" \
      --labels background method result unknown \
      --model_name "$MODEL" --output_dir "$out" \
      $COMMON --hidden_layers_g 1 --hidden_layers_d 1 --noise_size 768 \
      --dropout 0.20 --epsilon 2e-7 \
      --lr_d "$lr" --lr_g "$lr" --seed "$seed" "$@"
  ) > "$LOG_DIR/$name.log" 2>&1
  prune_checkpoints "$out"
  echo "   done: $name"
}

queue_gpu0() {
  run_gan      gan_lr2e-5_seed42       0 2e-5 42     # B-complement
  run_baseline baseline_lr2e-5_seed42  0 2e-5 42     # C replication (+ checkpoints)
  run_baseline baseline_lr2e-5_seed1   0 2e-5 1
  run_baseline baseline_lr2e-5_seed2   0 2e-5 2
  run_baseline baseline_lr2e-5_seed3   0 2e-5 3
}

queue_gpu1() {
  run_gan      gan_lr2e-7_seed1        1 2e-7 1      # C
  run_gan      gan_lr2e-7_seed2        1 2e-7 2
  run_gan      gan_lr2e-7_seed3        1 2e-7 3
  run_baseline baseline_ls0.1_seed42   1 2e-5 42 --label_smoothing 0.1   # D
  run_baseline baseline_wd0.1_seed42   1 2e-5 42 --weight_decay 0.1      # D
}

echo "== methodology verification runs starting =="
queue_gpu0 &
queue_gpu1 &
wait
echo "== all verification runs finished =="
