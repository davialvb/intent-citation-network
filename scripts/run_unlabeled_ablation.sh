#!/usr/bin/env bash
# Tier 1: is the unlabeled pool doing anything?
#
# SS-cGAN on SciCite / SciBERT / full fine-tuning at LR 2e-7, seeds 1/2/3/42,
# with the unlabeled pool REMOVED (--unlabeled_csv omitted, so cli_train.py
# leaves `unlabeled` as None and writes n_train_unlabeled: 0).
#
# Everything else is byte-identical to the existing results/verify/gan_lr2e-7_seed*
# runs, so the only thing varying is the presence of the 92 unlabeled rows.
#
# Motivation: losses.py:54 averages the real/fake discriminator loss over ALL
# real rows, and the pool is 92 rows against 8243 labeled ones -- so the
# unlabeled data may contribute nothing distinct. If the +0.012 gain survives
# its removal, the method is an adversarial regularizer on labeled data and the
# "semi-supervised" framing does not hold.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
OUT_ROOT="$REPO_ROOT/results/verify"
DATA_DIR="$REPO_ROOT/data/scicite/gan_bert"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

MODEL="allenai/scibert_scivocab_uncased"
COMMON="--epochs 20 --max_seq_length 160 --batch_size 32 --num_trainable_layers 12 --dataset_name scicite"

# Keep only the highest-F1 checkpoint in each weight directory.
prune_checkpoints() {
  local run_dir="$1"
  for sub in transformer discriminator generator; do
    local d="$run_dir/$sub"
    [ -d "$d" ] || continue
    ls "$d"/*.pth 2>/dev/null \
      | sed -E 's/.*_f1_([0-9.]+)\.pth$/\1 &/' \
      | sort -k1,1g | head -n -1 | cut -d' ' -f2- | xargs -r rm -f
  done
}

run_gan_nounlabeled() {
  local seed="$1" gpu="$2"
  local name="gan_nounlabeled_lr2e-7_seed${seed}"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  ( cd "$GAN_BERT_SRC"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train.py \
      --labeled_csv "$DATA_DIR/labeled_train.csv" \
      --val_csv "$DATA_DIR/val.csv" \
      --test_csv "$DATA_DIR/test.csv" \
      --labels background method result unknown \
      --model_name "$MODEL" --output_dir "$out" \
      $COMMON --hidden_layers_g 1 --hidden_layers_d 1 --noise_size 768 \
      --dropout 0.20 --epsilon 2e-7 \
      --lr_d 2e-7 --lr_g 2e-7 --seed "$seed"
  ) > "$LOG_DIR/$name.log" 2>&1
  prune_checkpoints "$out"
  echo "   [gpu$gpu] done: $name"
}

echo "== tier 1: unlabeled ablation, 4 runs =="
( run_gan_nounlabeled 1 0; run_gan_nounlabeled 3 0 ) &
( run_gan_nounlabeled 2 1; run_gan_nounlabeled 42 1 ) &
wait
echo "== tier 1 finished =="
