#!/usr/bin/env bash
# Is the conditional generator (noise + numeric label id) doing anything, or
# is it dead weight that can be replaced by a plain noise-only generator?
#
# SciCite / SciBERT / full fine-tuning (12 layers) / LR 2e-7, seeds 1/2/3/42,
# --opt_scheme decoupled (the fix that was missing from HEAD until this was
# reimplemented -- see memory), small 92-row unlabeled pool. Everything is
# byte-identical to results/verify/gan_lr2e-7_seed* except --generator_type.
#
# Motivation: generator_loss and discriminator_loss are both computed as
# batch-aggregate means and never reference the condition, so a generator that
# ignores its label input entirely minimizes both loss terms exactly as well
# as one that uses it. Prior: expect no significant difference.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
OUT_ROOT="$REPO_ROOT/results/conditional_generator_ablation"
DATA_DIR="$REPO_ROOT/data/scicite/gan_bert"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

MODEL="allenai/scibert_scivocab_uncased"
COMMON="--epochs 20 --max_seq_length 160 --batch_size 32 --num_trainable_layers 12 --dataset_name scicite"

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

run_one() {
  local gen_type="$1" seed="$2" gpu="$3"
  local name="${gen_type}_seed${seed}"
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
      --lr_d 2e-7 --lr_g 2e-7 --seed "$seed" \
      --opt_scheme decoupled --generator_type "$gen_type"
  ) > "$LOG_DIR/$name.log" 2>&1
  prune_checkpoints "$out"
  echo "   [gpu$gpu] done: $name"
}

echo "== conditional-generator ablation: 8 runs (2 arms x 4 seeds) =="
( run_one conditional   1 0; run_one conditional   2 0; run_one conditional   3 0; run_one conditional   42 0 ) &
( run_one unconditional 1 1; run_one unconditional 2 1; run_one unconditional 3 1; run_one unconditional 42 1 ) &
wait
echo "== ablation finished =="
