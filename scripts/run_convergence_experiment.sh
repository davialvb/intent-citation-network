#!/usr/bin/env bash
# Tier 2: is the SciCite gain real, or an under-convergence artifact?
#
# SciCite @ 2e-7 is the ONLY cell of the 26-run LR sweep where both arms have a
# validation-loss rebound of exactly 0.000, and the two arms end the budget in
# different states:
#
#   no-GAN   peaks at epoch 18/20, then flat   (0.8170 -> 0.8114 -> 0.8145)
#   SS-cGAN  peaks at epoch 20/20, still rising (0.8155 -> 0.8185 -> 0.8197)
#
# In every sweep cell where training actually converges, the GAN loses. So the
# +0.012 may be nothing more than the adversarial objective supplying gradient
# signal that a 2e-7 baseline cannot get on its own within 20 epochs.
#
# This doubles the budget to 40 epochs for both arms, 4 seeds each. If the
# baseline catches up, the headline result is a convergence artifact of an
# under-tuned learning rate. Everything else matches the tier-1 / verify runs.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
OUT_ROOT="$REPO_ROOT/results/seed_grid/convergence_40ep"
DATA_DIR="$REPO_ROOT/data/scicite/gan_bert"
LOG_DIR="$OUT_ROOT/logs"
JOBFILE="$OUT_ROOT/jobs.txt"
mkdir -p "$LOG_DIR"

MODEL="allenai/scibert_scivocab_uncased"
COMMON="--epochs 40 --max_seq_length 160 --batch_size 32 --num_trainable_layers 12 --dataset_name scicite"

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

run_job() {
  local arm="$1" seed="$2" gpu="$3"
  local name="${arm}_seed${seed}"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"

  if [ "$arm" = "gan" ]; then
    ( cd "$GAN_BERT_SRC"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train.py \
        --labeled_csv "$DATA_DIR/labeled_train.csv" \
        --unlabeled_csv "$DATA_DIR/unsupervised.csv" \
        --val_csv "$DATA_DIR/val.csv" --test_csv "$DATA_DIR/test.csv" \
        --labels background method result unknown \
        --model_name "$MODEL" --output_dir "$out" \
        $COMMON --hidden_layers_g 1 --hidden_layers_d 1 --noise_size 768 \
        --dropout 0.20 --epsilon 2e-7 \
        --lr_d 2e-7 --lr_g 2e-7 --seed "$seed"
    ) > "$LOG_DIR/$name.log" 2>&1
  else
    ( cd "$GAN_BERT_SRC"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train_baseline.py \
        --labeled_csv "$DATA_DIR/labeled_train.csv" \
        --val_csv "$DATA_DIR/val.csv" --test_csv "$DATA_DIR/test.csv" \
        --labels background method result \
        --model_name "$MODEL" --output_dir "$out" \
        $COMMON --lr 2e-7 --seed "$seed"
    ) > "$LOG_DIR/$name.log" 2>&1
  fi
  prune_checkpoints "$out"
  echo "   [gpu$gpu] done: $name"
}

# Shortest-first: the baseline arm is ~1.5x faster than the GAN arm.
: > "$JOBFILE"
for seed in 1 2 3 42; do echo "baseline $seed" >> "$JOBFILE"; done
for seed in 1 2 3 42; do echo "gan $seed"      >> "$JOBFILE"; done
touch "$JOBFILE.lock"

worker() {
  local gpu="$1"
  while true; do
    local job
    job=$(flock "$JOBFILE.lock" -c "head -n1 '$JOBFILE'; sed -i '1d' '$JOBFILE'")
    [ -z "$job" ] && break
    run_job $job "$gpu"
  done
}

echo "== tier 2: convergence experiment, 8 runs at 40 epochs =="
worker 0 &
worker 1 &
wait
echo "== tier 2 finished =="
