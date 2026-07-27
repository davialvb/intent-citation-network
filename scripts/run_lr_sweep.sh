#!/usr/bin/env bash
# Learning-rate sweep for BOTH arms on all three datasets.
#
# Motivation (see methodology_findings.md): the published SciCite SS-cGAN ran at
# 2e-7, a value inherited from the reference paper, and was still improving at
# the end of its epoch budget -- i.e. under-converged rather than regularized.
# Its no-GAN baseline ran at 2e-5 and was, separately, worse than the same
# baseline at 2e-7. Neither arm was tuned on this data.
#
# This sweep tunes BOTH arms over the SAME per-dataset grid, so the resulting
# comparison is learning-rate-matched by construction and cannot reproduce the
# confound the audit found. Backbone is fixed to SciBERT and fine-tuning to
# full (12 layers); all runs are deterministic (seed 42) and report test metrics
# from the best-validation-F1 checkpoint.
#
# Generator LR follows each dataset's existing convention: equal to the
# discriminator LR on SciCite, 10x it on ACL-ARC/3C. Decoupling them is a
# further experiment, not attempted here.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
OUT_ROOT="$REPO_ROOT/results/lr_sweep"
LOG_DIR="$OUT_ROOT/logs"
JOBFILE="$OUT_ROOT/jobs.txt"
mkdir -p "$LOG_DIR"

MODEL="allenai/scibert_scivocab_uncased"
SEED="${SEED:-42}"

# per-dataset fixed hyperparameters (unchanged from the thesis' runs)
declare -A EPOCHS=(  [scicite]=20  [acl-arc]=30  [3C]=30  )
declare -A BATCH=(   [scicite]=32  [acl-arc]=16  [3C]=16  )
declare -A MAXSEQ=(  [scicite]=160 [acl-arc]=64  [3C]=64  )
declare -A HG=(      [scicite]=1   [acl-arc]=2   [3C]=2   )
declare -A HD=(      [scicite]=1   [acl-arc]=1   [3C]=1   )
declare -A NOISE=(   [scicite]=768 [acl-arc]=100 [3C]=100 )
declare -A DROPOUT=( [scicite]=0.20 [acl-arc]=0.10 [3C]=0.10 )
declare -A GRATIO=(  [scicite]=1   [acl-arc]=10  [3C]=10  )
declare -A LABELS=(
  [scicite]="background method result"
  [acl-arc]="background uses compareorcontrast motivation extends future"
  [3C]="background comparecontrast extension future motivation uses"
)
# LR grids: centred on each dataset's published value, extended toward the
# region the audit suggests is under-explored.
declare -A GRID=(
  [scicite]="2e-7 1e-6 5e-6 1e-5 2e-5"
  [acl-arc]="1e-5 2e-5 5e-5 1e-4"
  [3C]="1e-5 2e-5 5e-5 1e-4"
)

# Generator LR = discriminator LR * GRATIO, formatted for the CLI.
gen_lr() { python3 -c "print('%g' % ($1 * $2))"; }

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

# ---- build the job list -------------------------------------------------
: > "$JOBFILE"
for ds in scicite acl-arc 3C; do
  for lr in ${GRID[$ds]}; do
    echo "$ds gan $lr"      >> "$JOBFILE"
    echo "$ds baseline $lr" >> "$JOBFILE"
  done
done
TOTAL=$(wc -l < "$JOBFILE")
echo "== LR sweep: $TOTAL runs =="

run_job() {
  local ds="$1" arm="$2" lr="$3" gpu="$4"
  local name="${ds}_${arm}_lr${lr}"
  local out="$OUT_ROOT/$ds/${arm}_lr${lr}"
  local data="$REPO_ROOT/data/$ds/gan_bert"
  mkdir -p "$out"

  if [ "$arm" = "gan" ]; then
    local lrg; lrg=$(gen_lr "$lr" "${GRATIO[$ds]}")
    ( cd "$GAN_BERT_SRC"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train.py \
        --labeled_csv "$data/labeled_train.csv" --unlabeled_csv "$data/unsupervised.csv" \
        --val_csv "$data/val.csv" --test_csv "$data/test.csv" \
        --labels ${LABELS[$ds]} unknown \
        --model_name "$MODEL" --output_dir "$out" \
        --epochs "${EPOCHS[$ds]}" --max_seq_length "${MAXSEQ[$ds]}" --batch_size "${BATCH[$ds]}" \
        --hidden_layers_g "${HG[$ds]}" --hidden_layers_d "${HD[$ds]}" --noise_size "${NOISE[$ds]}" \
        --dropout "${DROPOUT[$ds]}" --epsilon 2e-7 \
        --lr_d "$lr" --lr_g "$lrg" \
        --num_trainable_layers 12 --seed "$SEED" --dataset_name "$ds"
    ) > "$LOG_DIR/$name.log" 2>&1
  else
    ( cd "$GAN_BERT_SRC"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train_baseline.py \
        --labeled_csv "$data/labeled_train.csv" \
        --val_csv "$data/val.csv" --test_csv "$data/test.csv" \
        --labels ${LABELS[$ds]} \
        --model_name "$MODEL" --output_dir "$out" \
        --epochs "${EPOCHS[$ds]}" --max_seq_length "${MAXSEQ[$ds]}" --batch_size "${BATCH[$ds]}" \
        --lr "$lr" --num_trainable_layers 12 --seed "$SEED" --dataset_name "$ds"
    ) > "$LOG_DIR/$name.log" 2>&1
  fi
  prune_checkpoints "$out"
  echo "   [gpu$gpu] done: $name"
}

# ---- two workers, each popping the next job under a lock ----------------
worker() {
  local gpu="$1"
  while true; do
    local job
    job=$(flock "$JOBFILE.lock" -c "head -n1 '$JOBFILE'; sed -i '1d' '$JOBFILE'")
    [ -z "$job" ] && break
    run_job $job "$gpu"
  done
}

touch "$JOBFILE.lock"
worker 0 &
worker 1 &
wait
echo "== LR sweep finished =="
