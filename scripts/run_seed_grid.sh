#!/usr/bin/env bash
# Tiers 3-8: seed-replicated 2-arm grid.
#
#   3 backbones x 3 datasets x 2 arms x 4 seeds = 72 runs, full fine-tuning.
#
# Every cell in the thesis outside SciCite/SciBERT is single-seed, and measured
# run-to-run variance before determinism was enforced was 0.022 macro-F1 --
# larger than most of the effects being claimed. This grid puts error bars on
# all of them.
#
# ONE learning rate per dataset, shared by both arms, so the comparison is
# matched by construction and cannot reproduce the confound the audit found:
#
#   SciCite  2e-7  grid optimum for BOTH arms in the 26-run sweep; also the
#                  reference paper's a-priori value.
#   ACL-ARC  2e-5  ties for best validation F1 in the sweep (0.8442) and is the
#                  conventional BERT fine-tuning default.
#   3C       2e-5  best validation F1 (0.4655) and best test F1 (0.4269) for the
#                  baseline; mid-grid for the GAN.
#
# CAVEAT: these LRs come from a SciBERT-only sweep and are applied to SPECTER2
# and XLNet untested. That is a stated assumption, not a tuned result.
#
# The GAN-improved arm (consistency_weight=1.0, label_smoothing=0.1) is NOT run:
# against vanilla on the existing full-FT runs it averages -0.0091 and wins only
# 3 of 9 cells, all within single-seed noise.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAN_BERT_SRC="$REPO_ROOT/src/gan_bert_src"
OUT_ROOT="$REPO_ROOT/results/seed_grid"
LOG_DIR="$OUT_ROOT/logs"
JOBFILE="$OUT_ROOT/jobs.txt"
mkdir -p "$LOG_DIR"

SEEDS="1 2 3 42"

declare -A MODELS=(
  [scibert]="allenai/scibert_scivocab_uncased"
  [specter2]="allenai/specter2_base"
  [xlnet]="xlnet-base-cased"
)

# per-dataset fixed hyperparameters (unchanged from the thesis' runs)
declare -A EPOCHS=(  [scicite]=20   [acl-arc]=30   [3C]=30   )
declare -A BATCH=(   [scicite]=32   [acl-arc]=16   [3C]=16   )
declare -A MAXSEQ=(  [scicite]=160  [acl-arc]=64   [3C]=64   )
declare -A HG=(      [scicite]=1    [acl-arc]=2    [3C]=2    )
declare -A HD=(      [scicite]=1    [acl-arc]=1    [3C]=1    )
declare -A NOISE=(   [scicite]=768  [acl-arc]=100  [3C]=100  )
declare -A DROPOUT=( [scicite]=0.20 [acl-arc]=0.10 [3C]=0.10 )
declare -A GRATIO=(  [scicite]=1    [acl-arc]=10   [3C]=10   )
declare -A LR=(      [scicite]=2e-7 [acl-arc]=2e-5 [3C]=2e-5 )
declare -A LABELS=(
  [scicite]="background method result"
  [acl-arc]="background uses compareorcontrast motivation extends future"
  [3C]="background comparecontrast extension future motivation uses"
)

# Relative cost, used only to sort the queue shortest-first (see below).
# Base cost is roughly proportional to dataset size x sequence length; XLNet is
# ~2.1x SciBERT/SPECTER2. Measured from wall_clock_seconds of existing runs.
declare -A DSCOST=(  [acl-arc]=26  [3C]=35  [scicite]=144 )
declare -A BBCOST=(  [scibert]=10  [specter2]=10  [xlnet]=21 )

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

# ---- build the job list, sorted shortest-first --------------------------
# Both workers pop from the same locked queue, so sorting the file is all that
# is needed to get shortest-first scheduling across both GPUs.
TMPJOBS=$(mktemp)
for ds in scicite acl-arc 3C; do
  for bb in scibert specter2 xlnet; do
    for arm in baseline gan; do
      # GAN arm costs ~1.4x the baseline (generator + discriminator updates).
      cost=$(( DSCOST[$ds] * BBCOST[$bb] ))
      [ "$arm" = "gan" ] && cost=$(( cost * 14 / 10 ))
      for seed in $SEEDS; do
        echo "$cost $ds $bb $arm $seed" >> "$TMPJOBS"
      done
    done
  done
done
sort -n "$TMPJOBS" | cut -d' ' -f2- > "$JOBFILE"
rm -f "$TMPJOBS"
TOTAL=$(wc -l < "$JOBFILE")
echo "== seed grid: $TOTAL runs, shortest-first =="

run_job() {
  local ds="$1" bb="$2" arm="$3" seed="$4" gpu="$5"
  local name="${ds}_${bb}_${arm}_seed${seed}"
  local out="$OUT_ROOT/$ds/$bb/${arm}_seed${seed}"
  local data="$REPO_ROOT/data/$ds/gan_bert"
  local lr="${LR[$ds]}"
  mkdir -p "$out"

  if [ "$arm" = "gan" ]; then
    local lrg; lrg=$(gen_lr "$lr" "${GRATIO[$ds]}")
    ( cd "$GAN_BERT_SRC"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train.py \
        --labeled_csv "$data/labeled_train.csv" --unlabeled_csv "$data/unsupervised.csv" \
        --val_csv "$data/val.csv" --test_csv "$data/test.csv" \
        --labels ${LABELS[$ds]} unknown \
        --model_name "${MODELS[$bb]}" --output_dir "$out" \
        --epochs "${EPOCHS[$ds]}" --max_seq_length "${MAXSEQ[$ds]}" --batch_size "${BATCH[$ds]}" \
        --hidden_layers_g "${HG[$ds]}" --hidden_layers_d "${HD[$ds]}" --noise_size "${NOISE[$ds]}" \
        --dropout "${DROPOUT[$ds]}" --epsilon 2e-7 \
        --lr_d "$lr" --lr_g "$lrg" \
        --num_trainable_layers 12 --seed "$seed" --dataset_name "$ds"
    ) > "$LOG_DIR/$name.log" 2>&1
  else
    ( cd "$GAN_BERT_SRC"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 uv run --project "$REPO_ROOT" python cli_train_baseline.py \
        --labeled_csv "$data/labeled_train.csv" \
        --val_csv "$data/val.csv" --test_csv "$data/test.csv" \
        --labels ${LABELS[$ds]} \
        --model_name "${MODELS[$bb]}" --output_dir "$out" \
        --epochs "${EPOCHS[$ds]}" --max_seq_length "${MAXSEQ[$ds]}" --batch_size "${BATCH[$ds]}" \
        --lr "$lr" --num_trainable_layers 12 --seed "$seed" --dataset_name "$ds"
    ) > "$LOG_DIR/$name.log" 2>&1
  fi
  prune_checkpoints "$out"
  echo "   [gpu$gpu] done: $name"
}

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
echo "== seed grid finished =="
