# Unlabeled-pool ablation — is SS-cGAN actually semi-supervised?

**Question.** Does the SS-cGAN's benefit come from the unlabeled pool, or from the adversarial
objective applied to labeled data alone?

**Setup.** SciCite / SciBERT / full fine-tuning (12 layers), 20 epochs, batch 32,
`max_seq_length` 160, AdamW, constant schedule with 0.1 warmup, LR 2e-7 for both generator and
discriminator, seeds 1/2/3/42, deterministic. Test metrics from the best-validation-F1
checkpoint. The only thing that varies between arms (a) and (b) is the presence of the 92
unlabeled rows.

Run with `scripts/run_unlabeled_ablation.sh`, analysed with
`scripts/analyze_unlabeled_ablation.py`.

**Ablation confirmed genuine, two ways:** every arm-(b) `run_meta.json` records
`n_train_unlabeled: 0` with `n_train_labeled: 8243`, and the training logs show
**258 batches/epoch against the reference run's 261** — i.e. the 92 rows are removed from the
training stream, not silently carried along.

---

## Results

> **Revised.** The first version of this document compared arm (b) (deterministic) against
> arms (a) and (c) taken from `results/verify/`, whose seed-1/2/3 runs were made **before
> determinism was enforced** (`run_meta.json` shows `deterministic: null`). The arms therefore
> differed in one respect beyond the variable under test. The numbers below use the
> fully-deterministic seed-grid runs for (a) and (c), so all three arms are like-for-like.
> The verdict is unchanged and in fact strengthened; the significance of two secondary
> comparisons does not survive. Superseded figures are noted at the end.

Best-checkpoint test macro-F1, mean ± std over 4 seeds, all arms deterministic:

| arm | mean ± std | n |
|---|---|---|
| (a) SS-cGAN **with** unlabeled pool | **0.8788 ± 0.0078** | 4 |
| (b) SS-cGAN **without** unlabeled pool | **0.8768 ± 0.0060** | 4 |
| (c) no-GAN baseline | **0.8695 ± 0.0024** | 4 |

Per seed:

| seed | (a) with | (b) without | (c) no-GAN |
|---|---|---|---|
| 1 | 0.8685 | 0.8695 | 0.8663 |
| 2 | 0.8873 | 0.8839 | 0.8706 |
| 3 | 0.8787 | 0.8754 | 0.8719 |
| 42 | 0.8808 | 0.8785 | 0.8693 |

Note seed 1, where the arm *without* the unlabeled pool actually scores **higher** than the arm
with it (0.8695 vs 0.8685).

Validation-loss rebound (final − minimum), per seed — **0.000 in every run of every arm**,
except (b) seed 3 at +0.001 and (c) seed 2 at +0.001. At 2e-7 nothing overfits, so the rebound
carries no signal here and cannot distinguish the arms.

Best epoch (of 20): (a) 20, 16, 11, 20 — (b) 20, 16, 15, 20 — (c) 19, 20, 20, 18.
Most runs in all three arms peak at or near the end of the budget, which is what motivates the
separate 40-epoch convergence experiment.

### Statistics

| comparison | delta | t | p |
|---|---|---|---|
| **(a) − (b)** — does the unlabeled pool matter? | **+0.0020** | 0.40 | **0.70** |
| (b) − (c) — GAN without unlabeled vs no-GAN | **+0.0073** | 2.27 | **0.064** |
| (a) − (c) — the full claim | **+0.0093** | 2.28 | **0.063** |

---

## Verdict

**The gain is PRESERVED without unlabeled data.** Removing the entire unlabeled pool costs
**+0.0020** macro-F1 — indistinguishable from zero (p=0.70), and smaller than the gap between
two seeds of the same configuration. The SS-cGAN stripped of all unlabeled data retains
**+0.0073 of the full +0.0093 effect, i.e. 78 %**.

The like-for-like recomputation *strengthened* this conclusion: on the earlier mixed-provenance
comparison the pool appeared to be worth +0.0038 (p=0.42); with all arms deterministic it is
worth +0.0020 (p=0.70).

**Separately, note what does NOT survive:** neither (b) − (c) nor (a) − (c) reaches significance
at α=0.05 once all arms are deterministic (p=0.064 and p=0.063, against 0.042 and 0.013 before).
Those comparisons are about whether the *adversarial objective* helps at all — a different
question from this ablation — and they are addressed in `results/seed_grid/` and
`results_summary.md`. This document's claim is only that **whatever benefit exists does not come
from the unlabeled pool**, and that claim is now on firmer ground than before.

**The benefit is therefore an adversarial regulariser acting on labeled data, not
semi-supervision.** The "semi-supervised" framing of the method is not supported on SciCite.

This matches the mechanism visible in the code. `gan_bert/losses.py:54`:

```python
d_l_unsup_real = -1.0 * torch.mean(torch.log(1 - d_real_probs[:, -1] + epsilon))
```

The real/fake term averages over **all** real rows, labeled and unlabeled alike. Unlabeled rows
are not handled distinctly — they only add 92 more rows to an 8243-row real stream (1.12 %). The
ablation confirms the consequence: those rows contribute nothing measurable.

### Honest limits of this result

- **Absence of evidence, not evidence of absence.** With n=4 per arm and σ≈0.006–0.008, this
  design has low power against an effect the size of +0.0020. The point estimate is positive and
  accounts for ~22 % of the total gain, so a small genuine contribution from the unlabeled pool
  cannot be ruled out — only shown to be far from the main driver. Seed 1 reverses the sign
  outright.
- **SciCite-specific, and specific to this pool size.** The pool is 1.12 % of labeled data here,
  0.65 % on ACL-ARC and 1.69 % on 3C — all tiny. This says nothing about what a substantially
  larger unlabeled pool would do; the `gan_bert_improved_cross` runs (≈690 unlabeled rows) exist
  but were never seed-replicated.
- **One backbone, one learning rate.**

### Superseded figures

For traceability, the first version of this document reported, using
`results/verify/gan_lr2e-7_seed*` and `results/verify/baseline_lr2e-7_seed*` (seeds 1/2/3
non-deterministic) for arms (a) and (c):

| comparison | superseded | corrected |
|---|---|---|
| (a) mean ± std | 0.8806 ± 0.0064 | 0.8788 ± 0.0078 |
| (c) mean ± std | 0.8685 ± 0.0024 | 0.8695 ± 0.0024 |
| (a) − (b) | +0.0038, p=0.42 | **+0.0020, p=0.70** |
| (b) − (c) | +0.0083, p=0.042 | **+0.0073, p=0.064** |
| (a) − (c) | +0.0121, p=0.013 | **+0.0093, p=0.063** |

### Consequence for the thesis

Any sentence describing the contribution as *semi-supervised learning from unlabeled citation
contexts* overstates what the experiments show. The defensible claim is that the adversarial
discriminator acts as a regulariser on labeled data. If the semi-supervised framing is to be
kept, it needs an experiment with a materially larger unlabeled pool — which this project has
not run with seed replication.
