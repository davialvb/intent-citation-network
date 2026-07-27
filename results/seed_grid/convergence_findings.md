# Convergence experiment — is the SciCite gain robust to the epoch budget?

**Question.** The SS-cGAN's validation F1 was still climbing at epoch 20/20 while the no-GAN
baseline had plateaued at epoch 18. Is the +0.0121 gain simply an artifact of stopping the
budget at a point that favours the GAN — i.e. would the baseline catch up given more epochs?

**Setup.** SciCite / SciBERT / full fine-tuning, LR 2e-7 both arms, seeds 1/2/3/42,
deterministic, best-validation-F1 checkpoint. The only change is `--epochs 20` → `--epochs 40`.
Run with `scripts/run_convergence_experiment.sh`.

---

## Results

Test macro-F1 per seed, best-checkpoint (best epoch in parentheses):

| arm | epochs | seed 1 | seed 2 | seed 3 | seed 42 | mean ± std | mean best val F1 |
|---|---|---|---|---|---|---|---|
| no-GAN | 20 | 0.8652 (e19) | 0.8688 (e20) | 0.8708 (e20) | 0.8693 (e18) | **0.8685 ± 0.0024** | 0.8260 |
| no-GAN | 40 | 0.8585 (e38) | 0.8574 (e34) | 0.8586 (e40) | 0.8595 (e39) | **0.8585 ± 0.0008** | 0.8315 |
| SS-cGAN | 20 | 0.8722 (e20) | 0.8879 (e16) | 0.8816 (e11) | 0.8808 (e20) | **0.8806 ± 0.0064** | 0.8199 |
| SS-cGAN | 40 | 0.8622 (e36) | 0.8674 (e38) | 0.8539 (e38) | 0.8712 (e32) | **0.8637 ± 0.0075** | 0.8307 |

| budget | GAN − baseline | t | p | Cohen's d |
|---|---|---|---|---|
| **20 epochs** | **+0.0121** | 3.53 | **0.012** | 2.50 |
| **40 epochs** | **+0.0051** | 1.36 | **0.22 — not significant** | 0.96 |

Effect of doubling the budget, per arm:

| arm | test F1 | best val F1 |
|---|---|---|
| no-GAN | **−0.0100** | **+0.0055** |
| SS-cGAN | **−0.0170** | **+0.0108** |

---

## Verdict

**The original hypothesis is refuted, but the result is worse for the thesis than if it had been
confirmed.**

The baseline does *not* catch up — it also degrades. What actually happens is that **more
training makes validation F1 go up and test F1 go down, for both arms**, and the SS-cGAN
degrades faster (−0.0170 vs −0.0100). The consequence is that **the headline effect halves and
loses statistical significance when the epoch budget is doubled**: +0.0121 (p=0.012) at 20
epochs becomes +0.0051 (p=0.22) at 40.

So the +0.0121 is not a stable property of the adversarial objective. It is contingent on an
epoch budget of exactly 20 — a value inherited from the reference paper, never tuned on this
data, and which the audit had no reason to question until now.

### Why this happens

This is the SciCite train/val/test distribution mismatch acting directly, and this experiment is
a controlled before/after demonstration of it. SciCite is the only one of the three datasets
whose test split has a different label distribution from train and validation:

| split | background | method | result |
|---|---|---|---|
| train | 58.7 % | 27.8 % | 13.5 % |
| val | 58.4 % | 28.0 % | 13.6 % |
| **test** | **53.6 %** | **32.5 %** | **13.9 %** |

Additional training fits the train-side distribution harder. Validation, which shares that
distribution, rewards it (+0.0055 / +0.0108). Test, which does not, punishes it
(−0.0100 / −0.0170). Every 40-epoch run selected a checkpoint at epoch 32–40, so validation was
still signalling "keep going" the entire time test was getting worse.

The SS-cGAN degrades faster because it fits harder — consistent with it being an adversarial
regulariser that *accelerates* fitting rather than restraining it, which is also what the
rebound analysis in `results_summary.md` §4 found.

### What this does and does not say

- **Does not** say the SS-cGAN provides no benefit. At the thesis' actual configuration
  (20 epochs) the +0.0121 is real, replicated over 4 seeds, and significant at p=0.012.
- **Does** say the benefit is fragile to a hyperparameter nobody tuned, and that the direction
  of fragility is unfavourable — the effect shrinks with more training, it does not grow.
- **Does** mean any claim of the form "the adversarial objective improves citation-intent
  classification" must be stated as conditional on the training budget, or the thesis needs to
  justify why 20 epochs is the right budget on grounds other than "the reference paper used it".
- **Does not** generalise beyond SciCite/SciBERT — one dataset, one backbone, one LR.

### Interaction with the unlabeled ablation

Read together with `results/verify/unlabeled_ablation_findings.md`, the two tier-1/tier-2 results
say the SciCite effect is (a) not coming from unlabeled data, and (b) not robust to the epoch
budget. What remains defensible is narrow: *at 20 epochs on SciCite with SciBERT, an adversarial
discriminator on labeled data gives a small, replicated improvement over an LR-matched
baseline.*
