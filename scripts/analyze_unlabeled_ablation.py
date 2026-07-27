#!/usr/bin/env python
"""Three-arm comparison for the unlabeled-pool ablation (tier 1).

Arms, all SciCite / SciBERT / full fine-tuning @ LR 2e-7, seeds 1/2/3/42:

  (a) SS-cGAN WITH the 92-row unlabeled pool   results/verify/gan_lr2e-7_seed*
  (b) SS-cGAN WITHOUT it                       results/verify/gan_nounlabeled_lr2e-7_seed*
  (c) no-GAN baseline                          results/verify/baseline_lr2e-7_seed*

The seed-42 cells of (a) and (c) live in results/lr_sweep/scicite/ rather than
results/verify/, because that is where the deterministic seed-42 runs were made.

Reports best-checkpoint test macro-F1 as mean +/- std, the per-seed
validation-loss rebound, and a two-sample t-test between (a) and (b) -- i.e.
does removing the unlabeled data change anything?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "results" / "verify"
SWEEP = REPO / "results" / "lr_sweep" / "scicite"
SEEDS = [1, 2, 3, 42]

ARMS = {
    "(a) SS-cGAN + unlabeled": {
        s: (SWEEP / "gan_lr2e-7" if s == 42 else VERIFY / f"gan_lr2e-7_seed{s}")
        for s in SEEDS
    },
    "(b) SS-cGAN, no unlabeled": {
        s: VERIFY / f"gan_nounlabeled_lr2e-7_seed{s}" for s in SEEDS
    },
    "(c) no-GAN baseline": {
        s: (SWEEP / "baseline_lr2e-7" if s == 42 else VERIFY / f"baseline_lr2e-7_seed{s}")
        for s in SEEDS
    },
}


def read(run_dir: Path) -> dict | None:
    test, hist, meta = (run_dir / n for n in ("test_metrics.json", "history.json", "run_meta.json"))
    if not test.exists():
        return None
    t = json.loads(test.read_text())
    h = json.loads(hist.read_text()) if hist.exists() else []
    m = json.loads(meta.read_text()) if meta.exists() else {}
    vl = [x["val_avg_loss"] for x in h]
    return {
        "f1": t["f1_macro"],
        "selection": t.get("selection"),
        "rebound": (vl[-1] - min(vl)) if vl else float("nan"),
        "n_unlabeled": m.get("n_train_unlabeled"),
        "best_epoch": m.get("best_epoch"),
    }


def main() -> None:
    data: dict[str, dict[int, dict]] = {}
    for arm, paths in ARMS.items():
        got = {s: r for s, p in paths.items() if (r := read(p))}
        data[arm] = got

    print(f"{'arm':28}{'seed':>6}{'testF1':>10}{'rebound':>10}{'unlab':>8}{'bestEp':>8}  selection")
    for arm, runs in data.items():
        for s in SEEDS:
            r = runs.get(s)
            if r is None:
                print(f"{arm:28}{s:>6}{'--- missing ---':>10}")
                continue
            print(
                f"{arm:28}{s:>6}{r['f1']:>10.4f}{r['rebound']:>+10.3f}"
                f"{str(r['n_unlabeled']):>8}{str(r['best_epoch']):>8}  {r['selection']}"
            )
        print()

    print("=" * 78)
    summary = {}
    for arm, runs in data.items():
        if len(runs) < 2:
            print(f"{arm:28} incomplete ({len(runs)}/4 seeds)")
            continue
        v = np.array([runs[s]["f1"] for s in sorted(runs)])
        summary[arm] = v
        print(f"{arm:28} n={len(v)}  mean={v.mean():.4f}  std={v.std(ddof=1):.4f}")

    a = summary.get("(a) SS-cGAN + unlabeled")
    b = summary.get("(b) SS-cGAN, no unlabeled")
    c = summary.get("(c) no-GAN baseline")
    if a is not None and b is not None:
        t, p = stats.ttest_ind(a, b)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else float("nan")
        print(f"\n(a) - (b)  [does the unlabeled pool matter?]  "
              f"delta={a.mean() - b.mean():+.4f}  t={t:.2f}  p={p:.4f}  d={d:.2f}")
    if b is not None and c is not None:
        t, p = stats.ttest_ind(b, c)
        print(f"(b) - (c)  [GAN without unlabeled vs no-GAN]  "
              f"delta={b.mean() - c.mean():+.4f}  t={t:.2f}  p={p:.4f}")
    if a is not None and c is not None:
        print(f"(a) - (c)  [the published-style claim]        "
              f"delta={a.mean() - c.mean():+.4f}")


if __name__ == "__main__":
    main()
