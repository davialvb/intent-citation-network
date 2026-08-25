#!/usr/bin/env python
"""Paired comparison for the conditional-generator ablation.

Arms, both SciCite / SciBERT / full fine-tuning @ LR 2e-7 / opt_scheme=decoupled,
seeds 1/2/3/42, small 92-row unlabeled pool -- the only thing varying is
--generator_type:

  (a) conditional generator (noise + numeric label id)   results/conditional_generator_ablation/conditional_seed*
  (b) unconditional generator (noise only)                results/conditional_generator_ablation/unconditional_seed*

Reports best-checkpoint test macro-F1 as mean +/- std, per-seed values, and a
paired t-test between (a) and (b).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "results" / "conditional_generator_ablation"
SEEDS = [1, 2, 3, 42]

ARMS = {
    "(a) conditional generator": {s: OUT_ROOT / f"conditional_seed{s}" for s in SEEDS},
    "(b) unconditional generator": {s: OUT_ROOT / f"unconditional_seed{s}" for s in SEEDS},
}


def read(run_dir: Path) -> dict | None:
    test, hist, meta = (run_dir / n for n in ("test_metrics.json", "history.json", "run_meta.json"))
    if not test.exists():
        return None
    t = json.loads(test.read_text())
    h = json.loads(hist.read_text()) if hist.exists() else []
    m = json.loads(meta.read_text()) if meta.exists() else {}
    margins = [x.get("real_fake_margin") for x in h if x.get("real_fake_margin") is not None]
    return {
        "f1": t["f1_macro"],
        "selection": t.get("selection"),
        "opt_scheme": m.get("opt_scheme"),
        "generator_type": m.get("generator_type"),
        "best_epoch": m.get("best_epoch"),
        "final_margin": margins[-1] if margins else float("nan"),
    }


def main() -> None:
    data: dict[str, dict[int, dict]] = {}
    for arm, paths in ARMS.items():
        data[arm] = {s: r for s, p in paths.items() if (r := read(p))}

    print(f"{'arm':28}{'seed':>6}{'testF1':>10}{'bestEp':>8}{'margin':>10}  opt_scheme / generator_type")
    for arm, runs in data.items():
        for s in SEEDS:
            r = runs.get(s)
            if r is None:
                print(f"{arm:28}{s:>6}{'--- missing ---':>10}")
                continue
            print(
                f"{arm:28}{s:>6}{r['f1']:>10.4f}{str(r['best_epoch']):>8}{r['final_margin']:>10.3f}  "
                f"{r['opt_scheme']} / {r['generator_type']}  ({r['selection']})"
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

    a = summary.get("(a) conditional generator")
    b = summary.get("(b) unconditional generator")
    if a is not None and b is not None and len(a) == len(b):
        t_paired, p_paired = stats.ttest_rel(a, b)
        t_ind, p_ind = stats.ttest_ind(a, b)
        print(
            f"\n(a) - (b)  [does conditioning the generator help?]  "
            f"delta={a.mean() - b.mean():+.4f}  paired t={t_paired:.2f} p={p_paired:.4f}  "
            f"(unpaired t={t_ind:.2f} p={p_ind:.4f})"
        )


if __name__ == "__main__":
    main()
