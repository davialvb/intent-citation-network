#!/usr/bin/env python
"""Aggregates the verification runs in results/verify/ into the tables used by
methodology_findings.md.

Reports, for the SciCite / SciBERT / full-fine-tuning cell:
  - validation-loss trajectories and rebound (final - minimum) per arm and LR;
  - test macro-F1 as reported by the training script (final-epoch weights) and,
    where checkpoints survive, at the best-validation-F1 epoch;
  - across-seed mean/std for both arms.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "results" / "verify"


def load(run_dir: Path) -> dict | None:
    hist_p = run_dir / "history.json"
    if not hist_p.exists():
        return None
    hist = json.loads(hist_p.read_text())
    vl = [h["val_avg_loss"] for h in hist]
    vf = [h["val_f1_macro"] for h in hist]
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    test_p = run_dir / "test_metrics.json"
    test = json.loads(test_p.read_text()) if test_p.exists() else {}
    return {
        "name": run_dir.name,
        "epochs_done": len(hist),
        "val_loss": vl,
        "val_f1": vf,
        "min_val_loss": min(vl),
        "argmin_epoch": vl.index(min(vl)) + 1,
        "final_val_loss": vl[-1],
        "rebound": vl[-1] - min(vl),
        "best_val_f1": max(vf),
        "best_val_f1_epoch": vf.index(max(vf)) + 1,
        "test_f1_final_epoch": test.get("f1_macro"),
        "seed": meta.get("seed"),
    }


def row(r: dict) -> str:
    return (
        f"{r['name']:26} seed={str(r['seed']):>3} "
        f"minVL={r['min_val_loss']:.3f}@e{r['argmin_epoch']:<3} "
        f"finalVL={r['final_val_loss']:.3f} rebound={r['rebound']:+.3f} "
        f"bestValF1={r['best_val_f1']:.4f}@e{r['best_val_f1_epoch']:<3} "
        f"testF1(final-epoch)={r['test_f1_final_epoch']}"
    )


def main() -> None:
    runs = []
    for d in sorted(VERIFY.iterdir()):
        if not d.is_dir() or d.name == "logs":
            continue
        # the very first B run was written one level deeper
        for cand in (d, d / "scicite" / "scibert"):
            r = load(cand)
            if r:
                r["name"] = d.name
                runs.append(r)
                break

    print("=" * 110)
    print("VERIFICATION RUNS -- SciCite / SciBERT / full fine-tuning / 20 epochs")
    print("=" * 110)
    for r in runs:
        print(row(r))

    print()
    print("Validation-loss trajectories:")
    for r in runs:
        print(f"  {r['name']:26} " + " ".join(f"{v:.3f}" for v in r["val_loss"]))
    print()
    print("Validation macro-F1 trajectories:")
    for r in runs:
        print(f"  {r['name']:26} " + " ".join(f"{v:.3f}" for v in r["val_f1"]))

    # across-seed summaries
    import statistics as st

    for prefix in ("gan_lr2e-7_seed", "baseline_lr2e-5_seed"):
        group = [r for r in runs if r["name"].startswith(prefix)]
        if len(group) < 2:
            continue
        f1s = [r["test_f1_final_epoch"] for r in group if r["test_f1_final_epoch"] is not None]
        rbs = [r["rebound"] for r in group]
        print()
        print(f"{prefix}*  n={len(group)}")
        print(f"  test macro-F1 (final epoch): mean={st.mean(f1s):.4f} std={st.stdev(f1s):.4f} "
              f"min={min(f1s):.4f} max={max(f1s):.4f}")
        print(f"  rebound (final - min val loss): mean={st.mean(rbs):+.3f} std={st.stdev(rbs):.3f} "
              f"| rebounds (>0.05) in {sum(1 for x in rbs if x > 0.05)}/{len(rbs)} runs")


if __name__ == "__main__":
    main()
