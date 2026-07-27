#!/usr/bin/env python
"""Aggregates results/lr_sweep/ into the tables used by results_summary.md.

For every (dataset, arm, learning rate) cell it reports the best validation
macro-F1, the epoch it occurred at, the validation-loss rebound, and the test
macro-F1 of the best-validation checkpoint.

Selection rule: the winning learning rate per (dataset, arm) is chosen by
**validation** macro-F1. Choosing it by test F1 would be selecting on the test
set, so the test column is reported but never used to pick. Where the two
disagree the script says so explicitly, because that disagreement is itself a
finding on these splits.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "results" / "lr_sweep"
DATASETS = ["scicite", "acl-arc", "3C"]
ARMS = ["baseline", "gan"]
ARM_LABEL = {"baseline": "no-GAN", "gan": "SS-cGAN"}


def lr_key(name: str) -> float:
    return float(name.split("_lr")[-1])


def load(run_dir: Path) -> dict | None:
    hist = run_dir / "history.json"
    test = run_dir / "test_metrics.json"
    if not hist.exists() or not test.exists():
        return None
    h = json.loads(hist.read_text())
    t = json.loads(test.read_text())
    vl = [x["val_avg_loss"] for x in h]
    vf = [x["val_f1_macro"] for x in h]
    return {
        "lr": lr_key(run_dir.name),
        "arm": "gan" if run_dir.name.startswith("gan") else "baseline",
        "best_val_f1": max(vf),
        "best_val_epoch": vf.index(max(vf)) + 1,
        "rebound": vl[-1] - min(vl),
        "test_f1": t["f1_macro"],
        "test_acc": t["accuracy"],
        "selection": t.get("selection"),
        "epochs_done": len(h),
    }


def collect() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for ds in DATASETS:
        d = SWEEP / ds
        if not d.is_dir():
            continue
        runs = [r for r in (load(p) for p in sorted(d.iterdir()) if p.is_dir()) if r]
        if runs:
            out[ds] = sorted(runs, key=lambda r: (r["arm"], r["lr"]))
    return out


def main() -> None:
    data = collect()
    if not data:
        print("no completed sweep runs yet")
        return

    total = sum(len(v) for v in data.values())
    print(f"completed sweep runs: {total} / 26\n")

    for ds, runs in data.items():
        print("=" * 78)
        print(f"{ds}")
        print("=" * 78)
        print(f"{'arm':9}{'LR':10}{'bestValF1':12}{'@ep':6}{'rebound':10}{'testF1':10}testAcc")
        for r in runs:
            print(
                f"{ARM_LABEL[r['arm']]:9}{r['lr']:<10.1e}{r['best_val_f1']:<12.4f}"
                f"{r['best_val_epoch']:<6}{r['rebound']:<+10.3f}{r['test_f1']:<10.4f}{r['test_acc']:.4f}"
            )

        # winner per arm, selected on validation
        print()
        winners = {}
        for arm in ARMS:
            arm_runs = [r for r in runs if r["arm"] == arm]
            if not arm_runs:
                continue
            by_val = max(arm_runs, key=lambda r: r["best_val_f1"])
            by_test = max(arm_runs, key=lambda r: r["test_f1"])
            winners[arm] = by_val
            note = ""
            if by_val["lr"] != by_test["lr"]:
                note = (
                    f"   [val/test DISAGREE: test would pick lr={by_test['lr']:.1e} "
                    f"(test {by_test['test_f1']:.4f} vs {by_val['test_f1']:.4f})]"
                )
            print(
                f"  best {ARM_LABEL[arm]:8} by val: lr={by_val['lr']:.1e}  "
                f"val={by_val['best_val_f1']:.4f}  test={by_val['test_f1']:.4f}{note}"
            )

        if len(winners) == 2:
            g, b = winners["gan"], winners["baseline"]
            print(
                f"\n  tuned-vs-tuned (each arm at its own val-selected LR): "
                f"SS-cGAN {g['test_f1']:.4f} - no-GAN {b['test_f1']:.4f} = {g['test_f1'] - b['test_f1']:+.4f}"
            )
            same = [
                (r_g, r_b)
                for r_g in runs if r_g["arm"] == "gan"
                for r_b in runs if r_b["arm"] == "baseline" and r_b["lr"] == r_g["lr"]
            ]
            if same:
                best_matched = max(same, key=lambda pr: pr[0]["best_val_f1"])
                print(
                    f"  matched-LR at the SS-cGAN's val-selected LR "
                    f"({best_matched[0]['lr']:.1e}): "
                    f"{best_matched[0]['test_f1'] - best_matched[1]['test_f1']:+.4f}"
                )
        print()


if __name__ == "__main__":
    main()
