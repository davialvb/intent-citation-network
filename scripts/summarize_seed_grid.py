#!/usr/bin/env python
"""Aggregates results/seed_grid/ into the seed-replicated comparison tables.

For every (dataset, backbone) cell it reports both arms as mean +/- std over the
available seeds, the matched-LR delta, a two-sample t-test and Cohen's d.

Both arms run at the same learning rate within a dataset, so every delta here is
learning-rate-matched by construction -- unlike the thesis' published tables.

Cells with fewer than 2 seeds per arm are reported but not tested.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results" / "seed_grid"
DATASETS = ["scicite", "acl-arc", "3C"]
BACKBONES = ["scibert", "specter2", "xlnet"]
ARMS = ["baseline", "gan"]
ARM_LABEL = {"baseline": "no-GAN", "gan": "SS-cGAN"}
SEEDS = [1, 2, 3, 42]
LR = {"scicite": "2e-7", "acl-arc": "2e-5", "3C": "2e-5"}


def read(run_dir: Path) -> dict | None:
    test = run_dir / "test_metrics.json"
    if not test.exists():
        return None
    t = json.loads(test.read_text())
    meta = run_dir / "run_meta.json"
    m = json.loads(meta.read_text()) if meta.exists() else {}
    hist = run_dir / "history.json"
    vl = [x["val_avg_loss"] for x in json.loads(hist.read_text())] if hist.exists() else []
    return {
        "f1": t["f1_macro"],
        "selection": t.get("selection"),
        "best_epoch": m.get("best_epoch"),
        "rebound": (vl[-1] - min(vl)) if vl else float("nan"),
    }


def main() -> None:
    total = 0
    bad_selection = []
    rows = []

    for ds in DATASETS:
        for bb in BACKBONES:
            cell = {}
            for arm in ARMS:
                vals = {}
                for s in SEEDS:
                    r = read(GRID / ds / bb / f"{arm}_seed{s}")
                    if r:
                        vals[s] = r
                        total += 1
                        if r["selection"] != "best_val_f1_checkpoint":
                            bad_selection.append(f"{ds}/{bb}/{arm}_seed{s}: {r['selection']}")
                cell[arm] = vals
            if any(cell[a] for a in ARMS):
                rows.append((ds, bb, cell))

    print(f"completed seed-grid runs: {total} / 72\n")
    if bad_selection:
        print("!! runs NOT using the best-val checkpoint (do not average these in):")
        for b in bad_selection:
            print("   ", b)
        print()

    cur_ds = None
    for ds, bb, cell in rows:
        if ds != cur_ds:
            print("=" * 84)
            print(f"{ds}   (both arms at LR {LR[ds]}, matched by construction)")
            print("=" * 84)
            cur_ds = ds

        parts = []
        stat = {}
        for arm in ARMS:
            v = np.array([cell[arm][s]["f1"] for s in sorted(cell[arm])])
            if len(v) == 0:
                parts.append(f"{ARM_LABEL[arm]:8} --")
                continue
            stat[arm] = v
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            parts.append(f"{ARM_LABEL[arm]:8} {v.mean():.4f}±{sd:.4f} (n={len(v)})")
        print(f"  {bb:10} " + "   ".join(parts), end="")

        if len(stat) == 2 and all(len(v) >= 2 for v in stat.values()):
            g, b = stat["gan"], stat["baseline"]
            t, p = stats.ttest_ind(g, b)
            sp = np.sqrt(((len(g) - 1) * g.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                         / (len(g) + len(b) - 2))
            d = (g.mean() - b.mean()) / sp if sp > 0 else float("nan")
            sig = "*" if p < 0.05 else " "
            print(f"   delta={g.mean() - b.mean():+.4f}{sig} p={p:.3f} d={d:+.2f}")
        else:
            print()

    # per-dataset roll-up across backbones
    print()
    print("=" * 84)
    print("per-dataset summary (all backbones pooled)")
    print("=" * 84)
    for ds in DATASETS:
        deltas = []
        for _, bb, cell in [r for r in rows if r[0] == ds]:
            if cell["baseline"] and cell["gan"]:
                g = np.mean([cell["gan"][s]["f1"] for s in cell["gan"]])
                b = np.mean([cell["baseline"][s]["f1"] for s in cell["baseline"]])
                deltas.append(g - b)
        if deltas:
            arr = np.array(deltas)
            print(f"  {ds:9} matched-LR delta over {len(arr)} backbone(s): "
                  f"mean={arr.mean():+.4f}  positive={int((arr > 0).sum())}/{len(arr)}")


if __name__ == "__main__":
    main()
