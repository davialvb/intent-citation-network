"""Builds an expanded "unsupervised_cross.csv" for each of SciCite/ACL-ARC/3C by
pulling extra (label-stripped) citation-sentence text from the OTHER two
datasets' training splits, on top of each dataset's existing small in-domain
unsupervised sample (10% of val).

Motivation: GAN-BERT's unlabeled stream only trains the discriminator's
real/fake head (not pseudo-labels for the actual classes), so text from other
citation-intent datasets is a plausible source of extra "real" citation
sentences when the in-domain unlabeled pool is tiny (11-92 rows here).

Run:
    uv run python src/utils/build_cross_dataset_unlabeled.py --seed 42 --n_per_source 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

DATASETS = ["scicite", "acl-arc", "3C"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_per_source", type=int, default=300, help="Rows sampled from each *other* dataset.")
    args = p.parse_args()

    train_texts = {}
    for ds in DATASETS:
        df = pd.read_csv(DATA_ROOT / ds / "gan_bert" / "labeled_train.csv")
        train_texts[ds] = df["text"]

    for ds in DATASETS:
        own_unsup = pd.read_csv(DATA_ROOT / ds / "gan_bert" / "unsupervised.csv")
        cross_parts = [own_unsup[["text"]]]
        for other in DATASETS:
            if other == ds:
                continue
            n = min(args.n_per_source, len(train_texts[other]))
            sample = train_texts[other].sample(n=n, random_state=args.seed)
            cross_parts.append(pd.DataFrame({"text": sample.values}))

        combined = pd.concat(cross_parts, ignore_index=True)
        out_path = DATA_ROOT / ds / "gan_bert" / "unsupervised_cross.csv"
        combined.to_csv(out_path, index=False)
        print(f"[{ds}] unsupervised_cross.csv: {len(own_unsup)} in-domain + "
              f"{len(combined) - len(own_unsup)} cross-dataset = {len(combined)} total")


if __name__ == "__main__":
    main()
