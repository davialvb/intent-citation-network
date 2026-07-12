"""Unified pre-processing for the SciCite, ACL-ARC and 3C citation-intent datasets.

Produces, for each dataset, a standardized set of CSVs under
``data/<dataset>/gan_bert/``:

- ``labeled_train.csv``   -- supervised training data (columns: text, intent)
- ``val.csv``             -- validation data used for per-epoch evaluation (text, intent)
- ``unsupervised.csv``    -- a random sample drawn from the *val* split, with the
                             label column stripped (column: text only). This is
                             the data used as the "unlabeled" stream that trains
                             the GAN discriminator's real/fake head.
- ``test.csv``            -- held-out test data used for final reporting (text, intent)
- ``metadata.json``       -- split sizes, label set, and the random seed used.

The `unsupervised` sample is *removed* from `val.csv` to avoid evaluating on
examples that were also used (unlabeled) during training.

Run as a script to build all three datasets:

    uv run python src/utils/gan_bert_preprocessing.py --seed 42 --unsup_frac 0.1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

# Canonical section vocabulary shared across all three datasets, following the
# standard IMRaD-style discourse categories used in citation-context /
# section-scaffold literature (e.g. Cohan et al. 2019's SciCite section-title
# scaffold task). "unknown" covers missing values and datasets that don't
# provide section metadata at all (3C).
SECTION_VOCAB = [
    "introduction",
    "background",
    "related_work",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "unknown",
]

_NUMBERING_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s*")


def normalize_section_free_text(raw: object) -> str:
    """Normalize a free-text section heading (e.g. SciCite's `sectionName`,
    which has 1000+ raw variants like "4. Discussion", "RESULTS AND DISCUSSION",
    "1. INTRODUCTION") into one of SECTION_VOCAB via numbering-stripping,
    lowercasing, and keyword matching. Order matters: more specific / composite
    headings are matched before their component keywords.
    """
    if raw is None or (isinstance(raw, float)) or not str(raw).strip():
        return "unknown"
    s = _NUMBERING_RE.sub("", str(raw)).strip().lower()
    if not s:
        return "unknown"

    if "related work" in s or "literature review" in s or "prior work" in s:
        return "related_work"
    if "introduction" in s:
        return "introduction"
    if "conclusion" in s or "concluding" in s:
        return "conclusion"
    if "discussion" in s:
        return "discussion"
    if "result" in s or "experiment" in s or "evaluation" in s:
        return "results"
    if "method" in s or "approach" in s or "model" in s:
        return "methods"
    if "background" in s:
        return "background"
    return "unknown"


def normalize_section_clean(raw: object) -> str:
    """Normalize an already-clean, low-cardinality section field (ACL-ARC's
    `section_name`, or a thesis "chapter" column) into SECTION_VOCAB.
    """
    if raw is None or (isinstance(raw, float)) or not str(raw).strip():
        return "unknown"
    s = str(raw).strip().lower()
    mapping = {
        "related work": "related_work",
        "related_work": "related_work",
        "experiments": "results",
        "experiment": "results",
        "method": "methods",
        "methods": "methods",
        "methodology": "methods",
    }
    if s in mapping:
        return mapping[s]
    if s in SECTION_VOCAB:
        return s
    return "unknown"


# Canonical (lowercased) label sets per dataset, "unknown" always last (used for
# the unsupervised/unlabeled rows by the GAN-BERT training loop).
SCICITE_LABELS = ["background", "method", "result", "unknown"]
# Note: raw ACL-ARC intents are normalized via lower() + strip non-letters, so
# "CompareOrContrast" -> "compareorcontrast" (must match exactly, not "comparecontrast").
ACL_ARC_LABELS = ["background", "uses", "compareorcontrast", "motivation", "extends", "future", "unknown"]
THREEC_LABEL_MAP = {
    0: "background",
    1: "comparecontrast",
    2: "extension",
    3: "future",
    4: "motivation",
    5: "uses",
}
THREEC_LABELS = ["background", "comparecontrast", "extension", "future", "motivation", "uses", "unknown"]


def _split_unsupervised(val_df: pd.DataFrame, unsup_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly sample `unsup_frac` of val_df to become the stripped unsupervised set.

    Returns (remaining_val_df, unsupervised_df[text, section]).
    """
    unsup = val_df.sample(frac=unsup_frac, random_state=seed)
    remaining_val = val_df.drop(index=unsup.index).reset_index(drop=True)
    unsupervised = unsup[["text", "section"]].reset_index(drop=True)
    return remaining_val, unsupervised


def _write_outputs(
    out_dir: Path,
    labeled: pd.DataFrame,
    val: pd.DataFrame,
    unsupervised: pd.DataFrame,
    test: pd.DataFrame,
    labels: list[str],
    seed: int,
    unsup_frac: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled[["text", "intent", "section"]].to_csv(out_dir / "labeled_train.csv", index=False)
    val[["text", "intent", "section"]].to_csv(out_dir / "val.csv", index=False)
    unsupervised[["text", "section"]].to_csv(out_dir / "unsupervised.csv", index=False)
    test[["text", "intent", "section"]].to_csv(out_dir / "test.csv", index=False)

    metadata = {
        "labels": labels,
        "section_vocab": SECTION_VOCAB,
        "seed": seed,
        "unsup_frac": unsup_frac,
        "n_labeled_train": len(labeled),
        "n_val": len(val),
        "n_unsupervised": len(unsupervised),
        "n_test": len(test),
        "label_counts_train": labeled["intent"].value_counts().to_dict(),
        "label_counts_val": val["intent"].value_counts().to_dict(),
        "label_counts_test": test["intent"].value_counts().to_dict(),
        "section_counts_train": labeled["section"].value_counts().to_dict(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"[{out_dir}] labeled_train={len(labeled)} val={len(val)} "
          f"unsupervised={len(unsupervised)} test={len(test)}")


def build_scicite(seed: int, unsup_frac: float) -> None:
    raw = DATA_ROOT / "scicite" / "raw"
    train = pd.read_json(raw / "train.jsonl", lines=True)
    dev = pd.read_json(raw / "dev.jsonl", lines=True)
    test = pd.read_json(raw / "test.jsonl", lines=True)

    for df in (train, dev, test):
        df.rename(columns={"string": "text", "label": "intent"}, inplace=True)
        df["section"] = df["sectionName"].map(normalize_section_free_text)

    val, unsupervised = _split_unsupervised(dev[["text", "intent", "section"]], unsup_frac, seed)

    _write_outputs(
        DATA_ROOT / "scicite" / "gan_bert",
        train[["text", "intent", "section"]],
        val,
        unsupervised,
        test[["text", "intent", "section"]],
        SCICITE_LABELS,
        seed,
        unsup_frac,
    )


def build_acl_arc(seed: int, unsup_frac: float) -> None:
    raw = DATA_ROOT / "acl-arc" / "raw"
    train = pd.read_json(raw / "train.jsonl", lines=True)
    dev = pd.read_json(raw / "dev.jsonl", lines=True)
    test = pd.read_json(raw / "test.jsonl", lines=True)

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        section = df["section_name"].map(normalize_section_clean)
        df = df[["text", "intent"]].copy()
        df["intent"] = df["intent"].str.lower().str.replace(r"[^a-z]", "", regex=True)
        df["section"] = section
        return df

    train, dev, test = _norm(train), _norm(dev), _norm(test)

    val, unsupervised = _split_unsupervised(dev, unsup_frac, seed)

    _write_outputs(
        DATA_ROOT / "acl-arc" / "gan_bert",
        train,
        val,
        unsupervised,
        test,
        ACL_ARC_LABELS,
        seed,
        unsup_frac,
    )


def build_3c(seed: int, unsup_frac: float) -> None:
    raw = DATA_ROOT / "3C" / "raw"
    df = pd.read_csv(raw / "SDP_train.csv")
    df = df[["citation_context", "citation_class_label"]].copy()
    df.columns = ["text", "intent"]
    df["intent"] = df["intent"].map(THREEC_LABEL_MAP)
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    # 3C provides no section metadata at all.
    df["section"] = "unknown"

    # Held-out pool for val+test (25%), rest is supervised train.
    from sklearn.model_selection import train_test_split

    train, rest = train_test_split(df, test_size=0.25, random_state=seed, stratify=df["intent"])
    val_pool, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["intent"])

    val, unsupervised = _split_unsupervised(val_pool.reset_index(drop=True), unsup_frac, seed)

    _write_outputs(
        DATA_ROOT / "3C" / "gan_bert",
        train.reset_index(drop=True),
        val,
        unsupervised,
        test.reset_index(drop=True),
        THREEC_LABELS,
        seed,
        unsup_frac,
    )


def main():
    p = argparse.ArgumentParser(description="Build standardized GAN-BERT splits for all datasets.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unsup_frac", type=float, default=0.1, help="Fraction of val sampled (labels stripped) for GAN unsupervised data.")
    p.add_argument("--datasets", nargs="+", default=["scicite", "acl-arc", "3c"], choices=["scicite", "acl-arc", "3c"])
    args = p.parse_args()

    if "scicite" in args.datasets:
        build_scicite(args.seed, args.unsup_frac)
    if "acl-arc" in args.datasets:
        build_acl_arc(args.seed, args.unsup_frac)
    if "3c" in args.datasets:
        build_3c(args.seed, args.unsup_frac)


if __name__ == "__main__":
    main()
