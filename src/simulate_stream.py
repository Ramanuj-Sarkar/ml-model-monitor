"""Simulate a production data stream from static historical data.

How the simulation works (details also in README.md, section "Simulating the
production stream"):

1. The 150k rows of ``data/structured-ml-dataset.csv`` are shuffled once with a
   fixed seed. The first ``REFERENCE_SIZE`` rows become the *reference window*
   (``data/stream/reference.csv``) - the history the first model is trained on
   and the initial comparison window for drift detection.

2. The remaining rows are replayed in order as ``BATCH_SIZE``-row weekly
   batches (``data/stream/batch_0001.csv`` ... ``batch_0044.csv``), each one
   representing one week of "production traffic".

3. Deliberate drift is injected into selected later batches (config.EPISODES):

   * ``covariate_util_shift`` (batches 11-14):  *data drift* - the mean and
     variance of ``RevolvingUtilizationOfUnsecuredLines`` are shifted upward:
     ``util' = util * 1.6 + 0.15``.  The feature->label relationship is
     unchanged, only the input distribution moves.

   * ``category_dependents`` (batches 21-24):  *data drift* - the frequency of
     ``NumberOfDependents`` is re-weighted (many more households with 1-3
     dependents).  Detected by the chi-squared test.

   * ``concept_util_flip`` (batch 31 onward):  *concept drift* - the
     relationship between a strong feature and the label is flipped: rows with
     ``RevolvingUtilizationOfUnsecuredLines >= 0.8`` stop defaulting (their
     true label is set to 0 with probability 0.8).  Input distributions are
     untouched - this drift is invisible to feature-drift tests and must be
     caught by monitoring realised model performance.

All batches contain the (possibly drifted) ground-truth label column; the
monitor pretends labels arrive ``LABEL_LAG_BATCHES`` weeks late, which mirrors
a real deployment where outcomes are only observed after a delay.

Run:
    python -m src.simulate_stream
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config
from .data_prep import clean_features, read_raw


def build_stream(
    raw_path=config.RAW_DATA_CSV,
    out_dir=config.STREAM_DIR,
    reference_size=config.REFERENCE_SIZE,
    batch_size=config.BATCH_SIZE,
    seed=config.SEED,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_raw(raw_path)
    rng = np.random.default_rng(seed)

    # ---- 1. shuffle once: this defines "time" ---------------------------
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    reference = df.iloc[:reference_size].copy()
    rest = df.iloc[reference_size:].copy()

    # medians imputed from the reference window only (as at deploy time)
    feat_ref = reference.drop(columns=[config.TARGET])
    feat_ref_clipped, medians = clean_features(feat_ref)
    reference.loc[:, feat_ref.columns] = feat_ref_clipped

    reference.to_csv(out_dir / "reference.csv", index=False)

    n_batches = len(rest) // batch_size
    batches = rest.iloc[: n_batches * batch_size].copy().reset_index(drop=True)

    manifest_batches = []
    for b in range(1, n_batches + 1):
        lo = (b - 1) * batch_size
        hi = b * batch_size
        part = batches.iloc[lo:hi].copy()
        episode = config.episode_name_for_batch(b)
        applied = _inject_drift(part, episode, b, rng)
        part_clean, _ = clean_features(part, medians=medians)
        path = out_dir / f"batch_{b:04d}.csv"
        part_clean.to_csv(path, index=False)
        manifest_batches.append(
            {
                "batch": b,
                "file": path.name,
                "episode": episode,
                "injected": applied,
                "n_rows": int(len(part_clean)),
                "positive_rate": float(part_clean[config.TARGET].mean()),
            }
        )

    manifest = {
        "raw_file": str(raw_path),
        "n_rows_total": int(len(df)),
        "reference_size": int(reference_size),
        "reference_file": "reference.csv",
        "batch_size": int(batch_size),
        "n_batches": int(n_batches),
        "seed": int(seed),
        "label_lag_batches": config.LABEL_LAG_BATCHES,
        "caps": config.CAPS,
        "medians_imputed": medians,
        "episodes": {
            name: {**ep, "doc": ep["doc"]} for name, ep in config.EPISODES.items()
        },
        "batches": manifest_batches,
    }
    with open(config.MANIFEST_FILE, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # quick self-check summary printed to stdout / stderr
    _summarize(manifest)
    return manifest


def _inject_drift(part: pd.DataFrame, episode: str, batch_idx: int, rng) -> dict:
    """Mutate a batch according to its episode. Returns a description."""
    ep = config.EPISODES.get(episode)
    if ep is None:
        return {}
    if ep["kind"] == "numeric_shift":
        feat = ep["feature"]
        mul, add = ep["mul"], ep["add"]
        part[feat] = part[feat] * mul + add
        return {feat: f"x*{mul}+{add} (mean+variance shift)"}
    if ep["kind"] == "categorical_reweight":
        feat = ep["feature"]
        pmf = ep["pmf"]
        keys = list(pmf.keys())
        weights = np.array([pmf[k] for k in keys], dtype=float)
        weights = weights / weights.sum()
        new_vals = rng.choice(keys, size=len(part), p=weights).astype(float)
        part[feat] = new_vals
        return {feat: f"re-sampled from altered pmf {pmf}"}
    if ep["kind"] == "label_relationship_flip":
        seg_feat = ep["segment_feature"]
        seg_val = ep["segment_min"]
        mask = part[seg_feat] >= seg_val
        n_in = int(mask.sum())
        flip = rng.random(len(part)) < ep["flip_frac"]
        apply = mask & flip
        part.loc[apply, config.TARGET] = 0
        return {
            f"{seg_feat}>={seg_val}": (
                f"{n_in} rows ({n_in / len(part):.1%}), labels set to 0 for "
                f"{int(apply.sum())} ({ep['flip_frac']:.0%}) of them"
            )
        }
    raise ValueError(f"unknown drift kind: {ep['kind']}")


def _summarize(manifest: dict) -> None:
    print(f"Reference window : {manifest['reference_file']}  "
          f"({manifest['reference_size']} rows)")
    print(f"Batches          : {manifest['n_batches']} x {manifest['batch_size']} rows")
    for b in manifest["batches"]:
        if b["episode"] != "clean":
            print(f"  batch {b['batch']:>3} [{b['episode']:^22}] "
                  f"n={b['n_rows']} pos={b['positive_rate']:.4f} injected={b['injected']}")


if __name__ == "__main__":
    build_stream()
    sys.exit(0)
