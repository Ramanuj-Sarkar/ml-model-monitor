"""Data loading and cleaning helpers.

``clean_features`` is the ONE place where raw data becomes model/monitoring
input: it winsorizes extreme outliers with the global caps from ``config`` and
imputes missing values with medians taken from the reference window (the
initial training data), so that nothing downstream has to know about the raw
file's quirks (empty strings / 'NA' markers, the RevolvingUtilization==50708
artifact, ...).

``PreprocessTransformer`` is kept for backward compatibility with the archived
research script ``archive/model_comparison_original.py``; the production
pipeline does not use it because stream files are already cleaned
deterministically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

# Alias kept so archived code (`from data_prep import TARGET, load_data`) parses.
TARGET = config.TARGET


def read_raw(path=config.RAW_DATA_CSV) -> pd.DataFrame:
    """Read the raw CSV: drop the unnamed row-index column, clean names."""
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if unnamed:
        df = df.drop(columns=unnamed)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace({"NA": np.nan, "": np.nan, "nan": np.nan})
    for c in df.columns:
        if c != TARGET:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_features(df: pd.DataFrame, caps: dict | None = None, medians: dict | None = None,
                   topcodes: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clip outliers to global caps, top-code a couple of count features and
    impute missing values.

    Parameters
    ----------
    df       : features-only frame (target may be present, it is ignored).
    caps     : {feature: (lo, hi)}; defaults to config.CAPS.
    medians  : {feature: median}; when None, computed from ``df`` (only
               meaningful when df is the reference window).
    topcodes : {feature: upper-bound} applied *before* imputation so that
               imputed categories stay within the monitored support.
    """
    caps = caps or config.CAPS
    topcodes = topcodes if topcodes is not None else config.TOPCODES
    out = df.copy()
    for col, (lo, hi) in caps.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)
    for col, hi in topcodes.items():
        if col in out.columns:
            out[col] = out[col].clip(upper=hi)
    if medians is None:
        medians = {c: out[c].median() for c in out.columns if c != TARGET and c in out.columns}
    for col in out.columns:
        if col == TARGET:
            continue
        if col in medians:
            out[col] = out[col].fillna(medians[col])
    return out, medians


def load_clean(path=config.RAW_DATA_CSV, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Read + clean a raw dataset. Returns (features+target frame, medians)."""
    df = read_raw(path)
    return clean_features(df, medians=medians)


def reference_medians(path=config.RAW_DATA_CSV, reference_size: int = config.REFERENCE_SIZE,
                      seed: int = config.SEED) -> dict:
    """Medians of the reference window rows (used to impute the whole stream
    deterministically as if imputation happened once at deploy time)."""
    df = read_raw(path)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    w0 = df.iloc[perm[:reference_size]]
    feat = w0.drop(columns=[TARGET])
    _, med = clean_features(feat)
    return med


def feature_frame(df: pd.DataFrame, features=None) -> pd.DataFrame:
    features = features or config.MODEL_FEATURES
    return df[features].copy()


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-style transformer for the archived research script.

    Fits per-feature median imputation + [0.5%, 99.5%] winsorization caps on
    the training split and applies them to any later frame.
    """

    def __init__(self, low_q=0.005, high_q=0.995):
        self.low_q = low_q
        self.high_q = high_q
        self.medians_ = None
        self.lows_ = None
        self.highs_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            cols = [f"f{i}" for i in range(arr.shape[1])]
        lows, highs, meds = [], [], []
        for j in range(arr.shape[1]):
            col = arr[:, j]
            col = col[~np.isnan(col)]
            lows.append(np.quantile(col, self.low_q))
            highs.append(np.quantile(col, self.high_q))
            meds.append(np.nanmedian(arr[:, j]))
        self.lows_ = np.asarray(lows)
        self.highs_ = np.asarray(highs)
        self.medians_ = np.asarray(meds)
        self.feature_names_ = cols
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
        out = np.clip(arr, self.lows_, self.highs_)
        for j in range(out.shape[1]):
            mask = np.isnan(out[:, j])
            if mask.any():
                out[mask, j] = self.medians_[j]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)
