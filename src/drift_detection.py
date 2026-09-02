"""Evidently-based drift detection for one incoming batch.

Lifecycle of a batch in the simulated production stream
--------------------------------------------------------
*week k*       batch k *arrives*: it is scored with the currently deployed
               model (loaded from the MLflow Model Registry ``production``
               alias) and compared against the **reference window** - the
               initial training window of the first model (reference.csv),
               which is deliberately kept fixed ("golden baseline") so that
               transient drift episodes cannot silently rewrite the monitoring
               reference:
                 - Evidently data-drift report: PSI test for numeric features,
                   chi-squared test for the categorical feature, and the
                   prediction column => prediction-distribution drift,
                 - explicit KS-tests (numeric) via scipy as a supplement,
               The batch scores are stored - this is the model that actually
               served the batch.
*week k+LAG*   the true labels of batch k become available -> realised model
               performance (AUC, PR-AUC, accuracy, log loss) is computed from
               the *stored* scores, so batches are always evaluated on the
               model that served them.

Retraining policy (needs_retraining) = any PSI > 0.2 on a monitored numeric
feature OR chi-squared p < 0.01 with a material category-share change OR
prediction PSI > 0.2 OR smoothed realised AUC drop >= PERFORMANCE_DROP_MARGIN
vs the serving model's registered baseline.  Every batch's scores are logged
regardless, so the timeline shows the full history, not only flagged points.

The Evidently HTML report for each batch is saved under ``reports/html/`` and
is what the Streamlit app embeds per batch.

CLI (one batch, e.g. after training v1):
    python -m src.drift_detection --batch 12 --html
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .data_prep import feature_frame

warnings.filterwarnings("ignore")

# ---------------- Evidently (legacy facade shipped inside evidently>=0.7) ---
from evidently.legacy.calculations.stattests import chi_stat_test, psi_stat_test
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

EVIDENTLY_NUM_THRESHOLD = config.PSI_THRESHOLD        # PSI > 0.2
EVIDENTLY_CAT_THRESHOLD = config.CHI2_PVALUE_THRESHOLD
SCORES_DIR = config.STREAM_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Model scoring + score cache
# --------------------------------------------------------------------------
def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    X = feature_frame(df, config.MODEL_FEATURES)
    return np.asarray(model.predict_proba(X))[:, 1]


def save_scores(batch_index: int, version: int, scores: np.ndarray) -> Path:
    path = SCORES_DIR / f"batch_{batch_index:04d}_scores.npz"
    np.savez(path, scores=scores, version=version)
    return path


def load_scores(batch_index: int) -> tuple[np.ndarray | None, int | None]:
    path = SCORES_DIR / f"batch_{batch_index:04d}_scores.npz"
    if not path.exists():
        return None, None
    with np.load(path) as z:
        return z["scores"], int(z["version"])


# --------------------------------------------------------------------------
# Fixed reference window (the "golden baseline": v1 training data)
# --------------------------------------------------------------------------
def load_reference(sample: int = config.EV_REFERENCE_SAMPLE) -> pd.DataFrame:
    """Original reference window (features only), capped at ``sample`` rows."""
    df = pd.read_csv(config.STREAM_DIR / "reference.csv")
    feats = feature_frame(df)
    if len(feats) > sample:
        feats = feats.sample(n=sample, random_state=config.SEED)
    return feats.reset_index(drop=True)


# --------------------------------------------------------------------------
# Evidently report
# --------------------------------------------------------------------------
def column_mapping() -> ColumnMapping:
    return ColumnMapping(
        numerical_features=config.MONITOR_NUMERIC_FEATURES,
        categorical_features=config.MONITOR_CATEGORICAL_FEATURES,
        prediction=config.PREDICTION_COLUMN,
    )


def _build_preset() -> DataDriftPreset:
    return DataDriftPreset(
        num_stattest=psi_stat_test,
        num_stattest_threshold=EVIDENTLY_NUM_THRESHOLD,
        cat_stattest=chi_stat_test,
        cat_stattest_threshold=EVIDENTLY_CAT_THRESHOLD,
        drift_share=0.5,
    )


def _run_evidently_report(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    report = Report(metrics=[_build_preset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping())
    obj = json.loads(report.json())
    drift_table = None
    for m in obj.get("metrics", []):
        if m.get("metric") == "DataDriftTable":
            drift_table = m["result"]
    if drift_table is None:  # pragma: no cover - defensive
        drift_table = obj["metrics"][-1]["result"]
    parsed = {}
    for col, info in drift_table.get("drift_by_columns", {}).items():
        parsed[col] = {
            "column_type": info.get("column_type"),
            "stattest_name": info.get("stattest_name"),
            "threshold": info.get("stattest_threshold"),
            "drift_score": info.get("drift_score"),
            "drift_detected": bool(info.get("drift_detected")),
        }
    return {
        "dataset_drift": bool(drift_table.get("dataset_drift")),
        "n_drifted": int(drift_table.get("number_of_drifted_columns", 0)),
        "n_columns": int(drift_table.get("number_of_columns", 0)),
        "drift_share": drift_table.get("drift_share"),
        "columns": parsed,
    }


# --------------------------------------------------------------------------
# KS supplements + categorical effect size
# --------------------------------------------------------------------------
def ks_supplements(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    out = {}
    for col in config.MONITOR_NUMERIC_FEATURES:
        ks = stats.ks_2samp(ref[col].to_numpy(), cur[col].to_numpy())
        out[col] = {"ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}
    return out


def cat_effect_size(ref_feats: pd.DataFrame, cur_feats: pd.DataFrame, col: str) -> float:
    """Max absolute share difference between reference and current categories
    (effect-size gate on top of the chi-squared p-value)."""
    cats = sorted(set(ref_feats[col]) | set(cur_feats[col]))
    if not cats:
        return 0.0
    pr = ref_feats[col].value_counts(normalize=True)
    pc = cur_feats[col].value_counts(normalize=True)
    return float(max(abs(pr.get(c, 0.0) - pc.get(c, 0.0)) for c in cats))


# --------------------------------------------------------------------------
# Realised performance (with delayed true labels)
# --------------------------------------------------------------------------
def realised_performance(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    if n_pos < 1 or n_neg < 1 or not np.all(np.isfinite(scores)):
        return {"n": int(len(y_true)), "n_pos": n_pos, "n_neg": n_neg,
                "auc": None, "pr_auc": None, "accuracy_0.5": None,
                "log_loss": None, "brier": None}
    return {
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "accuracy_0.5": float(accuracy_score(y_true, (scores >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def evaluate_batch_performance(batch_index: int) -> dict:
    """Compute realised performance of a batch once its labels are available,
    using the scores stored when the batch arrived (i.e. the model that
    actually served it)."""
    scores, version = load_scores(batch_index)
    if scores is None:
        raise FileNotFoundError(f"no stored scores for batch {batch_index}")
    batch = pd.read_csv(config.batch_csv(batch_index))
    y = batch[config.TARGET].to_numpy()
    perf = realised_performance(y, scores)
    perf["batch"] = int(batch_index)
    perf["served_by_version"] = version
    return perf


# --------------------------------------------------------------------------
# Batch arrival check
# --------------------------------------------------------------------------
def check_batch(batch_index: int, *, model, deployed: dict,
                reference: pd.DataFrame | None = None, save_html: bool = True) -> dict:
    """Score one arriving batch with the deployed model and check drift vs the
    fixed reference window.

    Returns a dict with drift results; the batch scores are persisted so the
    batch can be evaluated against its labels LABEL_LAG_BATCHES weeks later.
    """
    batch = pd.read_csv(config.batch_csv(batch_index))
    X = feature_frame(batch)

    scores = predict_proba(model, X)
    save_scores(batch_index, int(deployed["version"]), scores)

    if reference is None:
        reference = load_reference()
    ref_ev = reference.copy()
    ref_ev[config.PREDICTION_COLUMN] = predict_proba(model, reference)
    cur_ev = X.copy()
    cur_ev[config.PREDICTION_COLUMN] = scores

    ev = _run_evidently_report(ref_ev, cur_ev)
    columns = ev["columns"]
    pred_col = columns.get(config.PREDICTION_COLUMN, {})
    pred_psi = float(pred_col.get("drift_score", np.nan)) if pred_col else np.nan
    pred_drifted = bool(pred_col.get("drift_detected", False)) if pred_col else False

    # effect-size gate for categorical drift flags (see config.CAT_EFFECT_SIZE)
    for cname, info in columns.items():
        if info.get("column_type") == "cat":
            info["cat_effect"] = cat_effect_size(reference, X, cname)

    ks = ks_supplements(reference, X)

    html_path = None
    if save_html:
        html_path = config.HTML_REPORT_DIR / f"batch_{batch_index:04d}_report.html"
        _save_evidently_html(ref_ev, cur_ev, html_path)

    return {
        "batch": int(batch_index),
        "episode": config.episode_name_for_batch(batch_index),
        "version": int(deployed["version"]),
        "served_by_version": int(deployed["version"]),
        "deployed_test_auc": float(deployed.get("test_auc") or 0.0),
        "n_rows": int(len(batch)),
        "dataset_drift": ev["dataset_drift"],
        "n_drifted": ev["n_drifted"],
        "drift_share": ev["drift_share"],
        "max_psi": float(max(
            (v["drift_score"] for v in columns.values()
             if v.get("column_type") == "num" and v["drift_score"] is not None),
            default=0.0)),
        "prediction_psi": pred_psi,
        "prediction_drifted": pred_drifted,
        "flagged_features": [c for c, v in columns.items() if v["drift_detected"]],
        "ks": ks,
        "per_column": columns,
        "html_report": str(html_path) if html_path else None,
    }


def _save_evidently_html(ref_ev: pd.DataFrame, cur_ev: pd.DataFrame, path: Path) -> None:
    report = Report(metrics=[_build_preset()])
    report.run(reference_data=ref_ev, current_data=cur_ev, column_mapping=column_mapping())
    path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(path))


def cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--html", action="store_true", help="save Evidently HTML report")
    args = ap.parse_args(argv)
    from . import registry

    registry.setup_tracking()
    deployed = registry.production_version()
    if deployed is None:
        print("no production model registered - train first", file=sys.stderr)
        return 1
    ledger = registry.load_deployments()
    entry = ledger["versions"].get(str(deployed["version"]))
    if entry is None:
        entry = {"version": deployed["version"], "test_auc": None}
    model = registry.load_model(deployed["version"])
    record = check_batch(args.batch, model=model, deployed=entry,
                         reference=load_reference(), save_html=args.html)
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
