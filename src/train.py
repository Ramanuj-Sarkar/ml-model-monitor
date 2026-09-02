"""Production trainer with MLflow tracking + Model Registry.

Trains a LightGBM or XGBoost binary classifier on a (possibly assembled)
training-window CSV and registers every run in MLflow:

* Tracking  : hyper-parameters, metrics (AUC, PR-AUC, KS, Gini, Brier, log
              loss, accuracy@0.5 / @Youden, ...), tags and the model artifact.
* Registry  : each run is registered as a new version of
              ``config.MODEL_NAME``; the version is version-bumped and becomes
              the ``production`` alias + ``stage=production`` tag.

Usage
-----
    python -m src.train --data-csv data/stream/reference.csv \
        --model-type lgbm --description "initial model on reference window" \
        --tag trained_on=reference

    python -m src.train --data-csv data/stream/train_window_v4.csv \
        --model-type lgbm --description "retrain after concept drift (batch 30)" \
        --tag trained_on="reference+batches_1..30" --tag trigger=auc_drop

The model simply predicts default probability on the 10 cleaned numeric
features; no scaling is needed for GBDTs.  All window CSVs produced by the
stream simulator are already imputed/winsorized deterministically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from . import config, registry
from .data_prep import feature_frame

SEED = config.SEED

# Fixed, documented hyper-parameters (tuned-ish defaults for tabular
# classification; logging them is exactly the point of MLflow tracking).
LGBM_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2500,
    random_state=SEED,
    n_jobs=8,
    verbose=-1,
)
XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=2500,
    tree_method="hist",
    n_jobs=8,
    random_state=SEED,
    eval_metric="auc",
)
EARLY_STOPPING_ROUNDS = 100


def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def all_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    youden = float(thr[np.argmax(tpr - fpr)])
    pred50 = (y_score >= 0.5).astype(int)
    pred_youden = (y_score >= youden).astype(int)
    return {
        "auc": float(auc),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score)),
        "accuracy_0.5": float(accuracy_score(y_true, pred50)),
        "youden_threshold": youden,
        "accuracy_youden": float(accuracy_score(y_true, pred_youden)),
        "n_estimators": 0,  # filled after fitting
    }


def train_on_window(data_csvs, model_type="lgbm", holdout_frac=0.15, seed=SEED,
                    register=True, description="", tags=None):
    """Train a model on the union of ``data_csvs`` and log/register it."""
    frames = [pd.read_csv(p) for p in data_csvs]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)
    X = feature_frame(df)
    y = df[config.TARGET].astype(int).to_numpy()

    if len(X) < 500:
        raise ValueError(f"training window too small ({len(X)} rows)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=holdout_frac, stratify=y, random_state=seed
    )

    # Probe fit with early stopping on an inner 10% split -> best tree count,
    # then refit on the whole training window with that tree count.
    idx_fit, idx_es = train_test_split(
        np.arange(len(X_tr)), test_size=0.1, stratify=y_tr, random_state=seed
    )
    hp = dict(LGBM_PARAMS if model_type == "lgbm" else XGB_PARAMS)
    es_hp = {k: v for k, v in hp.items() if k not in ("n_estimators", "eval_metric")}
    es_hp["n_estimators"] = 2500
    es_hp["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    if model_type == "lgbm":
        from lightgbm import LGBMClassifier

        probe = LGBMClassifier(**es_hp)
        probe.fit(X_tr.iloc[idx_fit], y_tr[idx_fit],
                  eval_set=[(X_tr.iloc[idx_es], y_tr[idx_es])])
        best_iter = int(probe.best_iteration_)
        final_hp = {k: v for k, v in hp.items() if k != "n_estimators"}
        model = LGBMClassifier(**final_hp, n_estimators=best_iter)
        model.fit(X_tr, y_tr)
    else:
        from xgboost import XGBClassifier

        probe = XGBClassifier(**es_hp)
        probe.fit(X_tr.iloc[idx_fit], y_tr[idx_fit],
                  eval_set=[(X_tr.iloc[idx_es], y_tr[idx_es])], verbose=False)
        best_iter = int(probe.best_iteration)
        final_hp = {k: v for k, v in hp.items() if k not in ("n_estimators", "eval_metric")}
        model = XGBClassifier(**final_hp, n_estimators=best_iter)
        model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    metrics = all_metrics(y_te, proba)
    metrics["n_estimators"] = best_iter
    train_proba = model.predict_proba(X_tr)[:, 1]
    metrics["train_auc"] = float(roc_auc_score(y_tr, train_proba))

    tags_all = {
        "model_type": model_type,
        "trained_on": (tags or {}).get("trained_on", "custom window"),
        "trigger": (tags or {}).get("trigger", "initial"),
        "run_name": f"{model_type}-{metrics['auc']:.4f}",
        "holdout_frac": str(holdout_frac),
        "window_files": json.dumps([str(p) for p in data_csvs]),
        "description": description,
    }
    params = {"model_type": model_type, "holdout_frac": holdout_frac,
              "seed": seed, "window_rows": int(len(df)), "best_iteration": int(best_iter)}
    params.update({f"hp.{k}": str(v) for k, v in hp.items() if k != "n_estimators"})

    input_example = X_te.head(3)
    signature = infer_signature(X_te.head(20).reset_index(drop=True),
                                model.predict_proba(X_te.head(20))[:, 1])
    info = registry.log_training_run(
        params=params,
        metrics=metrics,
        tags=tags_all,
        model=model,
        input_example=input_example,
        signature=signature,
        window_rows=int(len(df)),
        description=description,
        model_type=model_type,
        register=register,
    )

    if register:
        entry = {
            "version": info["version"],
            "run_id": info["run_id"],
            "model_type": model_type,
            "trained_on": tags_all["trained_on"],
            "trigger": tags_all["trigger"],
            "window_files": [str(p) for p in data_csvs],
            "window_rows": int(len(df)),
            "test_auc": float(metrics["auc"]),
            "metrics": metrics,
            "alias": config.PRODUCTION_ALIAS,
            "registered_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        registry.append_deployment(entry)
        print(f"registered model version {info['version']} as '{config.PRODUCTION_ALIAS}' "
              f"(run {info['run_id'][:8]})")
    else:
        print("not registered (dry run)")

    print(f"[{model_type}] trained on {len(df):,} rows, holdout {len(X_te):,}: "
          f"AUC={metrics['auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
          f"KS={metrics['ks']:.4f} acc@0.5={metrics['accuracy_0.5']:.4f} "
          f"trees={best_iter}")
    return {"version": info.get("version"), "run_id": info["run_id"], "metrics": metrics,
            "window_rows": int(len(df))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-csv", nargs="+", required=True,
                    help="training-window CSV file(s) (reference and/or batches)")
    ap.add_argument("--model-type", choices=["lgbm", "xgb"], default="lgbm")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--description", default="")
    ap.add_argument("--tag", action="append", default=[], metavar="K=V")
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args(argv)

    registry.setup_tracking()
    tags = {}
    for kv in args.tag:
        k, _, v = kv.partition("=")
        tags[k] = v
    result = train_on_window(
        [Path(p) for p in args.data_csv],
        model_type=args.model_type,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        register=not args.no_register,
        description=args.description,
        tags=tags,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
