"""Model comparison: XGBoost vs LightGBM vs CatBoost (+ LR baseline).

Methodology
-----------
1. Stratified 80/20 holdout: the 20% test set is never touched during tuning.
2. Hyperparameter tuning per GBDT: random search with **out-of-fold AUC**,
   computed with a manual loop so early stopping on an inner validation split
   is used honestly for every candidate (sklearn's RandomizedSearchCV cannot
   do this per fold).
3. Final 5-fold stratified CV with the tuned hyperparameters on the 80% train
   -> mean +/- std of AUC / PR-AUC / KS / Gini / Brier / log loss.
4. Refit on the full 80% train (with the tuned number of trees) -> metrics on
   the untouched 20% test set, plus ROC / PR / calibration plots.
5. Logistic Regression baseline (balanced class weights) via
   RandomizedSearchCV inside a scikit-learn Pipeline for context.

All preprocessing statistics (winsorization caps, imputation medians) are
re-fit inside every CV fold / test refit by ``PreprocessTransformer``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc as sk_auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import PreprocessTransformer, TARGET, load_data

SEED = 42
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
N_TUNE_FOLDS = 4
N_EVAL_FOLDS = 5
TEST_SIZE = 0.2
TUNE_ITER = {"xgb": 14, "lgbm": 14, "cat": 10}
EARLY_STOPPING_ROUNDS = 100
MAX_ROUNDS = 2000
N_JOBS = 12


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def all_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": float(auc),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score)),
    }


# --------------------------------------------------------------------------
# Model definitions and hyperparameter spaces
# --------------------------------------------------------------------------
def build_xgb(params, n_estimators=None, early_stopping=True):
    from xgboost import XGBClassifier

    kw = dict(
        n_estimators=n_estimators or MAX_ROUNDS,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=SEED,
    )
    if early_stopping:
        kw["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        kw["eval_metric"] = "auc"
    return XGBClassifier(**kw)


def build_lgbm(params, n_estimators=None, early_stopping=True):
    from lightgbm import LGBMClassifier

    kw = dict(
        n_estimators=n_estimators or MAX_ROUNDS,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        subsample_freq=1,
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=-1,
    )
    if early_stopping:
        kw["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return LGBMClassifier(**kw)


def build_cat(params, n_estimators=None, early_stopping=True):
    from catboost import CatBoostClassifier

    kw = dict(
        iterations=n_estimators or MAX_ROUNDS,
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        bagging_temperature=params["bagging_temperature"],
        random_strength=params["random_strength"],
        eval_metric="AUC",
        thread_count=N_JOBS,
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
    )
    if early_stopping:
        kw["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return CatBoostClassifier(**kw)


XGB_SPACE = {
    "learning_rate": loguniform(0.01, 0.15),
    "max_depth": randint(3, 9),
    "min_child_weight": loguniform(1.0, 20.0),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-2, 10.0),
}
LGBM_SPACE = {
    "learning_rate": loguniform(0.01, 0.15),
    "num_leaves": randint(8, 128),
    "min_child_samples": randint(10, 100),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": loguniform(1e-3, 10.0),
    "reg_lambda": loguniform(1e-3, 10.0),
}
CAT_SPACE = {
    "learning_rate": loguniform(0.01, 0.15),
    "depth": randint(4, 9),
    "l2_leaf_reg": loguniform(1.0, 30.0),
    "bagging_temperature": uniform(0.0, 1.0),
    "random_strength": uniform(0.0, 1.0),
}

BUILDERS = {"xgb": build_xgb, "lgbm": build_lgbm, "cat": build_cat}
SPACES = {"xgb": XGB_SPACE, "lgbm": LGBM_SPACE, "cat": CAT_SPACE}
BEST_ITER_ATTR = {"xgb": "best_iteration", "lgbm": "best_iteration_", "cat": "best_iteration_"}
N_EST_PARAM = {"xgb": "n_estimators", "lgbm": "n_estimators", "cat": "iterations"}


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    return {k: dist.rvs(random_state=rng) for k, dist in space.items()}


def fit_with_early_stopping(model, name, X_fit, y_fit, X_val, y_val):
    """Fit with early stopping; the three libraries differ in fit() kwargs."""
    if name == "lgbm":
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc")
    else:
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


# --------------------------------------------------------------------------
# Tuning: manual out-of-fold loop with early stopping
# --------------------------------------------------------------------------
def _oof_auc(name, build, params, X, y, n_folds=N_TUNE_FOLDS):
    """Out-of-fold AUC for one hyperparameter combination.

    Each outer fold: fit prep on fold-train, split fold-train into
    fit/val(80/20), train with early stopping on val, predict the outer
    validation fold. No leakage: prep statistics and early-stopping decisions
    both come from the fold-train only.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        fit_idx, val_idx = train_test_split(
            np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED
        )
        Xfit, Xval = Xtr.iloc[fit_idx], Xtr.iloc[val_idx]
        yfit, yval = ytr.iloc[fit_idx], ytr.iloc[val_idx]

        prep = PreprocessTransformer().fit(Xfit)
        Xfit_p = prep.transform(Xfit)
        Xval_p = prep.transform(Xval)
        Xva_p = prep.transform(Xva)

        model = build(params)
        fit_with_early_stopping(model, name, Xfit_p, yfit, Xval_p, yval)
        oof[va_idx] = model.predict_proba(Xva_p)[:, 1]
    return roc_auc_score(y, oof)


def tune_model(name, X, y, n_iter=None) -> dict:
    rng = np.random.default_rng(SEED)
    n_iter = n_iter or TUNE_ITER[name]
    build = BUILDERS[name]
    space = SPACES[name]
    best = None
    history = []
    for i in range(n_iter):
        params = sample_params(space, rng)
        try:
            auc = _oof_auc(name, build, params, X, y)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [{name}] combo {i + 1} failed: {exc}")
            continue
        history.append((auc, params))
        if best is None or auc > best[0]:
            best = (auc, params)
        print(f"  [{name}] combo {i + 1}/{n_iter}: OOF AUC = {auc:.5f}"
              + ("  <-- best" if best[0] == auc else ""))
    history.sort(key=lambda t: -t[0])
    return {"best_auc": best[0], "best_params": best[1], "history": history[:5]}


# --------------------------------------------------------------------------
# Final evaluation
# --------------------------------------------------------------------------
def run_cv(name, params, X, y, n_folds=N_EVAL_FOLDS):
    """5-fold stratified CV with the tuned hyperparameters.

    Prep is re-fit inside every fold and each fold model is trained with
    early stopping on an inner 20% split of the fold-train (so the tree count
    adapts per fold instead of being fixed from a single split). Predictions
    for the outer fold are never touched by prep fitting or early stopping.
    """
    build = BUILDERS[name]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        fit_idx, val_idx = train_test_split(
            np.arange(len(Xtr)), test_size=0.2, stratify=ytr, random_state=SEED
        )
        Xfit, Xval = Xtr.iloc[fit_idx], Xtr.iloc[val_idx]
        yfit, yval = ytr.iloc[fit_idx], ytr.iloc[val_idx]

        prep = PreprocessTransformer().fit(Xfit)
        Xfit_p = prep.transform(Xfit)
        Xval_p = prep.transform(Xval)
        Xva_p = prep.transform(Xva)

        model = build(params)
        fit_with_early_stopping(model, name, Xfit_p, yfit, Xval_p, yval)
        best_iter = int(getattr(model, BEST_ITER_ATTR[name]))
        proba = model.predict_proba(Xva_p)[:, 1]
        m = all_metrics(yva, proba)
        m.update({"fold": fold, "model": name, "n_estimators": best_iter})
        rows.append(m)
    return pd.DataFrame(rows)


def refit_and_eval_test(name, params, X_train, y_train, X_test, y_test):
    """Refit on the full 80% train, evaluate the untouched test set.

    The tree count is chosen by one early-stopping run on a 90/10 split of
    the training data, then the model is refit on 100% of the training data
    with that tree count (no early stopping), so no training rows are wasted.
    """
    build = BUILDERS[name]
    n_est_key = N_EST_PARAM[name]
    prep = PreprocessTransformer().fit(X_train)
    Xtr_p = prep.transform(X_train)
    Xte_p = prep.transform(X_test)
    fit_idx, val_idx = train_test_split(
        np.arange(len(Xtr_p)), test_size=0.1, stratify=y_train, random_state=SEED
    )
    probe = build(params)
    fit_with_early_stopping(probe, name, Xtr_p[fit_idx], y_train.iloc[fit_idx],
                            Xtr_p[val_idx], y_train.iloc[val_idx])
    best_iter = int(getattr(probe, BEST_ITER_ATTR[name]))
    model = build(params, n_estimators=best_iter, early_stopping=False)
    model.fit(Xtr_p, y_train)
    proba = model.predict_proba(Xte_p)[:, 1]
    joblib.dump({"prep": prep, "model": model}, MODELS_DIR / f"{name}.joblib")
    return all_metrics(y_test, proba), proba, best_iter


# --------------------------------------------------------------------------
# Logistic regression baseline (context)
# --------------------------------------------------------------------------
def fit_lr_baseline(X_train, y_train, X_test, y_test):
    pipe = Pipeline(
        [
            ("prep", PreprocessTransformer()),
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        {"lr__C": loguniform(1e-3, 1e2)},
        n_iter=12,
        cv=StratifiedKFold(n_splits=N_EVAL_FOLDS, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=0,
    )
    search.fit(X_train, y_train)
    proba = search.predict_proba(X_test)[:, 1]
    joblib.dump({"prep": search.best_estimator_.named_steps["prep"],
                 "model": search.best_estimator_}, MODELS_DIR / "lr.joblib")
    return all_metrics(y_test, proba), proba, search.best_params_


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_roc(probas: dict, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={sk_auc(fpr, tpr):.4f})", lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set (20%)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150); plt.close(fig)


def plot_pr(probas: dict, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={average_precision_score(y_test, p):.4f})", lw=1.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test set (20%)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=150); plt.close(fig)


def plot_calibration(probas: dict, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in probas.items():
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", ms=4, label=name, lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration on the held-out test set (10 quantile bins)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "calibration.png", dpi=150); plt.close(fig)


def plot_cv_boxplot(cv_folds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted(cv_folds["model"].unique())
    data = [cv_folds.loc[cv_folds["model"] == m, "auc"].values for m in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#DD8452", "#55A868"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("ROC-AUC"); ax.set_title("5-fold CV ROC-AUC by model (tuned hyperparameters)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "cv_auc_boxplot.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Data: {df.shape[0]} rows x {df.shape[1] - 1} features, "
          f"positive rate = {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Split: train={len(X_train)}, test={len(X_test)} (stratified holdout)")

    # sanity check the transformer end-to-end
    prep_sanity = PreprocessTransformer().fit(X_train)
    X_p = prep_sanity.transform(X_train)
    n_feat = X_p.shape[1]
    print(f"Feature matrix: {X_p.shape}, NaNs after transform: "
          f"{int(np.isnan(X_p).sum())}")

    results = {"data": {"rows": int(len(df)), "train": int(len(X_train)),
                        "test": int(len(X_test)), "positive_rate": float(y.mean()),
                        "n_features": int(n_feat),
                        "feature_names": prep_sanity.get_feature_names_out()}}

    # ---- 1. tune each GBDT ----------------------------------------------
    tuned = {}
    for name in BUILDERS:
        print(f"\n=== Tuning {name} ===")
        tuned[name] = tune_model(name, X_train, y_train)
        results.setdefault("tuning", {})[name] = {
            "best_oof_auc": tuned[name]["best_auc"],
            "best_params": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                            for k, v in tuned[name]["best_params"].items()},
            "top5_history": [{"auc": h[0], "params": str(h[1])}
                             for h in tuned[name]["history"]],
        }
        print(f"  best OOF AUC = {tuned[name]['best_auc']:.5f}")

    # ---- 2. 5-fold CV with tuned params (early stopping per fold) --------
    print("\n=== 5-fold CV (tuned hyperparameters, early stopping per fold) ===")
    cv_frames = []
    for name in BUILDERS:
        cv_df = run_cv(name, tuned[name]["best_params"], X_train, y_train)
        cv_frames.append(cv_df)
        agg = cv_df[["auc", "pr_auc", "ks", "gini", "brier", "log_loss"]].agg(["mean", "std"])
        results.setdefault("cv", {})[name] = {
            k: {"mean": float(agg.loc["mean", k]), "std": float(agg.loc["std", k])}
            for k in agg.columns
        }
        print(f"[{name}] CV AUC = {agg.loc['mean','auc']:.4f} +/- {agg.loc['std','auc']:.4f}"
              f"  (trees/fold: {int(cv_df['n_estimators'].mean())})")
    cv_folds = pd.concat(cv_frames, ignore_index=True)
    cv_folds.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    plot_cv_boxplot(cv_folds)

    # ---- 3. LR baseline ---------------------------------------------------
    print("\n=== Logistic Regression baseline (tuning C) ===")
    lr_test, lr_proba, lr_best = fit_lr_baseline(X_train, y_train, X_test, y_test)
    results["tuning"]["lr"] = {"best_params": lr_best}
    print(f"[lr] test AUC = {lr_test['auc']:.4f}")

    # ---- 4. test evaluation -----------------------------------------------
    print("\n=== Test evaluation (untouched 20% holdout) ===")
    probas = {"logistic_regression": lr_proba}
    for name in BUILDERS:
        test_m, proba, best_iter = refit_and_eval_test(
            name, tuned[name]["best_params"], X_train, y_train, X_test, y_test,
        )
        probas[name] = proba
        results.setdefault("test", {})[name] = test_m
        results["test"][name]["n_estimators"] = best_iter
        print(f"[{name}] test AUC = {test_m['auc']:.4f}, PR-AUC = {test_m['pr_auc']:.4f}, "
              f"KS = {test_m['ks']:.4f}, Brier = {test_m['brier']:.4f} "
              f"({best_iter} trees)")
    results["test"]["logistic_regression"] = lr_test

    # Youden threshold on the best model (by test AUC)
    best_model = max(probas, key=lambda m: results["test"][m]["auc"])
    fpr, tpr, thr = roc_curve(y_test, probas[best_model])
    youden = float(thr[np.argmax(tpr - fpr)])
    preds = (probas[best_model] >= youden).astype(int)
    tn = int(((preds == 0) & (y_test == 0)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tp = int(((preds == 1) & (y_test == 1)).sum())
    results["threshold"] = {
        "model": best_model,
        "youden_threshold": youden,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else None,
    }
    print(f"\nBest model: {best_model}, Youden threshold = {youden:.4f} -> "
          f"precision={results['threshold']['precision']:.4f}, "
          f"recall={results['threshold']['recall']:.4f}")

    # ---- 6. plots ----------------------------------------------------------
    plot_roc(probas, y_test)
    plot_pr(probas, y_test)
    plot_calibration(probas, y_test)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'metrics.json'}, "
          f"{RESULTS_DIR / 'cv_fold_metrics.csv'}, figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
