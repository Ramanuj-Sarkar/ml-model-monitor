"""Static summary figures + model-version CSV for the README and reports.

Reads the persisted timeline and deployment ledger and renders:
  reports/figures/timeline_auc.png      - realised AUC vs batch + retrain events
  reports/figures/timeline_drift.png    - PSI / prediction-PSI / chi2 signals
  reports/model_versions.csv            - version ledger (registry mirror)
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

BATCH_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
EPISODE_COLORS = {"covariate_util_shift": "#DD8452", "category_dependents": "#55A868",
                  "concept_util_flip": "#C44E52"}


def load_timeline() -> pd.DataFrame:
    df = pd.read_csv(config.TIMELINE_CSV)
    for c in ("auc", "max_psi", "prediction_psi", "auc_drop"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _episode_windows(manifest: dict) -> list[dict]:
    out = []
    for name, ep in manifest["episodes"].items():
        end = ep.get("end_batch") or int(manifest["n_batches"])
        out.append({"name": name, "start": int(ep["start_batch"]), "end": int(end),
                    "doc": ep.get("doc", "")})
    return out


def _shade_episodes(ax, windows, ymax):
    for i, w in enumerate(windows):
        color = EPISODE_COLORS.get(w["name"], "#999999")
        ax.axvspan(w["start"] - 0.5, w["end"] + 0.5, color=color, alpha=0.10,
                   label=w["name"].replace("_", " "))


def version_csv() -> pd.DataFrame:
    ledger = config.load_json(config.DEPLOYMENTS_FILE)
    rows = []
    for v, e in ledger.get("versions", {}).items():
        rows.append({
            "version": int(v),
            "alias": "production" if str(ledger.get("production")) == str(v) else "",
            "trained_on": e.get("trained_on"),
            "trigger": e.get("trigger"),
            "test_auc": e.get("test_auc"),
            "window_rows": e.get("window_rows"),
            "model_type": e.get("model_type"),
            "registered_at": e.get("registered_at"),
            "run_id": e.get("run_id"),
        })
    out = pd.DataFrame(rows).sort_values("version")
    out.to_csv(config.REPORTS_DIR / "model_versions.csv", index=False)
    return out


def figure_auc(tl: pd.DataFrame, windows: list[dict], out: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_episodes(ax, windows, 1.0)
    versions = sorted(tl["version"].dropna().unique())
    for v in versions:
        m = tl["version"] == v
        ax.plot(tl.loc[m, "batch"], tl.loc[m, "auc"], marker="o", ms=4, lw=1.4,
                color=BATCH_COLORS[(v - 1) % len(BATCH_COLORS)],
                label=f"v{v} deployed")
    retr = tl[tl["retrain_event"].fillna(False)]
    for _, r in retr.iterrows():
        ax.axvline(r["batch"], color="k", ls="--", lw=1.2, alpha=0.7)
        ax.annotate(f"retrain->v{int(r['retrain_to_version'])}", xy=(r["batch"], 0.62),
                    xytext=(r["batch"] + 0.6, 0.60), fontsize=8, rotation=90, va="top")
    ax.axhline(0.85, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("batch (week)"); ax.set_ylabel("realised AUC (labels +2 weeks)")
    ax.set_title("Realised model performance over the simulated production stream")
    ax.set_ylim(0.55, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def figure_drift(tl: pd.DataFrame, windows: list[dict], details: dict, out: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    _shade_episodes(ax1, windows, 5.0)
    _shade_episodes(ax2, windows, 5.0)

    # util PSI (the drifted numeric feature) + max PSI + prediction PSI
    util_psi, pred_psi, max_psi = [], [], []
    for b in sorted(details):
        cols = details[b]["per_column"]
        u = cols.get("RevolvingUtilizationOfUnsecuredLines", {}).get("drift_score")
        p = details[b].get("prediction_psi")
        mx = details[b].get("max_psi")
        util_psi.append(u if u is not None else np.nan)
        pred_psi.append(p if p is not None else np.nan)
        max_psi.append(mx if mx is not None else np.nan)
    x = sorted(details)
    ax1.plot(x, max_psi, color="#4C72B0", lw=1.6, label="max PSI (any numeric feature)")
    ax1.plot(x, util_psi, color="#DD8452", lw=1.2, alpha=0.8, label="util PSI")
    ax1.plot(x, pred_psi, color="#55A868", lw=1.2, ls="--",
             label="prediction PSI (score distribution)")
    ax1.axhline(config.PSI_THRESHOLD, color="r", lw=1, ls=":")
    ax1.text(1, config.PSI_THRESHOLD + 0.03, f"PSI threshold {config.PSI_THRESHOLD}",
             fontsize=8, color="r")
    ax1.set_yscale("symlog", linthresh=0.05)
    ax1.set_ylabel("PSI (log axis)"); ax1.set_title("Feature / prediction drift vs reference window")
    ax1.legend(fontsize=8, loc="upper left"); ax1.grid(alpha=0.3)

    deps_p = [details[b]["per_column"].get("NumberOfDependents", {}).get("drift_score")
              for b in x]
    deps_p = [v if v is not None and v > 0 else np.nan for v in deps_p]
    ax2.plot(x, deps_p, color="#55A868", lw=1.4, label="NumberOfDependents chi2 p-value")
    ax2.axhline(config.CHI2_PVALUE_THRESHOLD, color="r", lw=1, ls=":")
    ax2.text(1, config.CHI2_PVALUE_THRESHOLD * 3,
             f"chi2 p threshold {config.CHI2_PVALUE_THRESHOLD}", fontsize=8, color="r")
    ax2.set_yscale("log")
    ax2.set_ylabel("chi2 p-value (log)")
    ax2.set_xlabel("batch (week)")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3, which="both")
    retr = tl[tl["retrain_event"].fillna(False)]
    for axx in (ax1, ax2):
        for _, r in retr.iterrows():
            axx.axvline(r["batch"], color="k", ls="--", lw=1.2, alpha=0.6)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def load_details() -> dict:
    d = {}
    for p in sorted((config.REPORTS_DIR / "batch_details").glob("batch_*.json")):
        with open(p) as fh:
            r = json.load(fh)
        d[int(r["batch"])] = r
    return d


def main() -> int:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = config.load_json(config.MANIFEST_FILE)
    windows = _episode_windows(manifest)
    tl = load_timeline()
    details = load_details()
    version_csv()
    figure_auc(tl, windows, config.FIGURES_DIR / "timeline_auc.png")
    figure_drift(tl, windows, details, config.FIGURES_DIR / "timeline_drift.png")
    print("figures ->", config.FIGURES_DIR)
    print("model versions ->", config.REPORTS_DIR / "model_versions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
