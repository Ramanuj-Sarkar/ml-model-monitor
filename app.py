"""Streamlit dashboard: scrub through the simulated production stream.

Shows drift scores, model performance and retraining events on one timeline,
and embeds the original Evidently HTML drift report for each batch.

Run:
    python -m streamlit run app.py
    # optional: pre-generate all Evidently HTML reports first (faster scrubbing)
    python -m src.monitor --generate-html
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, drift_detection as dd, registry

st.set_page_config(page_title="ML drift monitor - credit risk", layout="wide",
                   page_icon="📊")

st.title("📊 Production drift monitor - credit-risk classifier")
st.caption("Simulated weekly batches · Evidently drift reports · MLflow Model Registry · "
           "retraining on threshold breach")


# --------------------------------------------------------------------------
# Cached data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_timeline() -> pd.DataFrame:
    df = pd.read_csv(config.TIMELINE_CSV)
    for c in ("auc", "max_psi", "prediction_psi", "auc_drop", "pr_auc", "accuracy_0.5",
              "deployed_test_auc", "log_loss"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return config.load_json(config.MANIFEST_FILE)


@st.cache_data(show_spinner=False)
def load_versions() -> pd.DataFrame:
    path = config.REPORTS_DIR / "model_versions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def batch_detail(batch: int) -> dict:
    p = config.REPORTS_DIR / "batch_details" / f"batch_{batch:04d}_detail.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_resource(show_spinner=False)
def model_for(version: int):
    return registry.load_model(int(version))


def episode_spans(manifest: dict) -> list[tuple[int, int, str, str]]:
    spans = []
    for name, ep in manifest.get("episodes", {}).items():
        end = ep.get("end_batch") or int(manifest["n_batches"])
        spans.append((int(ep["start_batch"]), int(end), name, ep.get("doc", "")))
    return spans


# --------------------------------------------------------------------------
# Header KPIs
# --------------------------------------------------------------------------
tl = load_timeline()
manifest = load_manifest()
versions = load_versions()
spans = episode_spans(manifest)

retrains = tl[tl["retrain_event"].fillna(False)]
flagged = tl[tl["needs_retrain"].fillna(False)]
perf_drops = tl[tl["perf_flagged"].fillna(False)]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Batches monitored", int(manifest["n_batches"]))
c2.metric("Model versions (registry)", len(versions) if not versions.empty else 0)
c3.metric("Retrains triggered", int(retrains.shape[0]))
c4.metric("Batches flagged", int(flagged.shape[0]))
c5.metric("Batches w/ AUC drop ≥ margin", int(perf_drops.shape[0]))

st.markdown(
    "**Reference window:** the initial 40 000-row training window (fixed golden baseline). "
    "**Retraining policy:** any feature PSI > 0.2, chi-squared p < 0.01 with a material share "
    "change, prediction PSI > 0.2, or a smoothed realised-AUC drop ≥ 0.05 "
    "(with a 6-week cooldown). Every batch is logged whether flagged or not."
)

tab_tl, tab_inspect, tab_models, tab_about = st.tabs(
    ["Timeline", "Batch inspector", "Model registry", "About / drift types"])

# --------------------------------------------------------------------------
# Timeline tab
# --------------------------------------------------------------------------
with tab_tl:
    st.subheader("One timeline: drift, performance, retrain events")

    def add_episode_shading(fig):
        for start, end, name, _ in spans:
            color = {"covariate_util_shift": "rgba(221,132,82,0.15)",
                     "category_dependents": "rgba(85,168,104,0.15)",
                     "concept_util_flip": "rgba(196,78,82,0.18)"}.get(name, "rgba(0,0,0,0.05)")
            fig.add_vrect(x0=start - 0.5, x1=end + 0.5, fillcolor=color, line_width=0,
                          layer="below")
        return fig

    # --- chart 1: realised AUC -------------------------------------------
    fig = go.Figure()
    auc_df = tl.dropna(subset=["auc"])
    for v in sorted(tl["version"].dropna().unique()):
        m = auc_df[auc_df["version"] == v]
        if m.empty:
            continue
        fig.add_trace(go.Scatter(x=m["batch"], y=m["auc"], mode="lines+markers",
                                 name=f"v{int(v)} deployed", line=dict(width=2),
                                 marker=dict(size=6)))
    for _, r in retrains.iterrows():
        fig.add_vline(x=r["batch"], line_dash="dash", line_color="black", opacity=0.7)
        fig.add_annotation(x=r["batch"], y=0.58, text=f"retrain → v{int(r['retrain_to_version'])}",
                           showarrow=False, textangle=90, font=dict(size=10))
    add_episode_shading(fig)
    fig.add_hline(y=0.86, line_dash="dot", line_color="grey", opacity=0.6)
    fig.update_layout(height=380, margin=dict(t=10, b=10),
                      yaxis_title="realised AUC (labels +2 wks)",
                      xaxis_title="batch (week)", yaxis_range=[0.5, 1.0],
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # --- chart 2: drift signals ------------------------------------------
    details = {int(b): batch_detail(int(b)) for b in tl["batch"]}
    fig2 = go.Figure()
    xs = list(details)
    fig2.add_trace(go.Scatter(x=xs, y=[details[b].get("max_psi") for b in xs],
                              name="max feature PSI", line=dict(color="#4C72B0")))
    fig2.add_trace(go.Scatter(x=xs, y=[details[b].get("prediction_psi") for b in xs],
                              name="prediction PSI", line=dict(color="#55A868", dash="dash")))
    fig2.add_hline(y=config.PSI_THRESHOLD, line_dash="dot", line_color="red",
                   annotation_text=f"PSI {config.PSI_THRESHOLD}")
    fig2.update_layout(height=340, margin=dict(t=10, b=10),
                       yaxis=dict(title="PSI", type="log"),
                       xaxis_title="batch (week)", legend=dict(orientation="h"))
    add_episode_shading(fig2)
    for _, r in retrains.iterrows():
        fig2.add_vline(x=r["batch"], line_dash="dash", line_color="black", opacity=0.6)
    st.plotly_chart(fig2, use_container_width=True)

    # --- chart 3: categorical chi2 + KS ----------------------------------
    fig3 = go.Figure()
    dep_p = []
    for b in xs:
        v = details[b]["per_column"].get("NumberOfDependents", {}).get("drift_score")
        dep_p.append(v if v and v > 0 else None)
    fig3.add_trace(go.Scatter(x=xs, y=dep_p, name="NumberOfDependents chi2 p",
                              line=dict(color="#C44E52"), mode="lines+markers"))
    fig3.add_hline(y=config.CHI2_PVALUE_THRESHOLD, line_dash="dot", line_color="red",
                   annotation_text="chi2 p threshold")
    fig3.update_layout(height=300, margin=dict(t=10, b=10), yaxis_type="log",
                       xaxis_title="batch (week)", yaxis_title="p-value (log)",
                       legend=dict(orientation="h"))
    add_episode_shading(fig3)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        "**How to read it:** weeks 11-14 show util PSI > 0.2 (covariate drift); weeks 21-24 show "
        "the NumberOfDependents chi2 signal (category-frequency drift); weeks 31+ are *silent* on "
        "all feature tests while realised AUC collapses - that is concept drift (the "
        "input→label relationship changed, inputs did not). Vertical dashed lines mark retrains."
    )

# --------------------------------------------------------------------------
# Batch inspector
# --------------------------------------------------------------------------
with tab_inspect:
    st.subheader("Scrub through batches")
    batch = st.slider("Batch (week)", 1, int(manifest["n_batches"]), 11,
                      help="Each step corresponds to one weekly check in the simulated stream.")
    d = batch_detail(batch)
    if not d:
        st.warning(f"No detail record for batch {batch}.")
    else:
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Served by", f"v{int(d['version'])}")
        colB.metric("Episode", d.get("episode", "clean").replace("_", " "))
        colC.metric("Decision", d.get("decision", "ok"))
        colD.metric("Max feature PSI", f"{d.get('max_psi', 0):.3f}")
        colE, colF, colG, colH = st.columns(4)
        colE.metric("Prediction PSI", f"{d.get('prediction_psi', 0):.3f}")
        colF.metric("Drifted features", ", ".join(d.get("flagged_features", [])) or "none")
        colG.metric("Realised AUC", f"{d.get('auc', float('nan')):.3f}"
                    if d.get("auc") is not None else "labels not in yet")
        colH.metric("AUC drop", f"{d.get('auc_drop', 0):+.3f}" if d.get("auc_drop") is not None
                    else "n/a")
        if d.get("reasons"):
            st.error("Flagged for retraining:\n\n- " + "\n- ".join(d["reasons"]))
        elif d.get("cooldown_suppressed"):
            st.warning("Drift present but retrain suppressed by the cooldown window.")
        else:
            st.success("No threshold crossed for this batch.")

        if d.get("auc") is not None and d.get("perf_flagged"):
            st.info("Realised AUC dropped past the margin on this batch "
                    "(concept-drift type signal).")

        st.markdown("**Drift per column (Evidently)**")
        cols = d.get("per_column", {})
        rows = []
        for name, info in cols.items():
            rows.append({
                "feature": name,
                "type": info.get("column_type"),
                "test": info.get("stattest_name"),
                "score": f"{info.get('drift_score'):.4g}" if info.get("drift_score") is not None else "",
                "threshold": info.get("threshold"),
                "drifted": "✔" if info.get("drift_detected") else "",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**KS supplements (scipy, numeric features)**")
        ks_rows = [{"feature": k, **v} for k, v in (d.get("ks") or {}).items()]
        if ks_rows:
            st.dataframe(pd.DataFrame(ks_rows).round(4), use_container_width=True,
                         hide_index=True)

        st.markdown("**Evidently HTML drift report**")
        html_path = config.HTML_REPORT_DIR / f"batch_{batch:04d}_report.html"
        if not html_path.exists():
            if st.button(f"Generate Evidently HTML report for batch {batch}",
                         key=f"gen_{batch}"):
                with st.spinner("Rendering Evidently report..."):
                    try:
                        ledger = registry.load_deployments()
                        entry = ledger["versions"][str(int(d["version"]))]
                        dd.check_batch(batch, model=model_for(int(d["version"])),
                                       deployed=entry, reference=dd.load_reference(),
                                       save_html=True)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not generate report: {exc}")
        if html_path.exists():
            st.caption(f"reports/html/{html_path.name} "
                       f"({html_path.stat().st_size / 1e6:.1f} MB)")
            with open(html_path, encoding="utf-8") as fh:
                html = fh.read()
            st.components.v1.html(html, height=1400, scrolling=True)
        else:
            st.info("Report not generated yet - click the button above (or run "
                    "`python -m src.monitor --generate-html` for all batches).")

# --------------------------------------------------------------------------
# Registry tab
# --------------------------------------------------------------------------
with tab_models:
    st.subheader("MLflow Model Registry: credit-risk-classifier")
    if not versions.empty:
        st.dataframe(versions, use_container_width=True, hide_index=True)
    st.markdown(
        "Each training run (initial + retrains) was logged to MLflow tracking with parameters, "
        "metrics and the model artifact, and registered as a **new model version**; the current "
        "version carries the `production` alias + `stage=production` tag. "
        "Run `mlflow ui` inside the repo for the full experiment view "
        f"(sqlite store: {config.MLFLOW_DB.name})."
    )
    try:
        prod = registry.production_version()
        if prod:
            st.success(f"Production alias currently points to **version {prod['version']}**.")
    except Exception:
        pass

# --------------------------------------------------------------------------
# About tab
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("What is being tracked, and the drift-type distinctions")
    st.markdown(
        """
**Data drift** - the *input* distribution changed (PSI > 0.2 on a numeric feature, chi2 on the
categorical feature). Weeks 11-14 (util shifted upward) and 21-24 (dependents re-weighted) are
data drift. The model can still be perfectly good: AUC stayed ≈ 0.86 through both episodes.

**Prediction-distribution drift** - the deployed model's *score distribution* changed vs what it
produces on the reference window (the `model_score` PSI inside the Evidently report). Score drift
can be caused by data drift or by a model swap; it is not the same as model *performance*.

**Concept drift** - the relationship between inputs and the label changed while inputs stayed the
same. Weeks 31+ relabel util≥0.6 accounts to "no longer defaults", so *every* feature test stays
silent (PSI ~ 0.01, chi2 quiet) yet realised AUC collapses from ≈0.86 to ≈0.69-0.75. This is only
visible through **actual model performance** on delayed labels - the two retrains at weeks 34 and
40 are concept-drift responses.

**Realised performance vs prediction drift** - a model can keep outputting the same score
distribution while being wrong (concept drift above), and can change its score distribution after
a retrain without any drop in quality. That is why this project monitors *both* Evidently drift
metrics and realised AUC/accuracy once true labels arrive, and why the retraining policy combines
both signal families.

See **README.md** for the full methodology, the exact drift simulation, thresholds and how the
`.github/workflows/monitor.yml` scheduled workflow reproduces this locally.
        """
    )

st.caption(f"timeline: {config.TIMELINE_CSV} · generated {time.strftime('%Y-%m-%d %H:%M')}")
