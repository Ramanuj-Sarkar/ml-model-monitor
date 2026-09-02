"""Monitor driver: replays the production stream batch-by-batch and emulates
the scheduled workflow in .github/workflows/monitor.yml.

For every batch (week) it:

1. loads the model currently registered as ``production``,
2. checks the batch for drift against the *fixed* reference window (the
   initial v1 training window, a golden baseline): Evidently PSI / chi-squared
   report + KS supplements; prediction-distribution drift is part of the
   Evidently report,
3. evaluates the realised performance of the batch whose labels became
   available ``LABEL_LAG_BATCHES`` weeks ago, using the scores stored at
   arrival (the model that actually served the batch),
4. applies the retraining policy:
     - PSI > 0.2 on any monitored numeric feature,
     - chi-squared p < 0.01 with a material category-share change,
     - prediction-distribution PSI > 0.2,
     - smoothed realised AUC drop (mean over the last <=3 released batches)
       >= PERFORMANCE_DROP_MARGIN below the serving models' baselines,
   with a COOLDOWN_BATCHES guard against retrain storms,
5. retrains (version bump in the MLflow Model Registry, new ``production``
   alias) when the policy fires, exactly like the conditional retrain job in
   the workflow,
6. logs every batch's drift scores / flags regardless of the decision, so the
   timeline chart shows the full history, not only flagged points.

Modes
-----
    python -m src.monitor --replay            # process every batch in order
    python -m src.monitor --check-next        # one scheduled invocation
    python -m src.monitor --retrain-flagged   # retrain the last flagged batch
    python -m src.monitor --flush-labels      # evaluate trailing labels

``--check-next`` prints machine-readable ``NEEDS_RETRAIN=true|false`` and
``RETRAIN_EXECUTED=false`` lines; the retrain job is a separate
``--retrain-flagged`` invocation in the workflow.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config, drift_detection as dd, registry, retrain as retrain_mod

CAT_EFFECT_MIN = config.CAT_EFFECT_SIZE


# --------------------------------------------------------------------------
# State (cursor of the scheduled workflow)
# --------------------------------------------------------------------------
DEFAULT_STATE = {
    "last_processed_batch": 0,
    "last_retrain_batch": None,
    "last_retrain_version": None,
    "retrains": [],          # events in order: {batch, version, reason, ...}
}


def load_state() -> dict:
    if config.STATE_FILE.exists():
        st = config.load_json(config.STATE_FILE)
        return {**DEFAULT_STATE, **st}
    return dict(DEFAULT_STATE)


def save_state(state: dict) -> None:
    config.save_json(config.STATE_FILE, state)


# --------------------------------------------------------------------------
# Timeline persistence
# --------------------------------------------------------------------------
def persist_timeline(rows: dict) -> None:
    ordered = [rows[k] for k in sorted(rows, key=int)]
    details_dir = config.REPORTS_DIR / "batch_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    light = []
    for row in ordered:
        b = int(row["batch"])
        with open(details_dir / f"batch_{b:04d}_detail.json", "w") as fh:
            json.dump(row, fh, indent=1, default=str)
        light_row = {k: v for k, v in row.items() if k not in ("per_column", "ks")}
        light.append(light_row)
    pd.DataFrame(light).to_csv(config.TIMELINE_CSV, index=False)
    with open(config.TIMELINE_JSON, "w") as fh:
        json.dump(ordered, fh, indent=1, default=str)


# --------------------------------------------------------------------------
# Deployment helpers
# --------------------------------------------------------------------------
def current_deployment() -> dict:
    ledger = registry.load_deployments()
    prod = ledger.get("production")
    if prod is None:
        raise RuntimeError("no production model - run train.py first")
    return dict(ledger["versions"][str(prod)])


def _model_cache():
    cache = {}

    def load(version: int):
        if version not in cache:
            cache[version] = registry.load_model(version)
        return cache[version]

    return load


def serving_baseline_auc(version: int) -> float | None:
    ledger = registry.load_deployments()
    sv = ledger["versions"].get(str(version))
    return sv.get("test_auc") if sv else None


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def drift_reasons(record: dict) -> list[str]:
    """Retraining reasons coming from batch-k feature/prediction drift."""
    reasons = []
    for cname, info in record["per_column"].items():
        if cname == config.PREDICTION_COLUMN:
            continue
        if not info["drift_detected"]:
            continue
        if info.get("column_type") == "num":
            reasons.append(
                f"PSI {info['drift_score']:.2f} > {config.PSI_THRESHOLD} on '{cname}'")
        else:  # categorical: chi2 + effect-size gate
            effect = float(info.get("cat_effect") or 0.0)
            if effect >= CAT_EFFECT_MIN:
                reasons.append(
                    f"chi2 p={info['drift_score']:.2e} on '{cname}' "
                    f"(max share change {effect:.0%})")
    if record["prediction_drifted"]:
        reasons.append(f"prediction PSI {record['prediction_psi']:.2f} > "
                       f"{config.PSI_THRESHOLD}")
    return reasons


def _smoothed_perf_reason(rows: dict, released_batch: int) -> list[str]:
    """Mean AUC drop over the last <=3 released batches (labels are only
    available once per week, so this is the natural smoothing window)."""
    evaluated = [b for b in sorted((int(x) for x in rows), reverse=True)
                 if b <= released_batch and rows[str(b)].get("auc_drop") is not None]
    recent = evaluated[:3]
    if not recent:
        return []
    drops = [float(rows[str(b)]["auc_drop"]) for b in recent]
    mean = sum(drops) / len(drops)
    if mean >= config.PERFORMANCE_DROP_MARGIN:
        span = f"batches {recent[-1]}-{recent[0]}" if len(recent) > 1 else f"batch {recent[0]}"
        return [f"smoothed AUC drop {mean:.3f} over {span} "
                f"(>={config.PERFORMANCE_DROP_MARGIN} margin)"]
    return []


# --------------------------------------------------------------------------
# One step of the scheduled workflow
# --------------------------------------------------------------------------
def process_batch(batch_index: int, *, state: dict, rows: dict, model_loader,
                  reference: pd.DataFrame, do_retrain: bool = True,
                  save_html: bool = True) -> dict:
    deployed = current_deployment()
    version = int(deployed["version"])
    model = model_loader(version)

    # 1) the batch arrives: score + drift check with the deployed model
    record = dd.check_batch(batch_index, model=model, deployed=deployed,
                            reference=reference, save_html=save_html)
    row = dict(record)
    rows[str(batch_index)] = row

    # 2) labels of batch (batch_index - LAG) became available this week
    released = batch_index - config.LABEL_LAG_BATCHES
    perf_reason = []
    if released >= 1 and str(released) in rows:
        perf = dd.evaluate_batch_performance(released)
        serving_version = int(perf.get("served_by_version") or 0)
        base_auc = serving_baseline_auc(serving_version)
        drop = None
        if perf.get("auc") is not None and base_auc:
            drop = float(base_auc) - float(perf["auc"])
        target = rows[str(released)]
        target.update({
            "auc": perf.get("auc"),
            "pr_auc": perf.get("pr_auc"),
            "accuracy_0.5": perf.get("accuracy_0.5"),
            "log_loss": perf.get("log_loss"),
            "n_pos": perf.get("n_pos"),
            "perf_served_by_version": serving_version,
            "auc_drop": drop,
            "perf_flagged": bool(drop is not None and drop >= config.PERFORMANCE_DROP_MARGIN),
            "perf_evaluated_at": batch_index,
        })
        perf_reason = _smoothed_perf_reason(rows, released)

    # 3) assemble every flag & decide
    drift = drift_reasons(record)
    reasons = drift + perf_reason
    needs_retrain = len(reasons) > 0
    cooldown_ok = True
    if needs_retrain and state.get("last_retrain_batch") is not None:
        since = batch_index - int(state["last_retrain_batch"])
        cooldown_ok = since >= config.COOLDOWN_BATCHES

    decision = "ok"
    if needs_retrain:
        decision = "needs_retraining" if cooldown_ok else "suppressed_by_cooldown"

    row.update({
        "drift_reasons": drift,
        "perf_reasons": perf_reason,
        "reasons": reasons,
        "needs_retrain": needs_retrain,
        "decision": decision,
        "cooldown_suppressed": needs_retrain and not cooldown_ok,
        "released_batch": released if released >= 1 else None,
        "retrain_event": False,
        "retrain_to_version": None,
    })

    # 4) act: retrain when the policy fires (emulates the conditional job)
    executed = False
    if decision == "needs_retraining" and do_retrain:
        reason_txt = " | ".join(reasons)
        info = retrain_mod.retrain_for_batch(batch_index, reason_txt)
        new_version = int(info["version"])
        state["last_retrain_batch"] = batch_index
        state["last_retrain_version"] = new_version
        state["retrains"].append({
            "batch": batch_index,
            "version": new_version,
            "reason": reason_txt,
            "labelled_through_batch": info.get("labelled_through_batch"),
            "test_auc": info["metrics"]["auc"],
            "window_file": info.get("window_file"),
        })
        row["retrain_event"] = True
        row["retrain_to_version"] = new_version
        executed = True

    state["last_processed_batch"] = batch_index
    save_state(state)
    persist_timeline(rows)
    return {"batch": batch_index, "decision": decision, "executed": executed,
            "reasons": reasons,
            "released_perf_batch": released if released >= 1 else None}


def flush_trailing_labels(*, state: dict, rows: dict, n_batches: int) -> list[dict]:
    """At the end of the observed window the labels of the last
    LABEL_LAG_BATCHES batches have not arrived yet; this emulates those
    remaining label-only checkpoints so the timeline's performance curve is
    complete."""
    out = []
    for b in range(max(1, n_batches - config.LABEL_LAG_BATCHES + 1), n_batches + 1):
        row = rows.get(str(b))
        if row is None or row.get("auc") is not None:
            continue
        perf = dd.evaluate_batch_performance(b)
        serving_version = int(perf.get("served_by_version") or 0)
        base_auc = serving_baseline_auc(serving_version)
        drop = None
        if perf.get("auc") is not None and base_auc:
            drop = float(base_auc) - float(perf["auc"])
        row.update({
            "auc": perf.get("auc"), "pr_auc": perf.get("pr_auc"),
            "accuracy_0.5": perf.get("accuracy_0.5"), "log_loss": perf.get("log_loss"),
            "n_pos": perf.get("n_pos"),
            "perf_served_by_version": serving_version,
            "auc_drop": drop,
            "perf_flagged": bool(drop is not None and drop >= config.PERFORMANCE_DROP_MARGIN),
            "perf_evaluated_at": "end-of-window flush",
        })
        out.append({"batch": b, "auc_drop": drop})
    persist_timeline(rows)
    return out


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def _load_existing_rows() -> dict:
    if config.TIMELINE_JSON.exists():
        with open(config.TIMELINE_JSON) as fh:
            return {str(r["batch"]): r for r in json.load(fh)}
    return {}


def replay(until: int | None = None, save_html: bool = True) -> list[dict]:
    manifest = config.load_json(config.MANIFEST_FILE)
    n_batches = int(manifest["n_batches"])
    until = until or n_batches
    state = load_state()
    rows = _load_existing_rows()
    reference = dd.load_reference()
    loader = _model_cache()

    start = int(state.get("last_processed_batch", 0)) + 1
    outcomes = []
    for b in range(start, until + 1):
        print(f"\n=== week {b:>2}/{until} ({config.episode_name_for_batch(b)}) ===",
              flush=True)
        outcome = process_batch(b, state=state, rows=rows, model_loader=loader,
                                reference=reference, do_retrain=True,
                                save_html=save_html)
        outcomes.append(outcome)
        print(f"  decision={outcome['decision']} reasons={outcome['reasons']} "
              f"executed={outcome['executed']}", flush=True)
    flush_trailing_labels(state=state, rows=rows, n_batches=n_batches)
    return outcomes


def check_next(save_html: bool = True) -> int:
    """One scheduled invocation: process the next unprocessed batch (no inline
    retrain; that is the workflow's retrain job)."""
    manifest = config.load_json(config.MANIFEST_FILE)
    n_batches = int(manifest["n_batches"])
    state = load_state()
    rows = _load_existing_rows()
    reference = dd.load_reference()
    loader = _model_cache()

    nxt = int(state.get("last_processed_batch", 0)) + 1
    if nxt > n_batches:
        flush_trailing_labels(state=state, rows=rows, n_batches=n_batches)
        print("NO_MORE_BATCHES")
        return 0
    outcome = process_batch(nxt, state=state, rows=rows, model_loader=loader,
                            reference=reference, do_retrain=False,
                            save_html=save_html)
    needs = "true" if outcome["decision"] == "needs_retraining" else "false"
    print(f"NEEDS_RETRAIN={needs}")
    print(f"RETRAIN_EXECUTED=false")
    print(f"PROCESSED_BATCH={outcome['batch']} DECISION={outcome['decision']} "
          f"REASONS={' ; '.join(outcome['reasons'])}")
    return 0


def retrain_pending_action() -> int:
    """The workflow's retrain job: retrain on the updated window for the batch
    that was flagged by ``--check-next``."""
    state = load_state()
    rows = _load_existing_rows()
    last = int(state.get("last_processed_batch", 0))
    row = rows.get(str(last), {})
    if not row or not row.get("needs_retrain"):
        print(f"NOTHING_TO_RETRAIN (last processed batch {last} was not flagged)")
        return 1
    reasons = row.get("reasons", [])
    reason_txt = " | ".join(reasons) if reasons else "drift threshold crossed"
    info = retrain_mod.retrain_for_batch(last, reason_txt)
    state["last_retrain_batch"] = last
    state["last_retrain_version"] = int(info["version"])
    state["retrains"].append({
        "batch": last,
        "version": int(info["version"]),
        "reason": reason_txt,
        "labelled_through_batch": info.get("labelled_through_batch"),
        "test_auc": info["metrics"]["auc"],
        "window_file": info.get("window_file"),
    })
    row["retrain_event"] = True
    row["retrain_to_version"] = int(info["version"])
    rows[str(last)] = row
    save_state(state)
    persist_timeline(rows)
    print(f"RETRAIN_OK version={info['version']} auc={info['metrics']['auc']:.4f}")
    return 0


def generate_html_reports() -> int:
    """Regenerate the per-batch Evidently HTML reports exactly as each batch
    saw them (each batch is rendered with the model version that served it and
    the fixed reference window)."""
    rows = _load_existing_rows()
    ledger = registry.load_deployments()
    reference = dd.load_reference()
    loader = _model_cache()
    made = 0
    for key in sorted(rows, key=int):
        row = rows[key]
        version = int(row.get("version") or row.get("served_by_version") or 0)
        entry = ledger["versions"].get(str(version))
        if entry is None:
            print(f"skip batch {row['batch']}: no ledger entry for version {version}")
            continue
        model = loader(version)
        dd.check_batch(int(row["batch"]), model=model, deployed=entry,
                       reference=reference, save_html=True)
        made += 1
        print(f"html {row['batch']:>3}/44 (served by v{version})", flush=True)
    print(f"GENERATED_HTML={made}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", action="store_true", help="process every batch in order")
    mode.add_argument("--check-next", action="store_true",
                      help="one scheduled run (workflow job 1)")
    mode.add_argument("--retrain-flagged", action="store_true",
                      help="retrain the last flagged batch (workflow job 2)")
    mode.add_argument("--flush-labels", action="store_true")
    mode.add_argument("--generate-html", action="store_true",
                      help="regenerate per-batch Evidently HTML reports")
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--no-html", action="store_true", help="skip Evidently HTML per batch")
    args = ap.parse_args(argv)

    registry.setup_tracking()
    save_html = not args.no_html
    if args.replay:
        replay(until=args.until, save_html=save_html)
    elif args.check_next:
        check_next(save_html=save_html)
    elif args.retrain_flagged:
        return retrain_pending_action()
    elif args.flush_labels:
        manifest = config.load_json(config.MANIFEST_FILE)
        flush_trailing_labels(state=load_state(), rows=_load_existing_rows(),
                              n_batches=int(manifest["n_batches"]))
    elif args.generate_html:
        return generate_html_reports()
    return 0


if __name__ == "__main__":
    sys.exit(main())
