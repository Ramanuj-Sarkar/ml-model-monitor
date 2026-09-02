"""MLflow experiment / Model Registry helpers shared by trainer and monitor.

The project uses MLflow with a local SQLite backend store
(``sqlite:///mlflow/mlflow.db``) because the SQLAlchemy store supports the full
Model Registry API (registered versions, the ``production`` alias and tags).
Every training run (initial and every retrain) is logged there; the version
that is currently deployed carries the alias ``production`` and the tag
``stage = production``.

A lightweight mirror ledger (``data/stream/deployments.json``) is written next
to the MLflow DB so the dashboard and replay driver can render the model
timeline without querying MLflow for every row.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from . import config

_PRODUCTION_TAG = "stage"          # tag key on a model version
_PRODUCTION_TAG_VAL = "production"


def setup_tracking() -> str:
    """Point MLflow at the project store/experiment. Returns the URI."""
    os.environ.setdefault("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        artifact_loc = str(config.MLFLOW_DIR / "artifacts")
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=artifact_loc)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    return config.MLFLOW_TRACKING_URI


def _client() -> MlflowClient:
    setup_tracking()
    return MlflowClient()


def production_version(client: Optional[MlflowClient] = None) -> Optional[dict]:
    """Return the currently aliased production model version (None if absent)."""
    try:
        client = client or _client()
        mv = client.get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS)
        return _version_dict(mv)
    except Exception:
        return None


def registered_versions() -> list[dict]:
    client = _client()
    try:
        versions = client.search_model_versions(f"name = '{config.MODEL_NAME}'")
    except Exception:
        return []
    out = []
    for mv in sorted(versions, key=lambda v: int(v.version)):
        d = _version_dict(mv)
        d["aliases"] = client.get_model_version_aliases(config.MODEL_NAME, mv.version)
        out.append(d)
    return out


def _version_dict(mv) -> dict:
    return {
        "version": int(mv.version),
        "run_id": mv.run_id,
        "status": mv.status,
        "registered_at": getattr(mv, "creation_timestamp", None),
        "tags": dict(mv.tags or {}),
        "description": mv.description or "",
    }


def model_uri(version: int) -> str:
    return f"models:/{config.MODEL_NAME}/{version}"


def load_model(version: int):
    """Load a registered model version as a sklearn-style estimator."""
    setup_tracking()
    return mlflow.sklearn.load_model(model_uri(version))


def _trusted_types(model_type: str) -> list[str]:
    """skops serialization whitelist for the model classes we log."""
    if model_type == "lgbm":
        return ["lightgbm.sklearn.LGBMClassifier", "lightgbm.basic.Booster",
                "collections.OrderedDict"]
    if model_type == "xgb":
        return ["xgboost.sklearn.XGBClassifier", "xgboost.core.Booster",
                "collections.OrderedDict"]
    return []


def log_training_run(*, params: dict, metrics: dict, tags: dict, model,
                     input_example, signature, window_rows: int, description: str,
                     model_type: str = "lgbm", register: bool = True) -> dict:
    """Start a run, log everything, optionally register the model and make it
    the new production version. Returns {'run_id', 'version', 'test_auc'}."""
    setup_tracking()
    with mlflow.start_run(run_name=tags.get("run_name", f"train-{int(time.time())}")) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        mlflow.log_metric("window_rows", window_rows)

        registered_name = config.MODEL_NAME if register else None
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_name,
            input_example=input_example,
            signature=signature,
            skops_trusted_types=_trusted_types(model_type),
            metadata={"description": description, "window_rows": window_rows},
        )
        version = None
        if register:
            client = MlflowClient()
            if model_info.registered_model_version is not None:
                version = int(model_info.registered_model_version)
            else:
                # Fallback: newest version for this registered model
                versions = client.search_model_versions(f"name = '{config.MODEL_NAME}'")
                version = max(int(v.version) for v in versions)
            # version bump: alias moves to the new version, tag it production
            client.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, version)
            client.set_model_version_tag(config.MODEL_NAME, version, _PRODUCTION_TAG, _PRODUCTION_TAG_VAL)
            client.set_model_version_tag(config.MODEL_NAME, version, "trained_on", tags.get("trained_on", ""))
        run_id = run.info.run_id
    return {"run_id": run_id, "version": version, "metrics": metrics}


def read_run_metrics(run_id: str) -> dict:
    client = _client()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def read_run_params(run_id: str) -> dict:
    client = _client()
    run = client.get_run(run_id)
    return dict(run.data.params)


def append_deployment(entry: dict) -> None:
    """Mirror a registered version into data/stream/deployments.json."""
    ledger = load_deployments()
    versions = ledger.setdefault("versions", {})
    key = str(entry["version"])
    versions[key] = entry
    if entry.get("alias") == config.PRODUCTION_ALIAS:
        ledger["production"] = key
    with open(config.DEPLOYMENTS_FILE, "w") as fh:
        json.dump(ledger, fh, indent=2, default=str)


def load_deployments() -> dict:
    if config.DEPLOYMENTS_FILE.exists():
        with open(config.DEPLOYMENTS_FILE) as fh:
            return json.load(fh)
    return {"versions": {}, "production": None}
