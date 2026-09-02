"""Central configuration for the model-monitoring project.

Every component (stream builder, trainer, drift detector, monitor, retrainer,
workflow, dashboard) reads its knobs from here so the numbers quoted in the
README always match the code.
"""
from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATA_CSV = DATA_DIR / "structured-ml-dataset.csv"
STREAM_DIR = DATA_DIR / "stream"
REPORTS_DIR = ROOT / "reports"
HTML_REPORT_DIR = REPORTS_DIR / "html"
FIGURES_DIR = REPORTS_DIR / "figures"
MLFLOW_DIR = ROOT / "mlflow"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"

STATE_FILE = STREAM_DIR / "monitor_state.json"        # processed-batch cursor + decisions
DEPLOYMENTS_FILE = STREAM_DIR / "deployments.json"    # version -> training window ledger
MANIFEST_FILE = STREAM_DIR / "manifest.json"          # stream layout + drift-episode calendar
TIMELINE_CSV = REPORTS_DIR / "timeline.csv"           # one row per batch (full history)
TIMELINE_JSON = REPORTS_DIR / "timeline.json"
MODEL_VERSIONS_CSV = REPORTS_DIR / "model_versions.csv"

for _d in (STREAM_DIR, REPORTS_DIR, HTML_REPORT_DIR, FIGURES_DIR, MLFLOW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Experiment / registry identity
# --------------------------------------------------------------------------
EXPERIMENT_NAME = "credit-risk-monitor"
MODEL_NAME = "credit-risk-classifier"      # Model Registry name
PRODUCTION_ALIAS = "production"

# --------------------------------------------------------------------------
# Data schema
# --------------------------------------------------------------------------
TARGET = "SeriousDlqin2yrs"

# All 10 features are fed to the model as numeric.
MODEL_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
# For drift monitoring the count feature NumberOfDependents is treated as a
# *categorical* column (chi-squared test); every other feature is numeric.
MONITOR_NUMERIC_FEATURES = [c for c in MODEL_FEATURES if c != "NumberOfDependents"]
MONITOR_CATEGORICAL_FEATURES = ["NumberOfDependents"]
PREDICTION_COLUMN = "model_score"          # Evidently prediction column name

# Winsorization caps applied ONCE when the stream is built (outlier hygiene,
# e.g. the dataset's famous RevolvingUtilization value of 50708).  Caps are
# deliberately wide so they do not interfere with injected drift.
CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": (0.0, 1.5),
    "age": (18.0, 100.0),
    "NumberOfTime30-59DaysPastDueNotWorse": (0.0, 20.0),
    "DebtRatio": (0.0, 10000.0),
    "MonthlyIncome": (0.0, 50000.0),
    "NumberOfOpenCreditLinesAndLoans": (0.0, 40.0),
    "NumberOfTimes90DaysLate": (0.0, 20.0),
    "NumberRealEstateLoansOrLines": (0.0, 10.0),
    "NumberOfTime60-89DaysPastDueNotWorse": (0.0, 20.0),
    "NumberOfDependents": (0.0, 13.0),
}
# NumberOfDependents is top-coded at 4 before monitoring: the raw tail has
# handful-of-row categories (13, 20 dependents) whose tiny expected counts
# destabilise the chi-squared test.  Only ~1% of rows are affected.
TOPCODES = {"NumberOfDependents": 4.0}

# Minimum absolute share difference (percentage points of the pmf) for a
# chi-squared flag to count as a retraining signal.  chi2 with n>=2500 is
# hypersensitive; tiny share moves should not trigger retrains.
CAT_EFFECT_SIZE = 0.03

# --------------------------------------------------------------------------
# Stream simulation (see README "Simulating the production stream")
# --------------------------------------------------------------------------
SEED = 42
REFERENCE_SIZE = 40_000          # first "9 months": the initial training window
BATCH_SIZE = 2_500               # one simulated week per batch
# Labels become available LABEL_LAG_BATCHES weeks after a batch arrives
# (a realistic label delay).
LABEL_LAG_BATCHES = 2

# Drift episodes injected into later batches (1-based batch numbers, where
# batch 1 is the first week after the reference window). See manifest.
#   1) Covariate drift : shift of RevolvingUtilizationOfUnsecuredLines
#   2) Category drift  : frequency re-weight of NumberOfDependents
#   3) Concept drift   : label <-> feature relationship flipped for util>=0.8
EPISODES = {
    "covariate_util_shift": {
        "kind": "numeric_shift",
        "start_batch": 11,
        "end_batch": 14,
        "feature": "RevolvingUtilizationOfUnsecuredLines",
        # util' = clip(util * MUL + ADD)
        "mul": 1.6,
        "add": 0.15,
        "doc": "mean + variance shift of revolving credit utilisation (covariate / data drift)",
    },
    "category_dependents": {
        "kind": "categorical_reweight",
        "start_batch": 21,
        "end_batch": 24,
        "feature": "NumberOfDependents",
        # Re-sample dependents from an altered probability mass function.
        "pmf": {0: 0.15, 1: 0.38, 2: 0.28, 3: 0.13, 4: 0.04, 5: 0.015, 6: 0.005, 8: 0.0},
        "doc": "category-frequency drift of NumberOfDependents (chi-squared fires)",
    },
    "concept_util_flip": {
        "kind": "label_relationship_flip",
        "start_batch": 31,
        "end_batch": None,             # persists to the end of the stream
        "feature": "RevolvingUtilizationOfUnsecuredLines",
        "segment_feature": "RevolvingUtilizationOfUnsecuredLines",
        "segment_min": 0.6,
        "flip_frac": 0.9,              # rows in the segment get y' = 0 w.p. flip_frac
        "doc": "feature->label relationship flipped: util>=0.6 accounts no longer default (calibrated to drop AUC ~0.13 while inputs stay unchanged)",
    },
}

# --------------------------------------------------------------------------
# Monitoring / retraining policy (see README "Monitoring methodology")
# --------------------------------------------------------------------------
PSI_THRESHOLD = 0.2            # numeric features: retrain gate
CHI2_PVALUE_THRESHOLD = 0.01   # categorical features: retrain gate
PERFORMANCE_DROP_MARGIN = 0.05 # absolute AUC loss vs deployed baseline
COOLDOWN_BATCHES = 6           # no retrain until >= this many batches after a retrain
EV_REFERENCE_SAMPLE = 4000     # rows handed to Evidently as the reference window
TRAIN_WINDOW_MAX_ROWS = 80_000 # rolling retraining window (reference + batches)

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def batch_label(batch_index: int) -> str:
    return f"batch_{batch_index:04d}"

def batch_csv(batch_index: int) -> Path:
    return STREAM_DIR / f"{batch_label(batch_index)}.csv"

def episode_name_for_batch(batch_index: int) -> str:
    """Which (if any) episode a batch belongs to, for manifest / labels."""
    for name, ep in EPISODES.items():
        start = ep["start_batch"]
        end = ep["end_batch"] if ep["end_batch"] is not None else 10**9
        if start <= batch_index <= end:
            return name
    return "clean"

def stream_meta() -> dict:
    """n_batches, episode windows, etc. (single source of truth for both the
    stream builder and the monitor)."""
    n_total = REFERENCE_SIZE
    # number of full batches that fit after the reference window
    return {"reference_size": REFERENCE_SIZE, "batch_size": BATCH_SIZE}


def load_json(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
