#!/usr/bin/env bash
# Emulate the GitHub Actions scheduled workflow (monitor.yml) locally, one week
# at a time, against the local MLflow sqlite store. This is the canonical way
# to run the monitoring loop end-to-end:
#
#   bash scripts/run_scheduled_local.sh
#
# It performs, for every weekly batch in order, exactly what the two workflow
# jobs do:
#   job 1 (drift-check):   python -m src.monitor --check-next
#   job 2 (retrain):       only when NEEDS_RETRAIN=true:
#                          python -m src.monitor --retrain-flagged
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the project virtualenv by default (override with PY=...)
PY="${PY:-.venv/bin/python}"
export MLFLOW_TRACKING_URI="sqlite:///mlflow/mlflow.db"
export MPLCONFIGDIR="$PWD/.mplconfig"

echo "==> Building the simulated stream (idempotent)"
"$PY" -m src.simulate_stream || true

echo "==> Bootstrap: train + register the initial model if needed"
"$PY" - <<'PY'
from src import registry
registry.setup_tracking()
if registry.production_version() is None:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "src.train", "--data-csv",
                    "data/stream/reference.csv",
                    "--description", "initial model on reference window",
                    "--tag", "trained_on=reference", "--tag", "trigger=initial"],
                   check=True)
    print("bootstrapped initial model v1")
else:
    print("production model already exists - skipping bootstrap")
PY

echo "==> Running the scheduled monitor loop (one batch per iteration)"
N_BATCHES=$("$PY" - <<'PY'
from src import config
print(config.load_json(config.MANIFEST_FILE)["n_batches"])
PY
)
for i in $(seq 1 "$N_BATCHES"); do
  echo ""
  echo "---- scheduled run #$i ----"
  OUT=$("$PY" -m src.monitor --check-next --no-html)
  echo "$OUT"
  NEEDS=$(echo "$OUT" | sed -n 's/^NEEDS_RETRAIN=\(.*\)$/\1/p')
  if [ "${NEEDS:-false}" = "true" ]; then
    echo ">> threshold crossed -> launching retrain job"
    "$PY" -m src.monitor --retrain-flagged
  else
    echo ">> no retrain needed"
  fi
done

echo ""
echo "==> Done. Refreshing reports."
"$PY" -m src.monitor --flush-labels
"$PY" -m src.figures
echo "==> Summary figures under reports/figures/"
ls -la reports/figures
