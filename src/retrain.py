"""Retraining on an updated data window (the workflow's retrain job).

Given the batch that triggered the drift alarm, assembles the *updated data
window* the next model should be trained on:

* labels are only available for batches up to ``current_batch - LABEL_LAG``,
* the window is the most recent ``RETRAIN_WINDOW_BATCHES`` labelled batches
  once enough history exists (rolling window - this is what lets a retrained
  model actually adapt to a *concept* drift: old-regime rows roll out of the
  training set), otherwise the reference window plus every labelled batch,
* ``src/train.py`` is invoked on the assembled window, which trains the model,
  logs it to MLflow and registers it as the new ``production`` version.

CLI (used by the retrain job in .github/workflows/monitor.yml):

    python -m src.retrain --trigger-batch 33 --reason "AUC drop ..." \
        --trained-on "reference+batches_1..31"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, registry
from .train import train_on_window

# how many recent labelled batches make up the rolling retraining window
RETRAIN_WINDOW_BATCHES = 12


def assemble_window_files(through_batch: int, manifest=None) -> list[Path]:
    """Files (in chronological order) of the updated training window."""
    manifest = manifest or config.load_json(config.MANIFEST_FILE)
    n_batches = int(manifest["n_batches"])
    labelled = min(int(through_batch), n_batches)
    if labelled < RETRAIN_WINDOW_BATCHES:
        files = [config.STREAM_DIR / "reference.csv"] + [
            config.batch_csv(b) for b in range(1, labelled + 1)
        ]
    else:
        files = [
            config.batch_csv(b)
            for b in range(labelled - RETRAIN_WINDOW_BATCHES + 1, labelled + 1)
        ]
    return files


def window_label(files: list[Path], labelled_through: int) -> str:
    """Human-readable description of a window: full history or rolling."""
    if any(p.name == "reference.csv" for p in files):
        return f"reference+batches_1..{labelled_through}"
    nums = [int(p.stem.split("_")[1]) for p in files if p.stem.startswith("batch_")]
    return f"rolling_batches_{min(nums)}..{max(nums)}" if nums else f"batches_1..{labelled_through}"


def assemble_window_csv(through_batch: int) -> Path:
    files = assemble_window_files(through_batch)
    out = config.STREAM_DIR / f"train_window_thru_batch_{through_batch:04d}.csv"
    frames = [pd.read_csv(p) for p in files]
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out


def retrain_for_batch(trigger_batch: int, reason: str, model_type: str = "lgbm",
                      description: str | None = None) -> dict:
    """Retrain on the window of everything labelled at ``trigger_batch`` and
    register the new production model. Returns registry info."""
    labelled_through = int(trigger_batch) - config.LABEL_LAG_BATCHES
    window_files = assemble_window_files(max(labelled_through, 0))
    window = assemble_window_csv(max(labelled_through, 0))
    trained_on = window_label(window_files, max(labelled_through, 0))
    desc = description or f"retrain triggered at batch {trigger_batch}: {reason}"
    info = train_on_window(
        [window],
        model_type=model_type,
        register=True,
        description=desc,
        tags={"trained_on": trained_on, "trigger": reason,
              "trigger_batch": str(trigger_batch)},
    )
    info["labelled_through_batch"] = int(labelled_through)
    info["window_file"] = str(window)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trigger-batch", type=int, required=True)
    ap.add_argument("--reason", default="drift threshold crossed")
    ap.add_argument("--model-type", default="lgbm")
    args = ap.parse_args(argv)
    registry.setup_tracking()
    info = retrain_for_batch(args.trigger_batch, args.reason, model_type=args.model_type)
    print(f"RETRAIN_OK version={info['version']} run={info['run_id'][:8]} "
          f"auc={info['metrics']['auc']:.4f} labelled_through={info['labelled_through_batch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
