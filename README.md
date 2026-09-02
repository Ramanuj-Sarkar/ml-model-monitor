# ML model monitoring with Evidently, MLflow & Streamlit

An end-to-end **model drift monitoring** project for a LightGBM/XGBoost credit-risk
classifier. It simulates a production stream from static historical data, injects three
kinds of deliberate drift into later "weeks", monitors every incoming batch with Evidently
(feature drift via PSI/KS/chi², prediction-distribution drift, and realised model
performance on delayed labels), logs every batch - flagged or not - and automatically
retrains through an MLflow-registered "production" model when thresholds are crossed.

```
data/structured-ml-dataset.csv ──► src/simulate_stream.py ──► reference window + 44 weekly batches
                                                                  (drift injected in later batches)
                                                                        │
                        ┌───────────────────────────────────────────────┘
                        ▼
              src/train.py  (LightGBM / XGBoost) ──► MLflow Tracking + Model Registry (production alias)
                        ▲                                                  │
                        │  retrain on updated window                       ▼
                        │        ▲                               incoming batch k arrives
        src/retrain.py ─┘        │                                        │
                        │        │                              src/drift_detection.py (Evidently)
       .github/workflows/monitor.yml  ◄── flag ── src/monitor.py ◄────────┘
        (scheduled jobs, conditional retrain)            │
                                                         ▼
                                    reports/timeline.csv · reports/html/batch_*.html
                                                         │
                                    app.py (Streamlit scrub UI) · reports/figures/*.png
```

The whole monitoring loop is deterministic and reproducible locally with one command
(`scripts/run_scheduled_local.sh`), and the same logic is encoded as a GitHub Actions
scheduled workflow (`.github/workflows/monitor.yml`).

---

## 1. Repository layout

| Path                                   | Purpose                                                                           |
|----------------------------------------|-----------------------------------------------------------------------------------|
| `data/structured-ml-dataset.csv`       | raw historical dataset (150k rows, credit-risk default prediction)                |
| `data/stream/` (generated)             | `reference.csv` + `batch_0001..0044.csv` + manifest/state/deployments ledger      |
| `src/config.py`                        | every knob: paths, schema, thresholds, drift episodes                             |
| `src/data_prep.py`                     | reading/cleaning utilities (caps, imputation, top-coding)                         |
| `src/simulate_stream.py`               | step 2: build the reference window + weekly batches + inject drift                |
| `src/train.py`                         | step 1 + 6: train LightGBM/XGBoost, log to MLflow, register production version    |
| `src/registry.py`                      | MLflow tracking/registry helpers (alias `production`, version ledger)             |
| `src/drift_detection.py`               | step 3 + 4: Evidently report per batch, KS supplements, delayed-label performance |
| `src/monitor.py`                       | step 5: the "scheduled job" - process one batch, flag, retrain on breach          |
| `src/retrain.py`                       | assembles the updated data window and re-runs training                            |
| `src/figures.py`                       | static summary figures + model-version CSV                                        |
| `app.py`                               | step 7: Streamlit scrub UI                                                        |
| `.github/workflows/monitor.yml`        | step 5: scheduled workflow with conditional retrain job                           |
| `scripts/run_scheduled_local.sh`       | local emulation of the workflow's two jobs                                        |
| `reports/` (generated)                 | `timeline.csv`, per-batch detail JSONs, `html/batch_*.html`, `figures/*.png`      |
| `mlflow/` (generated)                  | SQLite tracking store + artifacts (registry backend)                              |
| `archive/model_comparison_original.py` | the original research/benchmarking trainer, superseded by `src/train.py`          |

The original research script (tuning XGBoost vs LightGBM vs CatBoost with CV) is archived;
the production pipeline uses `src/train.py`, which the monitor can call hundreds of times a
week, so it deliberately trains fast with fixed, documented hyper-parameters.

---

## 2. Quickstart

```bash
# environment (Python >= 3.10; developed on 3.14)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) build the stream (reference window + 44 weekly batches, drift injected)
python -m src.simulate_stream

# 2) train + register the initial model (version 1, alias production)
python -m src.train --data-csv data/stream/reference.csv --model-type lgbm \
    --description "initial model on reference window" \
    --tag trained_on=reference --tag trigger=initial

# 3) run the whole monitoring loop exactly like the scheduled workflow
bash scripts/run_scheduled_local.sh          # 44 weekly checks + conditional retrains

# 4) regenerate the per-batch Evidently HTML reports (optional, ~150 MB) and charts
python -m src.monitor --generate-html
python -m src.figures

# 5) dashboards
mlflow ui                 # experiment + registry view (sqlite backend)
python -m streamlit run app.py
```

`python -m src.monitor --replay` is a one-shot convenience that does the same as step 3
without shelling out per batch; `scripts/run_scheduled_local.sh` mirrors the GitHub
workflow's two jobs more literally.

---

## 3. Data & model (`src/train.py`)

The dataset is a classic binary-credit-risk table (150 000 rows x 10 features + target
`SeriousDlqin2yrs`; ~6.7% positives). It contains missing values (encoded `NA`) and
famous outliers (e.g. `RevolvingUtilizationOfUnsecuredLines == 50708`), so
`src/data_prep.py` applies deterministic hygiene **once** when the stream is built:

* winsorisation caps (wide, e.g. utilisation capped at 1.5),
* top-coding of `NumberOfDependents` at 4 (the raw tail has handful-of-row categories
  that destabilise chi-squared tests; ~1% of rows affected),
* median imputation with medians computed on the **reference window only** (as a
  deployment would), so training, monitoring and scoring all see identical inputs.

`src/train.py` trains a LightGBM (default) or XGBoost classifier on the union of the
CSVs it is given, using early stopping on an inner 10% split and refitting on the full
window at the optimal tree count. It logs **every training run** (parameters, metrics,
tags, model artifact) to MLflow and registers the model:

* tracking store: `sqlite:///mlflow/mlflow.db` (SQLAlchemy store so the full Registry
  API - versions, alias, tags - works without a server);
* every run becomes a **new registered model version** of `credit-risk-classifier`
  (`mlflow.sklearn`, input example + signature included);
* the newly trained version is set as the `production` **alias** and tagged
  `stage=production` (the version bump on retrain is visible in the Registry UI).

```
v1 initial  ───────────────────────────── version bump ───────────────────────► vN
   registered model versions (alias "production" moves on every retrain)
```

A lightweight mirror ledger `data/stream/deployments.json` records each version's
training window, trigger reason and baseline `test_auc` so the monitor and dashboard do
not need to query MLflow for every row. `reports/model_versions.csv` summarises it.

---

## 4. Simulating the production stream (`src/simulate_stream.py`)

The CSV has no timestamps, so "time" is simulated:

1. the 150k rows are shuffled once (seed 42);
2. the first 40 000 rows form the **reference window** (`reference.csv`) - the history
   the first model is trained on and the *fixed* golden baseline for drift tests;
3. the remaining 110 000 rows are cut into **44 weekly batches of 2 500 rows** and
   replayed in order (`batch_0001.csv` … `batch_0044.csv`);
4. deliberate drift is injected into specific later batches (all parameters live in
   `src/config.py`, one episode per drift type):

| weeks | episode                | drift type        | injection                                                                     | why this is visible                                                                 |
|-------|------------------------|-------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 11-14 | `covariate_util_shift` | **data drift**    | `util' = util·1.6 + 0.15` (mean + variance shift)                             | util PSI ≈ 4.8 (> 0.2) on every shifted batch                                       |
| 21-24 | `category_dependents`  | **data drift**    | `NumberOfDependents` re-sampled from an altered pmf (0 dependents 60% -> 15%) | chi² p ≈ 0 (max share change ≈ 46 pp)                                               |
| 31-44 | `concept_util_flip`    | **concept drift** | rows with `util ≥ 0.6` (≈24% of a batch) relabel `y=0` with p=0.9             | realised AUC collapses ≈ 0.86 → ≈ 0.69-0.75 while **all feature tests stay silent** |

The episode parameters were **calibrated empirically**: the covariate shift was chosen so
its PSI is ~25x the 0.2 gate while AUC stays healthy (≈0.86, so it is a *pure* data-drift
episode), and the label-relationship flip was tested for several segment/flip
combinations to pick one whose per-batch AUC drop (≈0.13-0.17) is unmistakable without
making the batches degenerate.

Every batch CSV contains its ground-truth label column (possibly modified by the concept
episode - in that simulated world the labels really have changed). The monitor pretends
labels arrive `LABEL_LAG_BATCHES = 2` weeks late, exactly like a production default model
whose outcomes are only observed after a delay.

---

## 5. Monitoring methodology (`src/drift_detection.py`)

For each incoming batch the monitor (using the model currently registered as
`production`):

1. **Scores the batch** and stores the scores (the version that served the batch is
   remembered, so realised performance is always measured on the model that actually
   produced the predictions).
2. Runs an **Evidently `DataDriftPreset` report** comparing the batch against the
   reference window:
   * numeric features - **PSI test** (threshold 0.2, the retraining gate; Evidently
     reports the PSI value per feature),
   * categorical feature (`NumberOfDependents`) - **chi-squared test** (p < 0.01 gate),
   * the prediction column is part of the report, which yields
     **prediction-distribution drift** separately,
   * per-feature **KS tests** (scipy) are added as a supplement so the "KS for numeric,
     chi² for categorical" description is backed by real numbers in every batch record.
   The full report is saved as HTML per batch (`reports/html/batch_XXXX_report.html`).
3. When the batch's labels become available (2 weeks later) it computes **realised
   model performance** - AUC, PR-AUC, accuracy@0.5, log loss - from the *stored*
   scores, and records it on the same timeline row.

### Data drift vs concept drift vs prediction drift vs actual performance

These four things are deliberately tracked **separately**, and the simulated episodes
are designed so that each lights up only the right signal:

| signal                              | what it measures                                                                       | week 11-14                   | week 21-24                   | week 31+                                              |
|-------------------------------------|----------------------------------------------------------------------------------------|------------------------------|------------------------------|-------------------------------------------------------|
| feature PSI / KS / chi² (Evidently) | **data drift**: did the *inputs* move vs the reference?                                | util PSI ≈ 4.8 ✔            | deps chi² p≈0 ✔             | silent (max PSI 0.011-0.019) ✘                       |
| prediction PSI (Evidently)          | **prediction-distribution drift**: did the deployed model's score distribution change? | small (≈0.07)                | small                        | tiny (0.004-0.013) ✘                                 |
| realised AUC on delayed labels      | **actual model performance**                                                           | ≈ 0.86 (fine)                | ≈ 0.86 (fine)                | collapses to ≈ 0.69-0.75 ✔                           |
| diagnosis                           |                                                                                        | data drift, no concept drift | data drift, no concept drift | **concept drift**: P(y\|x) changed while P(x) did not |

Key take-aways (this is the point of the exercise):

* **Prediction-distribution drift ≠ model performance.** In weeks 31+ the model keeps
  producing almost the *same score distribution* (inputs didn't change) while being
  badly wrong - only realised AUC reveals it. Conversely, after a retrain the model's
  score distribution legitimately shifts without any quality loss.
* **Data drift ≠ concept drift.** Weeks 11-14 and 21-24 shift the inputs only; the
  learned input→label function is still correct (AUC stays ≈0.86) but the PSI/chi²
  gates fire. Weeks 31+ change the input→label function only; feature tests stay quiet
  while AUC collapses. Retraining on *data* drift refreshes the model's operating
  distribution; retraining on *concept* drift is what repairs a broken function.
* **Delayed labels are what separate the two.** Until the true labels for a batch
  arrive you cannot tell whether an input shift is harmless or harmful - which is why
  the monitor keeps both signal families and only *acts* on a combination of them.

---

## 6. Thresholds & retraining policy (`src/monitor.py`)

A batch is **flagged as "needs retraining"** when any of these fires:

1. PSI > **0.2** on any monitored numeric feature (Evidently `drift_detected`),
2. chi² p < **0.01** on the categorical feature **and** the max category-share change is
   ≥ **3 percentage points** (an effect-size gate - chi² with n ≥ 2500 is hypersensitive
   to tiny share moves),
3. prediction-distribution PSI > **0.2**,
4. **smoothed realised-AUC drop ≥ 0.05** - the mean drop of the last ≤ 3 label-released
   batches relative to each serving model's registered baseline AUC (single-batch AUC
   on ~100-positive batches is noisy; averaging over 3 released batches denoises it).

To avoid retrain storms, retraining only actually happens if at least `COOLDOWN_BATCHES
= 6` batches passed since the last retrain; flagged-but-suppressed batches are still
recorded. **Every batch is logged regardless of whether it crosses a threshold** -
`reports/timeline.csv` and `reports/batch_details/batch_XXXX_detail.json` keep the full
drift scores (per-feature PSI, chi² p-values, KS stats, prediction PSI), the decision
and the realised performance, so the timeline charts show the entire history, not only
the flagged points.

---

## 7. The scheduled workflow (`monitor.yml`)

`.github/workflows/monitor.yml` encodes the two-job pattern you asked for:

* **job `drift-check`** (scheduled cron, weekly, or manual dispatch) runs
  `python -m src.monitor --check-next` - the drift check against the *next* batch -
  and puts `needs_retrain: true|false` into the job output (parsed from the script's
  `NEEDS_RETRAIN=true|false` line);
* **job `retrain`** is conditional:
  `if: needs.drift-check.outputs.needs_retrain == 'true'` and runs
  `python -m src.monitor --retrain-flagged`, which trains on the updated data window,
  logs the run to MLflow and registers the new `production` model version.

```yaml
drift-check ──(NEEDS_RETRAIN=true?)──► retrain ──► new production version in the registry
```

`scripts/run_scheduled_local.sh` reproduces exactly this loop locally (44 iterations:
`--check-next`, then `--retrain-flagged` only when flagged). One caveat for real GitHub
hosted runners: they are ephemeral, so the monitoring state (`data/stream/*`), the MLflow
SQLite DB and the batch files must be persisted between runs (checked in, uploaded as an
artifact, or - the typical setup - run on a self-hosted runner with a persistent
workspace and a shared model store). Locally everything is just files, which is what the
demo uses.

---

## 8. What actually happened (canonical run)

> Numbers below come from the actual run of `scripts/run_scheduled_local.sh`
> (deterministic seed; reproduce with the commands in §2).

| version | registered | trigger (batch)                                   | trained on               | holdout AUC |
|---------|------------|---------------------------------------------------|--------------------------|-------------|
| v1      | initial    | -                                                 | reference (40 000 rows)  | 0.8615      |
| v2      | week 11    | util PSI 4.81 > 0.2 (data drift)                  | reference + batches 1..9 | 0.8557      |
| v3      | week 21    | deps chi² p≈0, 46% share change (data drift)      | rolling batches 8..19    | 0.8417      |
| v4      | week 34    | smoothed AUC drop ≥ 0.05 (concept drift)          | rolling batches 21..32   | 0.8499      |
| v5      | week 40    | smoothed AUC drop ≥ 0.05 persists (concept drift) | rolling batches 27..38   | 0.8024      |

Timeline behaviour (full numbers in `reports/timeline.csv`):

* **weeks 1-10** clean: no flags, realised AUC bounces around the v1 baseline (0.86).
* **week 11** the utilisation covariate shift is detected immediately (PSI ≈ 4.8) ->
  v2 retrained. Weeks 12-14 stay flagged but are suppressed by the cooldown; AUC stays
  healthy the whole time (data drift without performance loss).
* **week 21** the dependents category-frequency drift fires the chi² gate -> v3.
  Weeks 22-24 suppressed; weeks 25-30 quiet.
* **weeks 31+** concept drift is *invisible* to every feature test (max PSI
  0.011-0.019, far under the 0.2 gate) yet realised AUC of the batches served by v3/v4
  drops to **0.69-0.77**. Labels of week 31 arrive at week 33, so the smoothed drop
  crosses the 0.05 margin at **week 34** (detection latency = label delay + smoothing)
  -> v4. The regime persists, so at **week 40** -> v5, whose training window finally
  contains a majority of new-regime weeks. Batches 41-44 (served by v5) recover to
  AUC ≈ 0.75-0.81, and the end-of-window label flush confirms the recovery.
* 4 retrains total; `production` alias ends on version 5.

Figures (`reports/figures/`):

| figure               | shows                                                                                                                                                                     |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `timeline_auc.png`   | realised AUC per batch coloured by serving version, episode shading, retrain markers - the concept-drift collapse and recovery                                            |
| `timeline_drift.png` | top: max PSI / util PSI / prediction PSI vs the 0.2 gate (log axis); bottom: NumberOfDependents chi² p-values - the *silence* of feature tests during the concept episode |

---

## 9. HTML reports + Streamlit UI (`app.py`)

Evidently generates a self-contained HTML drift report per batch
(`reports/html/batch_XXXX_report.html`; regenerate all with
`python -m src.monitor --generate-html`). `app.py` wraps everything in a Streamlit app:

* a **timeline** tab: realised AUC, feature/prediction PSI, categorical chi², episode
  shading and retrain markers on one scrollable set of charts (Plotly);
* a **batch inspector**: scrub through the 44 weeks with a slider - per-batch KPI cards
  (serving version, decision, PSI/chi²/prediction drift, realised AUC & drop), the
  per-feature Evidently table, KS supplements, and the embedded **Evidently HTML
  report** (generated on demand if missing);
* a **model registry** tab: version ledger, and confirmation of which version holds the
  `production` alias;
* an **About** tab explaining the drift-type distinctions from §5.

```bash
python -m streamlit run app.py
```

---

## 10. Reproducing / extending

Everything is deterministic (fixed seeds) except wall-clock time. To re-run from
scratch: delete `data/stream/ mlflow/ reports/` and follow §2. To change the drift
episodes or thresholds, edit `src/config.py` (all knobs are there and quoted in this
README). The monitoring loop itself is generic: point `src/simulate_stream.py` at any
tabular dataset, adapt `src/config.py`, and the Evidently/MLflow/monitor stack runs
unchanged.
