# Cross-chemistry battery health estimation

Code accompanying **Cross-chemistry battery health estimation with limited labels through conditional ensemble transfer learning**, by Tianren Zhao, Yanhui Zhang, Yuekuan Zhou, and Minghao Wang.

The manuscript pipeline screens 39 features using source batteries, retains 20 inputs, and evaluates four predictors: CNN–BiLSTM–Attention, feature-gated MLP, window-based histogram gradient boosting (HGB), and window-based Ridge regression. Target validation batteries select predictor subsets and global or condition-specific fusion weights. Separate matched-budget experiments compare source-pretrained and randomly initialized neural models.

## 1. Release contents

| Files                                                        | Purpose                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `01_raw_data_processing_V2_2_FINAL.py`                       | Parse and harmonize the four battery cohorts; screen cycles; resample curves; export manifests and per-battery records. |
| `02_feature_processing_V2_0_FINAL.py`                        | Construct and validate 39 candidate features.                |
| `03_cross_domain_feature_selection_V3_0_FINAL.py`            | Source-domain feature–SOH associations, trend consistency, and redundancy-constrained selection of 20 features. |
| `04_cross_domain_ensemble_transfer_learning_V4_0_2.py`       | Battery splits, source scaling and neural pretraining, target adaptation, HGB/Ridge fitting, validation-based branch/fusion selection, and evaluation. |
| `05_model_interpretability_V5_0_2_CONDITIONAL_FINAL.py`      | Prediction replay and validation-reference feature/time replacement sensitivity. This analysis is not SHAP or a causal attribution. |
| `06_ablation_studies_V6_0_FINAL.py`                          | Feature compression, initialization, and fusion comparisons. |
| `07_few_shot_cross_domain_transfer_V7_2_0_CONDITIONAL_FINAL.py` | Zero-label inference and paired target-label-budget experiments. |
| `figures/figure1_*.py` through `figure7_*.py`                | Data-derived figures; shared plotting utilities are in `figures/plotting_common.py`. |
| `08_audit_existing_run.py`, `09_repeat_source_pretraining.py`, `10_summarize_repeats.py` | Optional saved-run audit and independent source-pretraining repeats; not part of the manuscript's original seven-stage results. |
| `joule_extension.py`, `joule_paths.py`                       | Optional-experiment integrity helpers and portable defaults. |
| `test_joule_extension.py`, `tests/`                          | Synthetic checks and repository/runner tests.                |
| `SOURCE_PROVENANCE.json`, `SHA256SUMS.txt`                   | Source lineage and release checksums.                        |

Original stage filenames and internal model identifiers are retained because downstream loaders, checkpoints, and report schemas refer to them. Comments and docstrings were removed from Python files; scientific configuration and calculations were retained. The runner was corrected to use the supplied stage filenames. The old, superseded README and cached bytecode are not included.

## 2. Installation

Recommended starting environment: Python 3.10 or 3.11, with a separate virtual environment. These are portability recommendations, not a reconstruction of the original authors' frozen environment.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install:

```bash
python -m pip install -r requirements.txt
```

The requirements specify compatibility ranges, not an experimentally verified lockfile. For CUDA, install the appropriate PyTorch build for the local driver. Training selects CUDA when available and otherwise uses CPU; CPU execution may be slow. SHAP is not required by this implementation. Memory and runtime depend on dataset coverage and device; this release has no measured hardware/runtime minimum. Before publication, archive the exact environment of the manuscript run, for example with `python -m pip freeze > environment-manuscript.txt` from that environment, not an unrelated machine.

## 3. Data preparation

Raw third-party datasets, trained checkpoints, and the manuscript's original run directories are **not included**. Obtain the datasets from the original publications and observe their access and redistribution terms:

| Cohort  | Original publication                                         | Required directory below the data root                       |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| XJTU    | Wang et al. (2024), *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*. [DOI](https://doi.org/10.1038/s41467-024-48779-z) | `Battery Dataset/Batch-1/` through `Battery Dataset/Batch-6/`, containing `.mat` battery files |
| HUST    | Ma et al. (2022), *Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning*. [DOI](https://doi.org/10.1039/d2ee01676a) | `our_data/`, containing `.pkl` files                         |
| TJU-NCM | Zhu et al. (2022), *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation*. [DOI](https://doi.org/10.1038/s41467-022-29837-w) | `TJU/Dataset_2_NCM_battery/`, containing the original CSV records |
| TJU-NCA | Same TJU publication                                         | `TJU/Dataset_1_NCA_battery/`, containing the original CSV records |

The default data root is `workspace/` beside `run_all.py`. Place the above directories there, or pass an existing data root with `--base-path`. Generated runs are written below that same root. Do not rename original dataset files or flatten their internal directory structure. XJTU temperature-compensation files are not battery cycling inputs. Load pickle/checkpoint files only from trusted sources.

The source cohorts are XJTU and HUST; the main target cohorts are TJU-NCM and TJU-NCA. The original dataset inventory is 253 cells; four short-history XJTU cells are excluded from sequence modeling. The historical split counts are:

| Cohort  | Source train | Source validation | Target adaptation | Target validation | Target test | Short-history exclusion |
| ------- | -----------: | ----------------: | ----------------: | ----------------: | ----------: | ----------------------: |
| XJTU    |           41 |                10 |                 — |                 — |           — |                       4 |
| HUST    |           62 |                15 |                 — |                 — |           — |                       0 |
| TJU-NCM |            — |                 — |                33 |                11 |          11 |                       0 |
| TJU-NCA |            — |                 — |                39 |                13 |          14 |                       0 |

These counts alone do not establish identical cell assignments. Compare generated manifests with the exact original `battery_split_manifest.csv` when reproducing a particular result.

## 4. Run manuscript experiments

From the repository root, with datasets in `workspace/`:

```bash
python run_all.py
```

With datasets already stored elsewhere:

```bash
python run_all.py --base-path "D:/battery_data"
```

The base path is the parent of `Battery Dataset`, `our_data`, and `TJU`, not the code directory or a timestamped result directory. No edits to Python source are needed. The runner stops on a nonzero subprocess exit and does not delete previous runs. It records package versions and newly created run paths in `execution_logs/` below the data root.

Run through the main models first:

```bash
python run_all.py --base-path "D:/battery_data" --through-stage 4 --skip-figures
```

Continue interpretation, ablations, and budget comparisons with an exact accepted Stage 4 run:

```bash
python run_all.py --base-path "D:/battery_data" --from-stage 5 --stage4-run "D:/battery_data/SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION/run_YYYYMMDD_HHMMSS_microseconds" --skip-figures
```

Replace the example run name with a real directory. Check each stage's validation reports. A `FAIL` report must be resolved before interpreting later results, even if a script returns exit status zero. A successful process exit alone is not scientific validation. A clean data root is recommended for a new full run: stage discovery can otherwise select previously accepted artifacts. The new runner passes a newly generated Stage 4 run explicitly to subsequent consumers, but stages 2–4 still use their original accepted-run discovery logic.

## 5. Outputs and figures

| Stage | Output parent below the data root                  |
| ----- | -------------------------------------------------- |
| 1     | `DATAPROCESSING_V2_2/`                             |
| 2     | `FEATUREPROCESSING_V2_0/`                          |
| 3     | `FEATURESELECTION_V3_0_1/`                         |
| 4     | `SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION/` |
| 5     | `INTERPRETABILITY_V5_0_2_CONDITIONAL/`             |
| 6     | `ABLATION_RESULTS_V6_0_CONDITIONAL/`               |
| 7     | `CROSS_DOMAIN_RESULTS_V7_2_0_CONDITIONAL/`         |

Each stage stores timestamped runs and relevant configurations, tables, manifests, checkpoints, predictions, and/or validation reports. Keep the full run and its recorded upstream dependencies together. Default discovery is for convenience, not a guarantee that artifacts from different historical executions belong together.

Draw all figures after completing the seven stages:

```bash
python run_all.py --base-path "D:/battery_data" --figures-only
```

To select an exact input for a figure:

```bash
python figures/figure4_ensemble_transfer_performance.py --base-path "D:/battery_data" --run "D:/battery_data/SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION/run_YYYYMMDD_HHMMSS_microseconds" --output-dir "D:/battery_figures"
```

All seven figure scripts support `--help`, `--base-path`, `--run`, and `--output-dir`. By default they discover accepted results; explicit `--run` is preferable for publication figures. The conceptual workflow and graphical abstract were separately prepared artwork and are not generated by these data-plotting scripts.

## 6. Protocol and metric interpretation

- Main experiment: XJTU+HUST source training and adaptation to each TJU target cohort. CNN/MLP transfer learned parameters; HGB/Ridge are fitted on target adaptation batteries.
- Budget experiments: XJTU+HUST → TJU-NCM and TJU-NCM → TJU-NCA. Source preprocessing is shared between pretrained and random-initialization arms; the contrast isolates neural initialization, not a wholly target-only alternative pipeline.
- Temporal inputs have 20 observed cycles. The main source split is 80%/20%; target adaptation/validation/test ratios are 60%/20%/20%. The main seed is 42; Stage 7 uses acquisition/adaptation seeds 42, 52, and 62 and requested fractions 0, 0.1, 0.2, 0.4, 0.6, and 0.8.
- Label budgets count acquired target batteries, including adaptation and calibration batteries, relative to all eligible target batteries. The requested 10% fraction corresponds to 6/55 TJU-NCM cells and 7/66 TJU-NCA cells. It is not 10% of cycle records.
- Pooled-window RMSE pools prediction windows; battery-macro RMSE is the arithmetic mean of per-battery RMSEs. SOH is on a 0–1 scale. Do not equate RMSE 0.00443 with a 0.00443% error.
- Stage 7 zero-label inference is reported separately. Each original transfer task uses one source pretraining run; its target seeds do not measure independent source-pretraining variability.

The manuscript reports pooled-window RMSE 0.00443/0.00799 and mean per-battery RMSE 0.00366/0.00980 for TJU-NCM/TJU-NCA. At the requested 10% budgets, the reported initialization-related reductions are 38.2% and 27.8% for the two specified transfer tasks. These are reference values from the manuscript, not outcomes verified by executing this release. Fresh runs may differ with dependencies, hardware, source data, and random execution. Do not select the best test run to match these values.

The benchmark is retrospective: its fixed test split informed iterative method development. Fusion validation is not fully nested validation of the entire pipeline. Original bootstrap intervals condition on fixed test batteries and do not include independent source-pretraining repeats or multiplicity adjustment. The features require completed charging/discharging records; this release does not establish online deployment, diagnostic-free operation, or measured calibration-time savings.

## 7. Optional source-pretraining repeats

These scripts are supplied separately from the default execution path because their results were not added to the current manuscript. They need the original accepted Stage 4 run and its exact Stage 3 feature artifacts. Stage 8 audits saved assignments/configuration; it does not prove historical test independence or audit every raw window.

```bash
python 08_audit_existing_run.py --base-path "D:/battery_data" --stage4-run "D:/battery_data/SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION/run_YYYYMMDD_HHMMSS_microseconds"
python 09_repeat_source_pretraining.py --base-path "D:/battery_data" --stage4-run "D:/battery_data/SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION/run_YYYYMMDD_HHMMSS_microseconds" --output "D:/battery_repeats" --plan-only
```

If the plan is correct, repeat the second command without `--plan-only`, then summarize:

```bash
python 10_summarize_repeats.py --root "D:/battery_repeats"
```

Default source seeds are 101, 211, 307, 401, and 503, crossed with target seeds 42, 52, and 62. The default five nonzero fractions produce 150 matched task/budget/seed cells. The summary reports all runs and conditional source-repeat intervals; it does not choose a best run. Resuming a directory after changing source hashes or protocol is rejected: choose a fresh output directory instead.

`CRPS_BASE_PATH`, `CRPS_STAGE4_RUN`, and `CRPS_REPEAT_OUTPUT` can supply optional defaults. A Stage 4 run must be explicitly supplied for scripts 08/09 unless `CRPS_STAGE4_RUN` is set. The helper relocates the exact recorded Stage 3 run by name; it will not silently substitute a different latest run. Other historical checkpoint paths may require restoration of their original dependency tree.

## 8. Tests and reproducibility status

```bash
python -m unittest discover -s tests -v
python -m unittest test_joule_extension.py -v
python -m compileall -q .
```

The tests use synthetic fixtures to check repository consistency, metric calculations, run selection, runner command construction, and optional-experiment helpers. They do not establish accuracy on real battery datasets. Packaging checks were performed without PyTorch or the raw datasets; a full end-to-end training rerun has not been completed in the packaging environment. See `VALIDATION.md` for the checks actually performed.

For an archival release linked to a manuscript, include the exact original split manifest, selected-feature list, run configurations, dependency versions, and the prediction tables used for reported metrics where redistribution permits. Original checkpoints and their source scaler are additionally needed for exact prediction replay. These artifacts are not present in the supplied code archives and are not invented by this release.

## 9. GitHub publication

Extract this directory and upload its **contents**, including the Python files, `figures/`, `tests/`, README, requirements, provenance, and validation record. Do not upload raw third-party datasets, personal paths, credentials, temporary environments, or cached bytecode. `.gitignore` excludes common generated artifacts; review any files stored outside `workspace/` before adding them.

No software license has been selected on the authors' behalf. Before making the repository public, confirm code ownership and choose an appropriate license with all relevant rights holders. Public visibility alone does not grant an open-source license. The manuscript has no verified DOI or GitHub repository URL in this release; add these when available rather than inserting a fabricated citation. A versioned GitHub release and an archival DOI can then identify the exact code version used for the submission.

## Contact and citation

Tianren Zhao, Yanhui Zhang, Yuekuan Zhou, and Minghao Wang. *Cross-chemistry battery health estimation with limited labels through conditional ensemble transfer learning*. Manuscript accompanying this repository.

For scientific questions, use the corresponding-author contact details in the submitted manuscript. Please also cite the original dataset publications when using their data.
