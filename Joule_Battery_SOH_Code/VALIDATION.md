# Release validation

Packaging date: 16 November 2026.

| Check | Result | Scope |
| --- | --- | --- |
| Python static compilation | PASS | All 23 supplied Python files compile. |
| Repository tests | 8/8 PASS | Script registry, comment/docstring removal, personal-path removal, runner dispatch/failure logging, and pooled/macro metric definitions. |
| Optional-experiment tests | 9/9 PASS | Synthetic run audit, exact Stage 3 relocation, locked protocol, duplicate manifests, paired metrics/fusion, repeat summary, source-seed isolation, and plan-only execution without PyTorch. |
| Manuscript implementation comparison | PASS | AST comparison of the 15 analysis and plotting files from the seven-stage archive, after stripping docstrings and changing only the default data path, matches the release. Scientific expressions, model definitions, hyperparameters, and split logic are retained. |
| Command-line checks | PASS | Help for the runner and optional repeat script; missing-input rejection before training. |
| Full dataset training | NOT RUN | Raw battery datasets and PyTorch are absent from the packaging environment. |
| Manuscript metric reproduction | NOT VERIFIED | Original prediction/checkpoint/configuration artifacts are not contained in the supplied code archives. |

Runner modifications correct outdated stage filenames, provide a repository-relative default data root, record the local execution environment and run paths, propagate selected Stage 4 runs, and limit partial-run figures to the requested stages. Optional scripts 08/09 now require an explicit Stage 4 run or environment setting instead of an author's machine-specific path. Tests deliberately mock heavy subprocesses where indicated; their success is not a training benchmark.

The original build/schema identifiers are retained for checkpoint/report compatibility. Release source hashes differ after comment removal and path cleanup; existing hash-locked optional experiments must not be resumed with modified code. Keep their original source snapshots or use a new output directory.

Before claiming exact reproducibility, execute the package against the original data in the actual research environment, compare cell assignments and selected features, retain all validation reports, and recompute tables from the archived predictions. No results or missing artifacts were fabricated during packaging.
