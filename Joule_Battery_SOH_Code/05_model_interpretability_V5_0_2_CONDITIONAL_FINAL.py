from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import random
import tempfile
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEMA_VERSION = "5.0.2"
SCRIPT_BUILD = "STAGE5_V5_0_2_CONDITIONAL_REPLAY_20261006"
IMPLEMENTATION_REVISION = "STAGE5_V5_0_2_PANDAS_SAFE_R2_20261006"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
STAGE4_SCRIPT_NAME = (
    "04_cross_domain_ensemble_transfer_learning_"
    "V4_0_2_CONDITIONAL_CONTRIBUTION.py"
)
TARGETS = ("TJU_NCM", "TJU_NCA")
NEURAL = ("CNN_BiLSTM_Attention", "MLP_Feature_Attention")
TABULAR = ("HGB_Window_SOH", "Ridge_Window_SOH")
MODELS = NEURAL + TABULAR
SEQUENCE_LENGTH = 20
TEST_WINDOWS_PER_BATTERY = 6
BACKGROUND_WINDOWS_PER_EXPLANATION = 8
BACKGROUND_WINDOWS_PER_BATTERY = 3
BOOTSTRAPS = 500
PREDICTION_BATCH_SIZE = 512
PREDICTION_ATOL = 2e-5
PREDICTION_RTOL = 1e-5
RANDOM_SEED = 42
BASE = Path(os.environ.get(
    "CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"
))
STAGE4_ROOT = BASE / "SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION"
OUTPUT_ROOT = BASE / "INTERPRETABILITY_V5_0_2_CONDITIONAL"


warnings.filterwarnings(
    "ignore", message=r"TypedStorage is deprecated\..*",
    category=UserWarning, module=r"torch\._utils",
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def stage4_build_in_source(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "SCRIPT_BUILD"
                   for t in node.targets):
                return str(ast.literal_eval(node.value))
    return ""


def import_stage4() -> Any:
    here = Path(__file__).resolve().parent
    explicit = os.environ.get("CRPS_STAGE4_SCRIPT")
    candidates = ([Path(explicit)] if explicit else []) + [
        here / STAGE4_SCRIPT_NAME,
        *sorted(here.glob("04_cross_domain_ensemble_transfer_learning*.py")),
    ]
    seen: set[str] = set()
    for path in candidates:
        path = path.resolve()
        if str(path) in seen or not path.is_file():
            continue
        seen.add(str(path))
        if stage4_build_in_source(path) != STAGE4_BUILD:
            continue
        spec = importlib.util.spec_from_file_location("stage4_frozen_experiment", path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot import matching Stage 4 script: " + str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if tuple(module.MODEL_NAMES) != MODELS:
            raise ValueError("The matching Stage 4 model names/order differ")
        print("Stage 4 implementation:", path, flush=True)
        return module
    raise FileNotFoundError(
        "Stage 4 Python script with build " + STAGE4_BUILD +
        " was not found beside Stage 5. Set CRPS_STAGE4_SCRIPT if necessary."
    )


def accepted_stage4_run() -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    explicit = os.environ.get("CRPS_STAGE4_RUN")
    candidates = ([Path(explicit)] if explicit else
                  sorted(STAGE4_ROOT.glob("run_*"), reverse=True))
    failures: List[str] = []
    for run in candidates:
        try:
            report = run / "validation_report"
            config = read_json(report / "processing_config.json")
            summary = read_json(report / "processing_summary.json")
            if config.get("script_build") != STAGE4_BUILD:
                raise ValueError("wrong Stage 4 build")
            if config.get("schema_version") != "4.0.2":
                raise ValueError("wrong schema version")
            if summary.get("validation_status") != "PASS" or summary.get("fatal_checks"):
                raise ValueError("Stage 4 validation did not pass")
            if tuple(config.get("models", ())) != MODELS:
                raise ValueError("Stage 4 model order differs")
            if int(config.get("sequence_length", -1)) != SEQUENCE_LENGTH:
                raise ValueError("sequence length differs")
            required = [run / "source_standardizer.pkl",
                        run / "battery_split_manifest.csv",
                        run / "selected_features.txt"]
            for target in TARGETS:
                required += [run / "checkpoints" / "targets" / target /
                             "fusion_policy.json",
                             run / "predictions" / target / "test_predictions.csv",
                             run / "predictions" / target / "validation_predictions.csv"]
                for name in NEURAL:
                    required.append(run / "checkpoints" / "targets" / target /
                                    (name + "_best.pth"))
                for name in TABULAR:
                    required.append(run / "checkpoints" / "targets" / target /
                                    (name + "_best.pkl"))
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError("missing Stage 4 artifacts: " + str(missing[:5]))
            return run, config, summary
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            failures.append(str(run) + ": " + str(exc))
    raise FileNotFoundError("No matching accepted Stage 4 run.\n" +
                            "\n".join(failures[:10]))


def load_standardizer(stage4: Any, run: Path, names: Sequence[str]) -> Any:
    with (run / "source_standardizer.pkl").open("rb") as handle:
        info = pickle.load(handle)
    if (info.get("script_build") != STAGE4_BUILD or
            list(info.get("feature_names", [])) != list(names) or
            not info.get("fitted_on_source_train_only")):
        raise ValueError("Stage 4 standardizer lineage or feature order differs")
    return stage4.FrozenStandardizer(
        info["mean"], info["scale"], names,
        info["fitted_datasets"], info["fitted_batteries"],
    )


def split_batteries(manifest: pd.DataFrame, target: str, split: str) -> set[str]:
    return set(manifest.loc[
        manifest["dataset"].eq(target) & manifest["split"].eq(split),
        "battery_id",
    ].astype(str))


def load_policy(run: Path, target: str, stage4: Any) -> Dict[str, Any]:
    path = run / "checkpoints" / "targets" / target / "fusion_policy.json"
    policy = read_json(path)
    if (policy.get("script_build") != STAGE4_BUILD or
            policy.get("target") != target or
            bool(policy.get("test_labels_used"))):
        raise ValueError(target + " fusion policy has invalid lineage")
    selected = list(policy.get("selected_models", ()))
    if (not selected or any(name not in MODELS for name in selected) or
            len(selected) > 3):
        raise ValueError(target + " fusion model subset is invalid")
    dummy = pd.DataFrame({"condition_group": ["UNSEEN"]})
    stage4.fusion_row_weights(dummy, policy)
    return policy


def load_models(stage4: Any, run: Path, target: str) -> Dict[str, Any]:
    folder = run / "checkpoints" / "targets" / target
    models: Dict[str, Any] = {}
    for name in NEURAL:
        path = folder / (name + "_best.pth")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if (payload.get("script_build") != STAGE4_BUILD or
                payload.get("model_name") != name):
            raise ValueError(str(path) + " is not the expected checkpoint")
        model = stage4.build_model(name)
        model.load_state_dict(payload["state_dict"], strict=True)
        models[name] = model.to(stage4.DEVICE).eval()
    expected_columns = ["f_%s_%02d" % (kind, i)
                        for kind in ("current", "mean5", "change20")
                        for i in range(stage4.FEATURE_COUNT)]
    for name in TABULAR:
        path = folder / (name + "_best.pkl")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if (payload.get("script_build") != STAGE4_BUILD or
                payload.get("model_name") != name or
                list(payload.get("feature_columns", [])) != expected_columns or
                bool(payload.get("test_labels_used"))):
            raise ValueError(str(path) + " has incompatible tabular features")
        models[name] = payload["regressor"]
    return models


def replay_predictions(stage4: Any, dataset: Any,
                       models: Mapping[str, Any], policy: Mapping[str, Any]) -> pd.DataFrame:
    branch_frames = {name: stage4.predict_model(models[name], dataset)
                     for name in NEURAL}
    for name in TABULAR:
        branch_frames[name] = stage4.tabular_prediction_frame(
            branch_frames[NEURAL[0]], dataset, models[name]
        )
    merged = stage4.merge_model_predictions(branch_frames)
    return stage4.add_ensemble_prediction(merged, policy)


def compare_replay(stored: pd.DataFrame, replay: pd.DataFrame,
                   target: str, split: str) -> List[Dict[str, Any]]:
    keys = ["dataset", "battery_id", "window_index", "condition_group",
            "history_start_cycle", "cycle", "history_length"]
    cols = ["y_true", *MODELS, "Adaptive_Ensemble",
            *("fusion_weight_" + name for name in MODELS)]
    if (any(col not in stored or col not in replay for col in keys + cols) or
            stored.duplicated(keys).any() or replay.duplicated(keys).any()):
        raise ValueError(target + " " + split + " prediction columns/keys differ")
    a = stored.sort_values(keys, kind="stable").reset_index(drop=True)
    b = replay.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(a) != len(b) or not a[keys].equals(b[keys]):
        raise ValueError(target + " " + split + " prediction keys do not align")
    result = []
    for col in cols:
        left, right = a[col].to_numpy(float), b[col].to_numpy(float)
        diff = np.abs(left - right)
        maximum = float(diff.max()) if len(diff) else 0.0
        atol = (1e-6 if col == "y_true" or col.startswith("fusion_weight_")
                else PREDICTION_ATOL)
        rtol = 0.0 if atol == 1e-6 else PREDICTION_RTOL
        passed = bool(np.isfinite(diff).all() and
                      np.allclose(left, right, rtol=rtol, atol=atol))
        result.append({"target": target, "split": split, "column": col,
                       "window_count": len(a), "maximum_absolute_error": maximum,
                       "atol": atol, "rtol": rtol, "check_pass": passed})
    failed = [row for row in result if not row["check_pass"]]
    if failed:
        raise ValueError(target + " " + split +
                         " prediction replay failed: " + str(failed))
    return result


def choose_battery_balanced_indices(dataset: Any, per_battery: int) -> List[int]:
    by_battery: Dict[str, List[int]] = {}
    for idx, (record_index, _) in enumerate(dataset.index_map):
        battery = str(dataset.records[record_index]["battery_id"])
        by_battery.setdefault(battery, []).append(idx)
    chosen: List[int] = []
    for battery in sorted(by_battery):
        choices = by_battery[battery]
        offsets = np.linspace(0, len(choices) - 1,
                              num=min(per_battery, len(choices)))
        chosen.extend(choices[int(round(k))] for k in offsets)
    return sorted(set(chosen))


def window_at(dataset: Any, index: int) -> np.ndarray:
    rec_index, end_index = dataset.index_map[index]
    start = end_index - dataset.sequence_length + 1
    return np.asarray(dataset.records[rec_index]["windows"][start],
                      dtype=np.float32)


def background_pool(dataset: Any) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    pool: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for index in choose_battery_balanced_indices(
            dataset, BACKGROUND_WINDOWS_PER_BATTERY):
        rec_index, _ = dataset.index_map[index]
        record = dataset.records[rec_index]
        condition = str(record["condition_group"])
        pool.setdefault(condition, []).append(
            (str(record["battery_id"]), window_at(dataset, index))
        )
    return pool


def select_references(pool: Mapping[str, List[Tuple[str, np.ndarray]]],
                      condition: str, seed: int) -> np.ndarray:

    candidates = pool.get(condition, [])
    if len(candidates) < 4:
        candidates = [row for rows in pool.values() for row in rows]
    if not candidates:
        raise ValueError("Target validation batteries provide no background windows")
    rng = np.random.default_rng(seed)
    batteries = sorted(set(battery for battery, _ in candidates))
    chosen: List[np.ndarray] = []
    for slot in range(BACKGROUND_WINDOWS_PER_EXPLANATION):
        battery = batteries[(slot + seed) % len(batteries)]
        local = [value for label, value in candidates if label == battery]
        chosen.append(local[int(rng.integers(len(local)))])
    return np.stack(chosen, axis=0)


def tabular_summary(X: np.ndarray) -> np.ndarray:
    current = X[:, -1, :]
    mean5 = X[:, -5:, :].mean(axis=1)
    change20 = current - X[:, 0, :]
    return np.concatenate((current, mean5, change20), axis=1).astype(np.float32)


def predict_frozen_ensemble(
    X: np.ndarray, condition: str, models: Mapping[str, Any],
    policy: Mapping[str, Any], stage4: Any,
) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (SEQUENCE_LENGTH, stage4.FEATURE_COUNT):
        raise ValueError("Interpretation windows have wrong shape")
    weights = stage4.fusion_row_weights(
        pd.DataFrame({"condition_group": [condition]}), policy
    )[0]
    result = np.zeros(len(X), dtype=np.float64)
    for j, name in enumerate(MODELS):
        if weights[j] <= 0:
            continue
        if name in NEURAL:
            chunks: List[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(X), PREDICTION_BATCH_SIZE):
                    tensor = torch.from_numpy(X[start:start + PREDICTION_BATCH_SIZE])
                    tensor = tensor.to(stage4.DEVICE)
                    chunks.append(models[name](tensor).detach().cpu().numpy())
            predicted = np.concatenate(chunks).astype(np.float64)
        else:
            predicted = np.asarray(
                models[name].predict(tabular_summary(X)), dtype=np.float64
            )
        result += float(weights[j]) * predicted
    if not np.isfinite(result).all():
        raise ValueError("Non-finite prediction while explaining frozen ensemble")
    return result


def perturbation_sensitivity(
    sample: np.ndarray, background: np.ndarray, condition: str,
    original_prediction: float, models: Mapping[str, Any], policy: Mapping[str, Any],
    stage4: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_count = stage4.FEATURE_COUNT
    k = len(background)
    cases = np.broadcast_to(sample,
                            (feature_count + SEQUENCE_LENGTH, k,
                             SEQUENCE_LENGTH, feature_count)).copy()
    for feature in range(feature_count):
        cases[feature, :, :, feature] = background[:, :, feature]
    for t in range(SEQUENCE_LENGTH):
        cases[feature_count + t, :, t, :] = background[:, t, :]
    predictions = predict_frozen_ensemble(
        cases.reshape(-1, SEQUENCE_LENGTH, feature_count),
        condition, models, policy, stage4,
    ).reshape(feature_count + SEQUENCE_LENGTH, k)
    changes = float(original_prediction) - predictions.mean(axis=1)
    if not np.isfinite(changes).all():
        raise ValueError("Non-finite replacement sensitivity")
    return changes[:feature_count], changes[feature_count:]


def summarize_sensitivity(
    samples: pd.DataFrame, prefix: str, names: Sequence[str], target: str,
) -> pd.DataFrame:
    columns = [prefix + str(i) for i in range(len(names))]
    if samples.empty or any(column not in samples for column in columns):
        raise ValueError("Missing sensitivity samples or component columns")


    battery_means: List[np.ndarray] = []
    for _, group in samples.groupby("battery_id", sort=True):
        block = group.loc[:, columns].to_numpy(dtype=np.float64)
        if block.size == 0 or not np.isfinite(block).all():
            raise ValueError("A battery has missing/non-finite sensitivity values")
        battery_means.append(np.mean(np.abs(block), axis=0))
    values = np.vstack(battery_means)
    estimate = values.mean(axis=0)
    rng = np.random.default_rng(RANDOM_SEED + (1 if target == TARGETS[1] else 0))
    picks = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    bootstrap = values[picks].mean(axis=1)
    denom = float(estimate.sum())
    if denom <= 0:
        raise ValueError("All explanation sensitivities were zero")
    normal = estimate / denom
    bootstrap = bootstrap / np.maximum(bootstrap.sum(axis=1, keepdims=True), 1e-12)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return pd.DataFrame({
        "target": target,
        "component": list(names),
        "mean_absolute_prediction_change": estimate,
        "normalized_sensitivity": normal,
        "normalized_ci95_lower": lower,
        "normalized_ci95_upper": upper,
        "battery_count": len(values),
        "explained_window_count": len(samples),
        "interpretation": "validation-reference replacement sensitivity, not SHAP",
    })


def explain_target(
    target: str, test_dataset: Any, validation_dataset: Any,
    replay: pd.DataFrame, names: Sequence[str], models: Mapping[str, Any],
    policy: Mapping[str, Any], stage4: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = background_pool(validation_dataset)
    by_index = replay.set_index("window_index", verify_integrity=True)
    sample_rows: List[Dict[str, Any]] = []
    chosen = choose_battery_balanced_indices(test_dataset, TEST_WINDOWS_PER_BATTERY)
    print(target, "test windows to explain:", len(chosen),
          "validation reference windows:", sum(map(len, pool.values())), flush=True)
    for position, index in enumerate(chosen, 1):
        metadata = test_dataset.metadata(index)
        condition = str(metadata["condition_group"])
        original = float(by_index.loc[index, "Adaptive_Ensemble"])
        background = select_references(pool, condition, RANDOM_SEED + index)
        feature, time = perturbation_sensitivity(
            window_at(test_dataset, index), background, condition,
            original, models, policy, stage4,
        )
        sample_rows.append({
            "target": target, "battery_id": metadata["battery_id"],
            "cycle": metadata["cycle"], "window_index": index,
            "condition_group": condition, "ensemble_prediction": original,
            "reference_count": len(background),
            **{"feature_change_%d" % k: float(value)
               for k, value in enumerate(feature)},
            **{"time_change_%d" % k: float(value)
               for k, value in enumerate(time)},
        })
        if position == 1 or position % 20 == 0 or position == len(chosen):
            print(target, "explanation", position, "/", len(chosen), flush=True)
    sample_frame = pd.DataFrame(sample_rows)
    feature_frame = summarize_sensitivity(sample_frame, "feature_change_", names, target)
    time_frame = summarize_sensitivity(
        sample_frame, "time_change_", [str(k - SEQUENCE_LENGTH + 1)
                                        for k in range(SEQUENCE_LENGTH)], target
    )
    return sample_frame, feature_frame, time_frame


def descriptive_branch_removal(test: pd.DataFrame, policy: Mapping[str, Any],
                               stage4: Any, target: str) -> pd.DataFrame:
    selected = list(policy["selected_models"])
    if len(selected) < 2:
        return pd.DataFrame()
    all_weights = stage4.fusion_row_weights(test, policy)
    all_predictions = test[list(MODELS)].to_numpy(float)
    baseline = stage4.fusion_rmse_scores(
        test, test["Adaptive_Ensemble"].to_numpy(float)
    )
    rows: List[Dict[str, Any]] = []
    for dropped in selected:
        index = MODELS.index(dropped)
        reduced = all_weights.copy()
        reduced[:, index] = 0.0
        mass = reduced.sum(axis=1)


        if (mass <= 1e-8).any():
            rows.append({"target": target, "dropped_branch": dropped,
                         "status": "undefined_zero_remaining_weight",
                         "test_labels_used_for_model_selection": False})
            continue
        reduced = reduced / mass[:, None]
        counterfactual = (all_predictions * reduced).sum(axis=1)
        scope_scores = stage4.fusion_rmse_scores(test, counterfactual)
        for scope in ("pooled_cycles", "battery_macro"):
            rows.append({
                "target": target, "dropped_branch": dropped,
                "evaluation_scope": scope,
                "frozen_ensemble_RMSE": baseline[scope],
                "without_branch_RMSE": scope_scores[scope],
                "RMSE_increase_when_branch_removed": scope_scores[scope] - baseline[scope],
                "status": "descriptive_test_evaluation_only",
                "test_labels_used_for_model_selection": False,
            })
    return pd.DataFrame(rows)


def draw_figure(feature: pd.DataFrame, time: pd.DataFrame,
                policies: Mapping[str, Mapping[str, Any]],
                feature_names: Sequence[str], output: Path) -> None:
    colors = {"TJU_NCM": "#457B9D", "TJU_NCA": "#E1A156"}
    branch_colors = {
        "CNN_BiLSTM_Attention": "#457B9D",
        "MLP_Feature_Attention": "#AC8DAF",
        "HGB_Window_SOH": "#69A69A",
        "Ridge_Window_SOH": "#E1A156",
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 11), constrained_layout=True)
    for j, target in enumerate(TARGETS):
        frame = feature.loc[feature["target"].eq(target)].nlargest(
            12, "normalized_sensitivity"
        ).sort_values("normalized_sensitivity")
        ax = axes[0, j]
        y = np.arange(len(frame))
        ax.barh(y, frame["normalized_sensitivity"], color=colors[target])
        ax.set_yticks(y, frame["component"], fontsize=6.6)
        ax.set_xlabel("Normalized absolute output sensitivity")
        ax.set_title(target + " | highest feature sensitivities")

    ax = axes[1, 0]
    pivot = feature.pivot(index="component", columns="target",
                          values="normalized_sensitivity")
    top = pivot.mean(axis=1).nlargest(12).index
    pivot = pivot.loc[top].iloc[::-1]
    y = np.arange(len(pivot))
    for i in range(len(pivot)):
        ax.plot([pivot.iloc[i, 0], pivot.iloc[i, 1]], [y[i], y[i]],
                color="#A9AFB5", lw=1)
    for target in TARGETS:
        ax.scatter(pivot[target], y, color=colors[target], s=16,
                   label=target, zorder=3)
    ax.set_yticks(y, pivot.index, fontsize=6.6)
    ax.set_xlabel("Normalized absolute output sensitivity")
    ax.set_title("Cross-target feature comparison")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1, 1]
    for target in TARGETS:
        frame = time.loc[time["target"].eq(target)].copy()
        frame["offset"] = frame["component"].astype(int)
        frame = frame.sort_values("offset")
        ax.plot(frame["offset"], frame["normalized_sensitivity"],
                marker="o", markersize=2.3, lw=1.4,
                color=colors[target], label=target)
    ax.set_xticks([-19, -15, -10, -5, 0])
    ax.set_xlabel("Cycles relative to current cycle")
    ax.set_ylabel("Normalized absolute output sensitivity")
    ax.set_title("Temporal replacement sensitivity")
    ax.legend(loc="upper left", frameon=False)

    for j, target in enumerate(TARGETS):
        ax = axes[2, j]
        policy = policies[target]
        groups = [("Global", np.asarray(policy["global_weights"], dtype=float))]
        groups.extend((condition, np.asarray(weight, dtype=float))
                      for condition, weight in sorted(policy["condition_weights"].items()))
        bottom = np.zeros(len(groups))
        for i, model in enumerate(MODELS):
            values = np.asarray([group[1][i] for group in groups])
            ax.bar(np.arange(len(groups)), values, bottom=bottom,
                   color=branch_colors[model], label=model.replace("_", " "),
                   width=0.6)
            bottom += values
        ax.set_xticks(np.arange(len(groups)), [name for name, _ in groups],
                      rotation=20, ha="right")
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Frozen fusion weight")
        ax.set_title(target + " | validation-learned fusion weights")
        ax.legend(loc="upper left", fontsize=6, frameon=False, ncol=2)

    for label, ax in zip("ABCDEF", axes.flat):
        ax.text(-0.16, 1.11, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / ("Figure5_conditional_ensemble_sensitivity." + extension),
                    dpi=300)
    plt.close(fig)


def preflight_compatibility() -> None:
    rows: List[Dict[str, Any]] = []
    for battery_index in range(2):
        for cycle_index in range(2):
            row: Dict[str, Any] = {"battery_id": str(battery_index)}
            for index in range(SEQUENCE_LENGTH):
                row["feature_change_%d" % index] = (
                    0.0001 * (index + 1) * (battery_index + cycle_index + 1)
                )
                row["time_change_%d" % index] = (
                    0.0002 * (SEQUENCE_LENGTH - index) * (battery_index + 1)
                )
            rows.append(row)
    synthetic = pd.DataFrame(rows)
    feature_names = ["feature_%d" % index for index in range(SEQUENCE_LENGTH)]
    feature = pd.concat([
        summarize_sensitivity(synthetic, "feature_change_", feature_names, target)
        for target in TARGETS
    ], ignore_index=True)
    temporal = pd.concat([
        summarize_sensitivity(
            synthetic, "time_change_",
            [str(index - SEQUENCE_LENGTH + 1) for index in range(SEQUENCE_LENGTH)],
            target,
        ) for target in TARGETS
    ], ignore_index=True)
    if not np.isclose(feature.loc[feature["target"].eq(TARGETS[0]),
                                  "normalized_sensitivity"].sum(), 1.0):
        raise AssertionError("pandas sensitivity preflight did not normalize")
    policies = {target: {
        "global_weights": [0.6, 0.0, 0.4, 0.0],
        "condition_weights": {"25C_05C": [0.7, 0.0, 0.3, 0.0]}
        if target == TARGETS[0] else {},
    } for target in TARGETS}
    with tempfile.TemporaryDirectory(prefix="crps_stage5_check_") as directory:
        output = Path(directory)
        draw_figure(feature, temporal, policies, feature_names, output)
        for extension in ("png", "pdf", "svg"):
            if not (output / (
                "Figure5_conditional_ensemble_sensitivity." + extension
            )).is_file():
                raise AssertionError("figure preflight failed for " + extension)
    print("Pandas aggregation and Figure 5 preflight: PASS", flush=True)


def main() -> None:
    print("=" * 68, flush=True)
    print(" Stage 5 V5.0.2 conditional ensemble interpretation", flush=True)
    print(" Script build:", SCRIPT_BUILD, flush=True)
    print("=" * 68, flush=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    preflight_compatibility()
    stage4 = import_stage4()
    run, config4, summary4 = accepted_stage4_run()
    print("Accepted Stage 4:", run, flush=True)
    stage3_run = Path(config4["stage3_run"])
    names = stage4.read_selected_feature_names(stage3_run)
    if (stage3_run.is_dir() is False or
            (run / "selected_features.txt").read_text(encoding="utf-8-sig").split()
            != names):
        raise ValueError("Stage 3 lineage/feature list does not match Stage 4")
    datasets = stage4.load_all_stage3_datasets(stage3_run, names)
    scaler = load_standardizer(stage4, run, names)
    manifest = pd.read_csv(run / "battery_split_manifest.csv", encoding="utf-8-sig")
    if manifest.duplicated(["dataset", "battery_id"]).any():
        raise ValueError("Stage 4 battery split manifest contains duplicate batteries")

    output = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    report = output / "validation_report"
    figures = output / "figures"
    report.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    replay_rows: List[Dict[str, Any]] = []
    features: List[pd.DataFrame] = []
    times: List[pd.DataFrame] = []
    samples: List[pd.DataFrame] = []
    branch_removals: List[pd.DataFrame] = []
    policy_by_target: Dict[str, Dict[str, Any]] = {}
    for target in TARGETS:
        print("\n========", target, "========", flush=True)
        train_ids = split_batteries(manifest, target, "target_adaptation_train")
        val_ids = split_batteries(manifest, target, "target_validation")
        test_ids = split_batteries(manifest, target, "target_test")
        if (not val_ids or not test_ids or train_ids & val_ids or
                train_ids & test_ids or val_ids & test_ids):
            raise ValueError(target + " test/validation/training batteries overlap")
        validation = stage4.BatteryWindowDataset(
            datasets, {target: val_ids}, scaler, SEQUENCE_LENGTH
        )
        test = stage4.BatteryWindowDataset(
            datasets, {target: test_ids}, scaler, SEQUENCE_LENGTH
        )
        policy = load_policy(run, target, stage4)
        policy_by_target[target] = policy
        models = load_models(stage4, run, target)
        for split, dataset in (("validation", validation), ("test", test)):
            stored_path = (run / "predictions" / target /
                           ("validation_predictions.csv" if split == "validation"
                            else "test_predictions.csv"))
            stored = pd.read_csv(stored_path, encoding="utf-8-sig")
            if set(stored["battery_id"].astype(str)) != (val_ids if split == "validation"
                                                       else test_ids):
                raise ValueError(target + " " + split + " prediction battery IDs differ")
            print(target, "replaying", split, len(dataset), "windows", flush=True)
            replay = replay_predictions(stage4, dataset, models, policy)
            replay_rows.extend(compare_replay(stored, replay, target, split))
            if split == "test":
                f, importance, temporal = explain_target(
                    target, dataset, validation, replay, names, models, policy, stage4
                )
                samples.append(f)
                features.append(importance)
                times.append(temporal)
                branch_removals.append(
                    descriptive_branch_removal(stored, policy, stage4, target)
                )
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    replay_frame = pd.DataFrame(replay_rows)
    sample_frame = pd.concat(samples, ignore_index=True)
    feature_frame = pd.concat(features, ignore_index=True)
    time_frame = pd.concat(times, ignore_index=True)
    removal_frame = pd.concat(branch_removals, ignore_index=True)
    if not replay_frame["check_pass"].all():
        raise ValueError("Stage 4 prediction replay did not pass")
    for target in TARGETS:
        for name, frame in (("feature", feature_frame), ("time", time_frame)):
            value = frame.loc[frame["target"].eq(target),
                              "normalized_sensitivity"].sum()
            if not np.isclose(value, 1.0, atol=1e-7):
                raise ValueError(target + " " + name + " scores do not sum to one")
    save_csv(replay_frame, report / "prediction_replay_checks.csv")
    save_csv(sample_frame, report / "window_level_sensitivity.csv")
    save_csv(feature_frame, report / "feature_sensitivity.csv")
    save_csv(time_frame, report / "temporal_sensitivity.csv")
    save_csv(removal_frame, report / "branch_removal_test_description.csv")
    save_json({
        "schema_version": SCHEMA_VERSION, "script_build": SCRIPT_BUILD,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "stage4_run": str(run), "stage4_script_build": STAGE4_BUILD,
        "stage3_run": str(stage3_run), "feature_names": names,
        "models": list(MODELS), "sequence_length": SEQUENCE_LENGTH,
        "test_windows_per_battery": TEST_WINDOWS_PER_BATTERY,
        "background_windows_per_battery": BACKGROUND_WINDOWS_PER_BATTERY,
        "background_windows_per_explanation": BACKGROUND_WINDOWS_PER_EXPLANATION,
        "bootstrap_battery_repetitions": BOOTSTRAPS,
        "explanation_measure": "non-additive validation-reference replacement sensitivity",
        "reference_labels_used": False,
        "test_labels_used_for_explanation_or_model_selection": False,
        "branch_removal_test_labels_used_for_reporting_only": True,
        "prediction_replay_atol": PREDICTION_ATOL,
        "prediction_replay_rtol": PREDICTION_RTOL,
    }, report / "processing_config.json")
    save_json({
        "validation_status": "PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "stage4_run": str(run),
        "stage4_validation_status": summary4.get("validation_status"),
        "replayed_window_count_by_target": {
            target: int(replay_frame.loc[(replay_frame["target"] == target) &
                                         (replay_frame["column"] == "Adaptive_Ensemble"),
                                         "window_count"].sum())
            for target in TARGETS
        },
        "explained_window_count_by_target": {
            target: int(sample_frame["target"].eq(target).sum())
            for target in TARGETS
        },
        "selected_branches_by_target": {
            target: policy_by_target[target]["selected_models"]
            for target in TARGETS
        },
        "fusion_mode_by_target": {
            target: policy_by_target[target]["mode"] for target in TARGETS
        },
        "maximum_absolute_replay_error": float(
            replay_frame["maximum_absolute_error"].max()
        ),
    }, report / "processing_summary.json")
    draw_figure(feature_frame, time_frame, policy_by_target, names, figures)
    (report / "README.txt").write_text(
        "Stage 5 interpretation of the accepted Stage 4 conditional ensemble.\n"
        "feature_sensitivity.csv and temporal_sensitivity.csv contain mean\n"
        "absolute prediction changes under validation-reference replacement.\n"
        "The axes are normalized independently and are NOT SHAP explanations.\n"
        "The reference/sample selection does not use SOH labels.\n"
        "The branch-removal table uses final-test SOH only for descriptive\n"
        "evaluation; it never controls selected models or weights.\n"
        "Figure E/F show saved global/condition weights, not TCN attention.\n",
        encoding="utf-8",
    )
    zipped = output / "STAGE5_VALIDATION_REPORT.zip"
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(report.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    print("\nValidation status: PASS", flush=True)
    print("Selected branches:", {k: v["selected_models"]
                                  for k, v in policy_by_target.items()}, flush=True)
    print("Maximum replay error:", replay_frame["maximum_absolute_error"].max(),
          flush=True)
    print("Feature sensitivity:", report / "feature_sensitivity.csv", flush=True)
    print("Figure:", figures / "Figure5_conditional_ensemble_sensitivity.png",
          flush=True)
    print("Validation report:", zipped, flush=True)


if __name__ == "__main__":
    main()
