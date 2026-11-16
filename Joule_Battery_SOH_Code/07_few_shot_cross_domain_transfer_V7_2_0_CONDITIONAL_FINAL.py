from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


BASE = Path(os.environ.get(
    "CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"
))
STAGE4_ROOT = BASE / "SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION"
OUTPUT_ROOT = BASE / "CROSS_DOMAIN_RESULTS_V7_2_0_CONDITIONAL"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
BUILD = "STAGE7_V7_2_0_CONDITIONAL_MATCHED_20261006"
SCHEMA = "7.2.0"
SEEDS = (42, 52, 62)
BUDGETS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
BOOTSTRAPS = 1000
TARGET_PATIENCE = 20
NAMES = (
    "CNN_BiLSTM_Attention", "MLP_Feature_Attention",
    "HGB_Window_SOH", "Ridge_Window_SOH",
)
OUTPUT_NAMES = NAMES + ("Average_Ensemble", "Adaptive_Ensemble")
NEURAL = NAMES[:2]
KEYS = ["dataset", "battery_id", "window_index", "condition_group",
        "history_start_cycle", "cycle", "history_length"]
CASES = {
    "XJTU_HUST_to_TJU_NCM": {
        "source": {"XJTU": ("source_train", "source_validation"),
                   "HUST": ("source_train", "source_validation")},
        "target": "TJU_NCM",
    },
    "TJU_NCM_to_TJU_NCA": {
        "source": {"TJU_NCM": ("target_adaptation_train", "target_validation")},
        "target": "TJU_NCA",
    },
}


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=BASE)
    parser.add_argument("--stage4-run", type=Path)
    parser.add_argument("--stage4-script", type=Path)
    parser.add_argument("--run-dir", type=Path, help="Existing Stage 7 run to resume")
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot JSON encode " + type(value).__name__)


def read_report_json(run: Path, name: str) -> Dict[str, Any]:
    direct = run / "validation_report" / name
    if direct.is_file():
        return json.loads(direct.read_text(encoding="utf-8-sig"))
    archive = run / "STAGE4_VALIDATION_REPORT.zip"
    with zipfile.ZipFile(archive) as zipped:
        member = next((item for item in zipped.namelist()
                       if Path(item).name == name), None)
        if member is None:
            raise FileNotFoundError(name + " is missing from " + str(archive))
        return json.loads(zipped.read(member).decode("utf-8-sig"))


def accepted_stage4(root: Path, explicit: Optional[Path]) -> Tuple[Path, Dict[str, Any]]:
    candidates = [explicit] if explicit else sorted(root.glob("run_*"), reverse=True)
    rejected: List[str] = []
    for run in candidates:
        try:
            if not run.is_dir():
                raise FileNotFoundError(run)
            summary = read_report_json(run, "processing_summary.json")
            config = read_report_json(run, "processing_config.json")
            if summary.get("validation_status") != "PASS":
                raise ValueError("validation is not PASS")
            if summary.get("script_build") != STAGE4_BUILD or config.get("script_build") != STAGE4_BUILD:
                raise ValueError("Stage 4 build mismatch")
            if config.get("schema_version") != "4.0.2" or summary.get("schema_version") != "4.0.2":
                raise ValueError("Stage 4 schema mismatch")
            if config.get("test_labels_used_for_training") is not False:
                raise ValueError("Stage 4 reports test-label leakage")
            for name in ("battery_split_manifest.csv", "source_standardizer.pkl"):
                if not (run / name).is_file():
                    raise FileNotFoundError(run / name)
            return run, config
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            rejected.append(str(run) + ": " + str(error))
    raise FileNotFoundError("No accepted conditional Stage 4 run:\n" + "\n".join(rejected[:8]))


def import_stage4(explicit: Optional[Path]) -> Any:
    folder = Path(__file__).resolve().parent
    candidates = ([explicit] if explicit else [
        folder / "04_cross_domain_ensemble_transfer_learning_V4_0_2.py",
        folder / "04_cross_domain_ensemble_transfer_learning_V4_0_2_CONDITIONAL_CONTRIBUTION.py",
    ])
    candidates += sorted(folder.glob("04_cross_domain_ensemble_transfer_learning*.py"))
    for path in candidates:
        if path is None or not path.is_file():
            continue

        source = path.read_text(encoding="utf-8-sig")
        if STAGE4_BUILD not in source:
            continue
        spec = importlib.util.spec_from_file_location("crps_stage4_conditional_v402", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.SCRIPT_BUILD != STAGE4_BUILD or tuple(module.MODEL_NAMES) != NAMES:
            continue
        return module
    raise FileNotFoundError("Place the accepted Stage 4 R2 script next to Stage 7 "
                            "or provide --stage4-script PATH (required build: " + STAGE4_BUILD + ")")


def manifest_selections(manifest: pd.DataFrame, datasets: Mapping[str, Any],
                        case: str, stage4: Any) -> Tuple[Dict[str, Set[str]],
                                                            Dict[str, Set[str]], Set[str], Set[str]]:
    needed = {"dataset", "battery_id", "split", "condition_group", "sequence_eligible"}
    if needed - set(manifest):
        raise ValueError("Stage 4 manifest lacks " + repr(sorted(needed - set(manifest))))
    frame = manifest.copy()
    for key in ("dataset", "battery_id", "split", "condition_group"):
        frame[key] = frame[key].astype(str)
    if frame["sequence_eligible"].dtype != bool:
        values = frame["sequence_eligible"].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError("Invalid sequence_eligible values in Stage 4 manifest")
        frame["sequence_eligible"] = values.eq("true")
    if frame.duplicated(["dataset", "battery_id"]).any():
        raise ValueError("Duplicate battery IDs in the Stage 4 manifest")
    source_train: Dict[str, Set[str]] = {}
    source_val: Dict[str, Set[str]] = {}
    for domain, (train_split, val_split) in CASES[case]["source"].items():
        source_train[domain] = set(frame.loc[(frame.dataset == domain)
            & (frame.split == train_split) & frame.sequence_eligible.astype(bool), "battery_id"])
        source_val[domain] = set(frame.loc[(frame.dataset == domain)
            & (frame.split == val_split) & frame.sequence_eligible.astype(bool), "battery_id"])
        if not source_train[domain] or not source_val[domain] or source_train[domain] & source_val[domain]:
            raise ValueError(case + " source train/validation overlap or empty")
    target = CASES[case]["target"]
    target_frame = frame[frame.dataset == target]
    acquisition = set(target_frame.loc[target_frame.split.isin(
        ["target_adaptation_train", "target_validation"])
        & target_frame.sequence_eligible.astype(bool), "battery_id"])
    final_test = set(target_frame.loc[(target_frame.split == "target_test")
        & target_frame.sequence_eligible.astype(bool), "battery_id"])
    ids = stage4.battery_indices(datasets[target])
    eligible = {key for key, value in ids.items() if len(value) >= stage4.SEQUENCE_LENGTH}
    if not acquisition or not final_test or acquisition & final_test or acquisition | final_test != eligible:
        raise ValueError(case + " incomplete or overlapping target allocation")
    return source_train, source_val, acquisition, final_test


def fit_source_scaler(stage4: Any, datasets: Mapping[str, Any],
                      selections: Mapping[str, Set[str]], names: Sequence[str]) -> Any:
    means, seconds, fitted = [], [], {}
    for domain, battery_ids in selections.items():
        X = np.asarray(datasets[domain]["X"], dtype=np.float64)
        grouped = stage4.battery_indices(datasets[domain])
        per_battery = [X[grouped[key]] for key in sorted(battery_ids)]
        means.append(np.stack([part.mean(axis=0) for part in per_battery]).mean(axis=0))
        seconds.append(np.stack([(part * part).mean(axis=0) for part in per_battery]).mean(axis=0))
        fitted[domain] = sorted(battery_ids)
    mean = np.stack(means).mean(axis=0)
    scale = np.sqrt(np.maximum(np.stack(seconds).mean(axis=0) - mean**2, 0))
    scale[scale < 1e-8] = 1
    return stage4.FrozenStandardizer(mean, scale, names, list(selections), fitted)


def acquisition_order(rows: pd.DataFrame, eligible_ids: Set[str], seed: int) -> List[str]:
    if seed < 0:
        raise ValueError("Acquisition seed must be nonnegative")
    rng = np.random.default_rng(seed)
    selected = rows.loc[rows.battery_id.astype(str).isin(eligible_ids)]
    if selected.battery_id.nunique() != len(eligible_ids):
        raise ValueError("Missing target acquisition battery condition")
    groups = {str(key): [items[i] for i in rng.permutation(len(items))]
              for key, block in selected.groupby("condition_group", sort=True)
              for items in [sorted(block.battery_id.astype(str).tolist())]}
    consumed = defaultdict(int)
    order = []
    while len(order) < len(eligible_ids):
        available = [key for key in groups if consumed[key] < len(groups[key])]
        chosen = min(available, key=lambda key: (consumed[key] / len(groups[key]), key))
        order.append(groups[chosen][consumed[chosen]])
        consumed[chosen] += 1
    if len(set(order)) != len(eligible_ids):
        raise ValueError("Nonunique acquired batteries")
    return order


def budget_split(ordered: Sequence[str], count: int, seed: int) -> Tuple[Set[str], Set[str]]:
    selected = list(ordered[:count])
    if len(selected) < 2:
        raise ValueError("Nonzero budget needs adaptation and calibration batteries")
    calibration_size = min(len(selected) - 1, max(1, round(0.2 * len(selected))))
    rng = np.random.default_rng(seed + 100003)
    positions = rng.permutation(len(selected))
    calibration = {selected[int(i)] for i in positions[:calibration_size]}
    return set(selected) - calibration, calibration


def zero_label_test_payload(datasets: Mapping[str, Any], target: str) -> Dict[str, Any]:
    copy = dict(datasets)
    payload = dict(datasets[target])
    payload["y_soh"] = np.zeros(len(payload["y_soh"]), dtype=np.float32)
    copy[target] = payload
    return copy


def output_frame(stage4: Any, dataset: Any, models: Mapping[str, Any],
                 template: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    frames = {}
    for name in NEURAL:
        frames[name] = stage4.predict_model(models[name], dataset)
    base = frames[NEURAL[0]] if template is None else template
    for name in NAMES[2:]:
        frames[name] = stage4.tabular_prediction_frame(base, dataset, models[name])
    merged = stage4.merge_model_predictions(frames)
    if not np.isfinite(merged[list(NAMES)].to_numpy(float)).all():
        raise ValueError("Non-finite candidate predictions")
    return merged


def attach_truth(predictions: pd.DataFrame, target: str, original: Mapping[str, Any]) -> pd.DataFrame:
    payload = original[target]
    labels = pd.DataFrame({
        "dataset": target,
        "battery_id": np.asarray(payload["battery_id"]).astype(str),
        "cycle": np.asarray(payload["cycle"]).astype(int),
        "y_final": np.asarray(payload["y_soh"], dtype=float),
    })
    if labels.duplicated(["dataset", "battery_id", "cycle"]).any():
        raise ValueError("Duplicate test keys")
    joined = predictions.drop(columns="y_true").merge(
        labels, on=["dataset", "battery_id", "cycle"], how="left",
        validate="many_to_one", sort=False)
    if len(joined) != len(predictions) or joined.y_final.isna().any():
        raise ValueError("Cannot attach final-test truth")
    return joined.rename(columns={"y_final": "y_true"})


def average_and_adaptive(stage4: Any, frame: pd.DataFrame,
                         policy: Mapping[str, Any]) -> pd.DataFrame:
    output = stage4.add_ensemble_prediction(frame, policy)
    output["Average_Ensemble"] = output[list(NAMES)].to_numpy(float).mean(axis=1)
    return output


def calibration_policy(stage4: Any, frame: pd.DataFrame,
                       target: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    count = int(frame.battery_id.nunique())
    if count >= 4:
        policy, diagnostics = stage4.select_validation_ensemble(frame, target)
        policy["selection_method"] = "Stage4_battery_CV"
        return policy, diagnostics

    from scipy.optimize import minimize
    matrix = frame[list(NAMES)].to_numpy(float)
    y = frame.y_true.to_numpy(float)

    prior = np.full(len(NAMES), 1.0 / len(NAMES))
    lambda_batteries = 2.0 / count
    error_scale = max(float(np.mean((matrix.mean(axis=1) - y)**2)), 1e-6)

    def objective(w: np.ndarray) -> float:
        error = matrix @ w - y
        battery_mse = np.mean([float(np.mean(error[index]**2))
            for index in frame.groupby("battery_id", sort=True).indices.values()])
        return float(battery_mse / error_scale + lambda_batteries * np.sum((w - prior)**2))

    solution = minimize(objective, prior.copy(), method="SLSQP",
        bounds=[(0, 1)] * len(NAMES),
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1)}],
        options={"maxiter": 250, "ftol": 1e-10})
    if not solution.success or not np.isfinite(solution.x).all():
        print(target, "low-budget fusion solver did not converge; using the "
              "predeclared equal-weight fallback:", solution.message, flush=True)
        raw = prior
        method = "equal_prior_fallback_solver_failed"
    else:
        raw = solution.x
        method = "regularized_target_calibration_1_to_3_batteries"
    weights = np.maximum(raw, 0)
    weights /= weights.sum()
    policy = {"mode": "global", "selected_models": list(NAMES),
              "global_weights": weights, "condition_weights": {},
              "selection_method": method}
    diagnostics = pd.DataFrame([{"target": target, "selected": True,
        "candidate_models": "|".join(NAMES), "fusion_mode": "global",
        "calibration_batteries": count, "test_labels_used": False,
        "selection_method": policy["selection_method"]}])
    return policy, diagnostics


def train_or_resume(stage4: Any, folder: Path, model_name: str, train: Any,
                    validation: Any, seed: int, phase: str, source_state: Optional[Mapping[str, Any]],
                    max_epochs: int, lr: float, patience: int) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint = folder / (model_name + "_best.pth")
    info_path = folder / (model_name + "_training.json")
    history_path = folder / (model_name + "_history.csv")
    stage4.set_seed(seed)
    model = stage4.build_model(model_name)
    if checkpoint.is_file() and info_path.is_file() and history_path.is_file():
        saved = torch.load(checkpoint, map_location="cpu")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if (saved.get("script_build") == BUILD and saved.get("phase") == phase
            and saved.get("model_name") == model_name and info.get("seed") == seed):
            model.load_state_dict(saved["state_dict"], strict=True)
            model.to(stage4.DEVICE).eval()
            print("Resumed checkpoint:", checkpoint, flush=True)
            return model, pd.read_csv(history_path).to_dict("records"), info
        raise ValueError("Existing Stage 7 checkpoint lineage differs: " + str(checkpoint))
    if source_state is not None:
        model.load_state_dict(source_state, strict=True)
    print(phase, model_name, "training started; windows", len(train), len(validation), flush=True)
    model, history, info = stage4.train_model(
        model, train, validation, max_epochs, patience, lr, checkpoint,
        phase, model_name, seed)
    saved = torch.load(checkpoint, map_location="cpu")
    saved.update({"script_build": BUILD, "stage4_model_build": STAGE4_BUILD,
                  "seed": seed, "case_phase": phase})
    torch.save(saved, checkpoint)
    info = dict(info, seed=seed, phase=phase, model=model_name,
                best_checkpoint_from_source=source_state is not None,
                training_budget_ceiling_reached=int(info["best_epoch"]) == max_epochs)
    pd.DataFrame(history).to_csv(history_path, index=False)
    write_json(info_path, info)
    return model, history, info


def fit_or_resume_tabular(stage4: Any, folder: Path, model_name: str,
                          dataset: Any, seed: int) -> Any:
    path = folder / (model_name + "_best.pkl")
    if path.is_file():
        with path.open("rb") as source:
            saved = pickle.load(source)
        if saved.get("script_build") != BUILD or saved.get("model_name") != model_name:
            raise ValueError("Tabular checkpoint mismatch: " + str(path))
        print("Resumed tabular:", path, flush=True)
        return saved["model"]
    print("Fitting tabular branch:", folder, model_name, flush=True)
    fitted = (stage4.fit_independent_hgb(dataset, seed) if model_name == NAMES[2]
              else stage4.fit_independent_ridge(dataset, seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        pickle.dump({"script_build": BUILD, "model_name": model_name,
                     "seed": seed, "model": fitted}, output,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return fitted


def fit_arm(stage4: Any, folder: Path, train: Any, validation: Any,
            source_states: Optional[Mapping[str, Any]], seed: int, phase: str,
            source: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fitted, history_all, summaries = {}, [], []
    for index, name in enumerate(NEURAL):
        epochs = (stage4.MAX_EPOCHS_SOURCE if source else
                  max(350 if name == NEURAL[0] else 250,
                      stage4.TARGET_MAX_EPOCHS_BY_MODEL[name]))
        model, history, info = train_or_resume(
            stage4, folder, name, train, validation, seed + index,
            phase, None if source_states is None else source_states[name],
            epochs,
            stage4.SOURCE_LEARNING_RATE if source else stage4.TARGET_LEARNING_RATE,
            stage4.PATIENCE_SOURCE if source else TARGET_PATIENCE)
        fitted[name] = model
        history_all.extend(history)
        summaries.append(info)
    for index, name in enumerate(NAMES[2:], start=2):
        fitted[name] = fit_or_resume_tabular(stage4, folder, name, train, seed + index)
    return fitted, history_all, summaries


def release_models(stage4: Any, models: Mapping[str, Any]) -> None:
    for name in NEURAL:
        if name in models:
            models[name].cpu()
    if stage4.DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def metrics_for(frame: pd.DataFrame, case: str, arm: str, fraction: float,
                seed: int, n_acquired: int, n_train: int, n_val: int,
                total: int) -> List[Dict[str, Any]]:
    rows = []
    for name in OUTPUT_NAMES:
        for scope in ("pooled_cycles", "battery_macro"):
            if scope == "pooled_cycles":
                values = [score(frame.y_true.to_numpy(float), frame[name].to_numpy(float))]
            else:
                values = [score(group.y_true.to_numpy(float), group[name].to_numpy(float))
                    for _, group in frame.groupby("battery_id", sort=True)]
            row = {"case": case, "source_domains": "+".join(CASES[case]["source"]),
                   "target_domain": CASES[case]["target"], "evaluation_domain": CASES[case]["target"],
                   "evaluation_split": "immutable_final_test", "arm": arm,
                   "replicate_seed": int(seed), "requested_target_label_fraction": fraction,
                   "achieved_target_label_fraction": n_acquired / total,
                   "acquired_target_battery_count": n_acquired,
                   "target_adaptation_battery_count": n_train,
                   "target_calibration_battery_count": n_val,
                   "model": name, "evaluation_scope": scope,
                   "sample_count": len(frame), "battery_count": frame.battery_id.nunique()}
            row.update({key: float(np.nanmean([v[key] for v in values]))
                        for key in ("RMSE", "MAE", "MAPE_percent", "R2")})
            rows.append(row)
    return rows


def score(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    err = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
    y = np.asarray(truth, dtype=float)
    denominator = float(np.sum((y - y.mean())**2))
    return {"RMSE": float(np.sqrt(np.mean(err**2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE_percent": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-8)) * 100),
            "R2": float(1 - np.sum(err**2) / denominator) if denominator > 0 else float("nan")}


def learning_curve(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["case", "source_domains", "target_domain", "evaluation_domain",
               "evaluation_split", "arm", "requested_target_label_fraction",
               "achieved_target_label_fraction", "acquired_target_battery_count",
               "target_adaptation_battery_count", "target_calibration_battery_count",
               "model", "evaluation_scope", "sample_count", "battery_count"]
    rows = []
    for keys, group in frame.groupby(columns, sort=False, dropna=False):
        result = dict(zip(columns, keys))
        result["replicate_count"] = group.replicate_seed.nunique()
        result["independent_source_training_count"] = (1 if float(result[
            "requested_target_label_fraction"]) == 0 else 0)
        for column in ("RMSE", "MAE", "MAPE_percent", "R2"):
            values = group[column].to_numpy(float)
            result[column + "_mean"] = float(np.nanmean(values))
            result[column + "_std_across_seeds"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"))
        rows.append(result)
    return pd.DataFrame(rows)


def aligned(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = a.sort_values(KEYS, kind="stable").reset_index(drop=True)
    right = b.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if (len(left) != len(right) or not left[KEYS].equals(right[KEYS]) or
            not np.allclose(left.y_true, right.y_true, atol=1e-8, rtol=0)):
        raise ValueError("Paired target predictions have different battery-cycle keys or labels")
    return left, right


def paired_effects(predictions: Mapping[Tuple[str, int, float, str], pd.DataFrame],
                   against: str, iterations: int, seed: int,
                   model: str = "Adaptive_Ensemble") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for case in CASES:
        for fraction in BUDGETS[1:]:
            by_seed = {}
            for current_seed in SEEDS:
                treated = predictions[(case, current_seed, fraction, "transfer")]
                control = predictions[(case, -1, 0.0, "transfer") if against == "zero_shot"
                                      else (case, current_seed, fraction, "scratch")]
                a, b = aligned(treated, control)
                battery = a.battery_id.astype(str).to_numpy()
                truth = a.y_true.to_numpy(float)
                groups = []
                for battery_id in sorted(set(battery)):
                    mask = battery == battery_id
                    e_a = a.loc[mask, model].to_numpy(float) - truth[mask]
                    e_b = b.loc[mask, model].to_numpy(float) - truth[mask]
                    groups.append((int(mask.sum()), float(np.sum(e_b**2)),
                                   float(np.sum(e_a**2))))
                by_seed[current_seed] = np.asarray(groups, dtype=float)
            def difference(items: Sequence[np.ndarray]) -> float:
                return float(np.mean([np.sqrt(part[:, 1].sum() / part[:, 0].sum())
                           - np.sqrt(part[:, 2].sum() / part[:, 0].sum())
                           for part in items]))
            observed = difference(list(by_seed.values()))
            drawn = np.empty(iterations)
            for i in range(iterations):
                per_seed = []
                for sampled_seed in rng.choice(SEEDS, size=len(SEEDS), replace=True):
                    array = by_seed[int(sampled_seed)]
                    per_seed.append(array[rng.integers(0, len(array), len(array))])
                drawn[i] = difference(per_seed)
            example = predictions[(case, SEEDS[0], fraction, "transfer")]
            lower, upper = np.percentile(drawn, [2.5, 97.5])
            row = {"case": case, "model": model,
                "requested_target_label_fraction": fraction,
                "achieved_target_label_fraction": float(example.achieved_target_label_fraction.iloc[0]),
                "acquired_target_battery_count": int(example.acquired_target_battery_count.iloc[0]),
                "observed_RMSE_improvement_mean_across_seeds": observed,
                "RMSE_improvement_ci95_lower": float(lower),
                "RMSE_improvement_ci95_upper": float(upper),
                "probability_treatment_lower_RMSE": float(np.mean(drawn > 0)),
                "replicate_count": len(SEEDS), "bootstrap_iterations": iterations,
                "comparison": "zero_shot_minus_few_shot" if against == "zero_shot"
                              else "scratch_minus_transfer"}
            if against == "zero_shot":
                row["few_shot_requested_target_label_fraction"] = fraction
            rows.append(row)
    return pd.DataFrame(rows)


def shift_rows(stage4: Any, source: Any, target: Any, case: str,
               names: Sequence[str]) -> List[Dict[str, Any]]:

    def balanced_mean(dataset: Any) -> np.ndarray:
        by_battery = [record["windows"][:, -1, :].mean(axis=0)
                      for record in dataset.records]
        return np.stack(by_battery).mean(axis=0)
    left, right = balanced_mean(source), balanced_mean(target)
    return [{"case": case, "feature": name,
             "absolute_standardized_mean_shift": float(abs(a-b))}
            for name, a, b in zip(names, left, right)]


def validate(protocol: pd.DataFrame, metrics: pd.DataFrame,
             histories: pd.DataFrame, predictions: Mapping[Tuple[str, int, float, str], pd.DataFrame],
             stage4: Any) -> Tuple[str, List[str], List[str]]:
    fatal, warnings_list = [], []
    for case in CASES:
        test_ids = set(protocol.loc[(protocol.case == case)
            & (protocol.role == "immutable_final_test"), "battery_id"])
        for seed in SEEDS:
            previous = set()
            for fraction in BUDGETS[1:]:
                subset = protocol.loc[(protocol.case == case)
                    & (protocol.replicate_seed == seed)
                    & (np.isclose(protocol.requested_target_label_fraction, fraction))]
                train = set(subset.loc[subset.role == "target_adaptation", "battery_id"])
                val = set(subset.loc[subset.role == "target_calibration", "battery_id"])
                acquired = train | val
                if (train & val or acquired & test_ids or not previous.issubset(acquired)
                        or len(acquired) != int(subset.acquired_target_battery_count.iloc[0])):
                    fatal.append("budget overlap / nesting: " + case + " " + str((seed, fraction)))
                previous = acquired
                for arm in ("transfer", "scratch"):
                    key = (case, seed, fraction, arm)
                    if key not in predictions:
                        fatal.append("missing " + str(key))
                    elif set(predictions[key].battery_id.astype(str)) != test_ids:
                        fatal.append("test battery mismatch " + str(key))
                    elif (not np.isfinite(predictions[key][list(OUTPUT_NAMES)]
                            .to_numpy(float)).all()):
                        fatal.append("nonfinite predictions " + str(key))
        key = (case, -1, 0.0, "transfer")
        if key not in predictions:
            fatal.append("zero-shot prediction missing " + case)
    counts = metrics.groupby(["case", "arm", "requested_target_label_fraction",
                              "replicate_seed", "evaluation_scope"])["model"].nunique()
    if not counts.eq(len(OUTPUT_NAMES)).all():
        fatal.append("candidate or ensemble metrics incomplete")
    if not histories.empty:
        ceiling = histories.loc[histories.training_budget_ceiling_reached.fillna(False).astype(bool)]
        if len(ceiling):
            warnings_list.append("Best epoch reached ceiling in " + str(len(ceiling))
                                 + " neural fits; inspect training_summary.csv")
    return ("PASS" if not fatal else "FAIL"), fatal, warnings_list


def main() -> None:
    args = args_parser()
    if args.bootstrap_iterations < 100:
        raise ValueError("Bootstrap iterations must be >=100")
    stage4 = import_stage4(args.stage4_script)
    root = args.base_path / STAGE4_ROOT.name
    stage4_run, config = accepted_stage4(root, args.stage4_run)
    stage3_run = Path(config["stage3_run"])
    names = stage4.read_selected_feature_names(stage3_run)
    datasets = stage4.load_all_stage3_datasets(stage3_run, names)
    manifest = pd.read_csv(stage4_run / "battery_split_manifest.csv",
                           encoding="utf-8-sig", dtype={"battery_id": str})
    run = (args.run_dir if args.run_dir else
           args.base_path / OUTPUT_ROOT.name / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    report = run / "validation_report"
    report.mkdir(parents=True, exist_ok=True)
    identity = run / "run_identity.json"
    info = {"script_build": BUILD, "stage4_run": str(stage4_run.resolve()),
            "stage3_run": str(stage3_run.resolve()), "replicate_seeds": list(SEEDS),
            "budgets": list(BUDGETS), "models": list(NAMES)}
    if identity.is_file() and json.loads(identity.read_text(encoding="utf-8")) != info:
        raise ValueError("--run-dir belongs to a different Stage 3/4 lineage")
    write_json(identity, info)
    print("=" * 72, "\nStage 7 conditional ensemble and matched scratch", "\nBuild:", BUILD,
          "\nStage 4:", stage4_run, "\nStage 3:", stage3_run,
          "\nDevice:", stage4.DEVICE, "\nOutput:", run, "\n" + "=" * 72, flush=True)

    metrics_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    shifts: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, int, float, str], pd.DataFrame] = {}

    for case_index, (case, definition) in enumerate(CASES.items()):
        target = definition["target"]
        print("\n===", case, "===", flush=True)
        source_train, source_val, acquisition, test_ids = manifest_selections(
            manifest, datasets, case, stage4)
        scaler = fit_source_scaler(stage4, datasets, source_train, names)
        case_root = run / "checkpoints" / case
        case_root.mkdir(parents=True, exist_ok=True)
        with (case_root / "source_scaler.pkl").open("wb") as output:
            pickle.dump(scaler.to_serializable_dict(), output)
        masked = zero_label_test_payload(datasets, target)
        source_data = stage4.BatteryWindowDataset(datasets, source_train, scaler,
                                                  stage4.SEQUENCE_LENGTH)
        source_val_data = stage4.BatteryWindowDataset(datasets, source_val, scaler,
                                                      stage4.SEQUENCE_LENGTH)
        test_data = stage4.BatteryWindowDataset(masked, {target: test_ids}, scaler,
                                                stage4.SEQUENCE_LENGTH)
        source_models, h, s = fit_arm(stage4, case_root / "source", source_data,
                                      source_val_data, None, 10000 + case_index * 1000,
                                      case + "_source_pretraining", source=True)
        history_rows.extend(dict(row, case=case, arm="source") for row in h)
        summaries.extend(dict(row, case=case, arm="source", fraction=0.0) for row in s)
        source_val_frame = output_frame(stage4, source_val_data, source_models)
        source_policy, _ = calibration_policy(stage4, source_val_frame, case + "_source")
        write_json(case_root / "zero_shot_source_fusion_policy.json",
                   dict(source_policy, script_build=BUILD,
                        selected_on="source_validation_batteries", test_labels_used=False))
        source_state = {name: stage4.cpu_state_dict(source_models[name]) for name in NEURAL}
        test_merged = output_frame(stage4, test_data, source_models)
        if not np.allclose(test_merged.y_true, 0):
            raise ValueError("Test labels were visible to model inference")
        zero = attach_truth(average_and_adaptive(stage4, test_merged, source_policy),
                            target, datasets)
        release_models(stage4, source_models)
        total_target = len(acquisition) + len(test_ids)
        zero = zero.assign(case=case, replicate_seed=-1,
                           requested_target_label_fraction=0.0,
                           achieved_target_label_fraction=0.0,
                           acquired_target_battery_count=0,
                           target_adaptation_battery_count=0,
                           target_calibration_battery_count=0, arm="transfer")
        predictions[(case, -1, 0.0, "transfer")] = zero
        metrics_rows.extend(metrics_for(zero, case, "transfer", 0.0, -1, 0, 0, 0, total_target))
        for name, weight in zip(NAMES, stage4.battery_average_weights(zero, source_policy)):
            weight_rows.append({"case": case, "target": target, "arm": "transfer",
                "replicate_seed": -1, "requested_target_label_fraction": 0.0,
                "model": name, "weight": float(weight),
                "selection_method": source_policy["selection_method"],
                "learned_on": "source_validation", "test_labels_used": False})
        (run / "predictions").mkdir(parents=True, exist_ok=True)
        zero.to_csv(run / "predictions" / (case + "_zero_shot.csv"), index=False)
        for domain, batteries in source_train.items():
            for battery_id in sorted(batteries):
                protocol_rows.append({"case": case, "replicate_seed": -1,
                    "requested_target_label_fraction": 0.0, "battery_id": battery_id,
                    "dataset": domain, "role": "source_train",
                    "acquired_target_battery_count": 0})
        for domain, batteries in source_val.items():
            for battery_id in sorted(batteries):
                protocol_rows.append({"case": case, "replicate_seed": -1,
                    "requested_target_label_fraction": 0.0, "battery_id": battery_id,
                    "dataset": domain, "role": "source_validation",
                    "acquired_target_battery_count": 0})
        for battery_id in sorted(test_ids):
            protocol_rows.append({"case": case, "replicate_seed": -1,
                "requested_target_label_fraction": 0.0, "battery_id": battery_id,
                "role": "immutable_final_test", "acquired_target_battery_count": 0})
        shifts.extend(shift_rows(stage4, source_data, test_data, case, names))
        condition_rows = manifest[(manifest.dataset.astype(str) == target)
                                  & manifest.battery_id.astype(str).isin(acquisition)]
        print("Source train/val/test windows:", len(source_data),
              len(source_val_data), len(test_data), "acquisition/test batteries:",
              len(acquisition), len(test_ids), flush=True)

        for seed in SEEDS:
            ordered = acquisition_order(condition_rows, acquisition, seed)
            for fraction in BUDGETS[1:]:
                number = min(len(acquisition), max(2, round(total_target * fraction)))
                adaptation, calibration = budget_split(ordered, number, seed)
                split_root = run / "predictions" / case / ("seed_%d" % seed) / ("budget_%02d" % round(fraction * 100))
                split_root.mkdir(parents=True, exist_ok=True)
                write_json(split_root / "acquired_batteries.json", {
                    "adaptation": sorted(adaptation), "calibration": sorted(calibration),
                    "requested_fraction": fraction, "achieved_fraction": number / total_target})
                for battery_id in sorted(adaptation):
                    protocol_rows.append({"case": case, "replicate_seed": seed,
                        "requested_target_label_fraction": fraction, "battery_id": battery_id,
                        "role": "target_adaptation", "acquired_target_battery_count": number})
                for battery_id in sorted(calibration):
                    protocol_rows.append({"case": case, "replicate_seed": seed,
                        "requested_target_label_fraction": fraction, "battery_id": battery_id,
                        "role": "target_calibration", "acquired_target_battery_count": number})
                train_data = stage4.BatteryWindowDataset(datasets, {target: adaptation}, scaler,
                                                         stage4.SEQUENCE_LENGTH)
                cal_data = stage4.BatteryWindowDataset(datasets, {target: calibration}, scaler,
                                                       stage4.SEQUENCE_LENGTH)
                print("\n", case, "seed", seed, "budget", fraction,
                      "train/calibration windows", len(train_data), len(cal_data), flush=True)
                for arm in ("transfer", "scratch"):
                    output_path = split_root / (arm + "_immutable_final_test.csv")
                    policy_path = split_root / (arm + "_fusion_policy.json")
                    arm_folder = case_root / arm / ("seed_%d" % seed) / ("budget_%02d" % round(fraction * 100))
                    if output_path.is_file() and policy_path.is_file():
                        final = pd.read_csv(output_path, dtype={"battery_id": str})
                        policy = json.loads(policy_path.read_text(encoding="utf-8"))
                        if (policy.get("script_build") != BUILD or
                            len(final) != len(test_data) or
                            set(final.battery_id.astype(str)) != test_ids):
                            raise ValueError("Resumed prediction lineage/count mismatch: " + str(output_path))
                        print("Resumed final prediction:", output_path, flush=True)
                        for name in NEURAL:
                            info_path = arm_folder / (name + "_training.json")
                            hist_path = arm_folder / (name + "_history.csv")
                            if not info_path.is_file() or not hist_path.is_file():
                                raise FileNotFoundError("Completed predictions lack training provenance: "
                                                        + str(arm_folder))
                            stored = json.loads(info_path.read_text(encoding="utf-8"))
                            if stored.get("seed") != seed + 100000 * case_index + 1000 * round(fraction * 100) + NEURAL.index(name):
                                raise ValueError("Resumed training seed differs: " + str(info_path))
                            summaries.append(dict(stored, case=case, arm=arm,
                                                  fraction=fraction, replicate_seed=seed))
                            history_rows.extend(dict(row, case=case, arm=arm, fraction=fraction,
                                replicate_seed=seed) for row in pd.read_csv(hist_path).to_dict("records"))
                    else:
                        target_seed = seed + 100000 * case_index + 1000 * round(fraction * 100)
                        models, h, s = fit_arm(stage4, arm_folder, train_data, cal_data,
                            source_state if arm == "transfer" else None,
                            target_seed, case + "_" + arm)
                        history_rows.extend(dict(row, case=case, arm=arm, fraction=fraction,
                                                replicate_seed=seed) for row in h)
                        summaries.extend(dict(row, case=case, arm=arm, fraction=fraction,
                                              replicate_seed=seed) for row in s)
                        calibration_frame = output_frame(stage4, cal_data, models)
                        policy, diagnostics = calibration_policy(stage4, calibration_frame, target)
                        diagnostics.to_csv(split_root / (arm + "_fusion_candidates.csv"), index=False)
                        test_frame = output_frame(stage4, test_data, models)
                        if not np.allclose(test_frame.y_true, 0):
                            raise ValueError("Final-test labels were visible before fusion")
                        final = attach_truth(average_and_adaptive(stage4, test_frame, policy),
                                             target, datasets)
                        final = final.assign(case=case, arm=arm, replicate_seed=seed,
                            requested_target_label_fraction=fraction,
                            achieved_target_label_fraction=number / total_target,
                            acquired_target_battery_count=number,
                            target_adaptation_battery_count=len(adaptation),
                            target_calibration_battery_count=len(calibration))
                        write_json(policy_path, dict(policy, script_build=BUILD,
                            selected_on="within_budget_target_calibration_batteries",
                            test_labels_used=False))
                        final.to_csv(output_path, index=False)
                        release_models(stage4, models)
                    predictions[(case, seed, fraction, arm)] = final
                    metrics_rows.extend(metrics_for(final, case, arm, fraction, seed,
                        number, len(adaptation), len(calibration), total_target))
                    weights = stage4.battery_average_weights(final, policy)
                    for name, weight in zip(NAMES, weights):
                        weight_rows.append({"case": case, "target": target, "arm": arm,
                            "replicate_seed": seed, "requested_target_label_fraction": fraction,
                            "model": name, "weight": float(weight),
                            "selection_method": policy["selection_method"],
                            "learned_on": "within_budget_calibration", "test_labels_used": False})

    metrics = pd.DataFrame(metrics_rows)
    history = pd.DataFrame(history_rows)
    training = pd.DataFrame(summaries)
    protocol = pd.DataFrame(protocol_rows)
    status, fatal, notes = validate(protocol, metrics, training, predictions, stage4)
    transfer = metrics.loc[metrics.arm == "transfer"].copy()
    scratch = metrics.loc[metrics.arm == "scratch"].copy()
    transfer_curve = learning_curve(transfer)
    scratch_curve = learning_curve(scratch)
    zero_effects = paired_effects(predictions, "zero_shot", args.bootstrap_iterations, 7601)
    scratch_effects = pd.concat([
        paired_effects(predictions, "scratch", args.bootstrap_iterations,
                       7602 + index, model)
        for index, model in enumerate(NAMES + ("Adaptive_Ensemble",))
    ], ignore_index=True)
    outputs = {
        "few_shot_metrics_by_seed.csv": transfer,
        "from_scratch_metrics_by_seed.csv": scratch,
        "learning_curve_summary.csv": transfer_curve,
        "from_scratch_learning_curve_summary.csv": scratch_curve,
        "paired_hierarchical_bootstrap_effects.csv": zero_effects,
        "transfer_vs_scratch_paired_effects.csv": scratch_effects,
        "feature_domain_shift.csv": pd.DataFrame(shifts),
        "battery_budget_manifest.csv": protocol,
        "ensemble_weights.csv": pd.DataFrame(weight_rows),
        "training_summary.csv": training,
        "training_history.csv": history,
    }
    for filename, frame in outputs.items():
        frame.to_csv(report / filename, index=False)
    pd.concat([frame for key, frame in predictions.items() if key[3] == "transfer"],
              ignore_index=True).to_csv(report / "all_final_test_predictions.csv", index=False)
    pd.concat([frame for key, frame in predictions.items() if key[3] == "scratch"],
              ignore_index=True).to_csv(report / "all_from_scratch_test_predictions.csv", index=False)
    write_json(report / "processing_config.json", {
        "schema_version": SCHEMA, "script_build": BUILD,
        "stage4_run": stage4_run, "stage4_build": STAGE4_BUILD,
        "stage3_run": stage3_run, "source_scaler_fit": "source-training batteries only",
        "zero_shot_source_fit_count_per_case": 1,
        "zero_shot_replicate_seed": -1,
        "target_test_truth_hidden_until_prediction_and_policy_frozen": True,
        "calibration_within_budget": True,
        "paired_target_only_scratch_control": True,
        "nonzero_replicate_seeds": SEEDS, "budgets": BUDGETS,
        "matched_target_patience": TARGET_PATIENCE,
        "model_names": NAMES,
        "fusion_rule": "Stage 4 battery CV when >=4 calibration batteries; "
                       "fixed regularized convex fusion when 1-3; all arms matched",
        "bootstrap_iterations": args.bootstrap_iterations,
        "performance_direction_controls_validation": False,
    })
    write_json(report / "processing_summary.json", {
        "schema_version": SCHEMA, "script_build": BUILD,
        "validation_status": status, "fatal_checks": fatal, "warnings": notes,
        "case_count": len(CASES), "final_test_prediction_rows": sum(len(x) for x in predictions.values()),
        "zero_shot_points_are_independent_repetitions": False,
    })
    (report / "README.txt").write_text(
        "Stage 7 V7.2 conditional transfer / matched target-only scratch\n"
        "Positive zero-shot effect: target adaptation improves prediction.\n"
        "Positive scratch-minus-transfer effect: source neural initialization helps.\n"
        "Tabular HGB and Ridge use target labels in both nonzero-budget arms; if "
        "these dominate fusion, an ensemble gain cannot be attributed to pretraining.\n"
        "Every target calibration battery counts toward the acquisition budget.\n"
        "Zero-shot is one source fit, not three independent replicates.\n"
        "Conditionally selected fusion may retain only a subset of branches.\n"
        "PASS means protocol/data integrity, not improved RMSE.\n", encoding="utf-8")
    archive = run / "STAGE7_VALIDATION_REPORT.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(report.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(run))
    display = transfer_curve.loc[(transfer_curve.model == "Adaptive_Ensemble")
                                  & (transfer_curve.evaluation_scope == "pooled_cycles")]
    print("\n" + "=" * 72, "\nValidation status:", status,
          "\nAdaptive ensemble learning curve:\n",
          display[["case", "requested_target_label_fraction", "RMSE_mean", "R2_mean"]].to_string(index=False),
          "\nScratch RMSE minus transfer RMSE (positive means transfer helps):\n",
          scratch_effects.loc[scratch_effects.model == "Adaptive_Ensemble",
                          ["case", "requested_target_label_fraction",
                           "observed_RMSE_improvement_mean_across_seeds",
                           "RMSE_improvement_ci95_lower", "RMSE_improvement_ci95_upper"]].to_string(index=False),
          "\nValidation report:", archive, "\n" + "=" * 72, flush=True)
    if fatal:
        raise RuntimeError("Stage 7 protocol validation failed:\n- " + "\n- ".join(fatal))


if __name__ == "__main__":
    main()
