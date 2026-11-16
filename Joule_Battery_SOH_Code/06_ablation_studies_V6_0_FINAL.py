from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

STAGE6_BUILD = "STAGE6_V6_0_CONDITIONAL_MATCHED_20261006"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
TARGETS = ("TJU_NCM", "TJU_NCA")
MODELS = ("CNN_BiLSTM_Attention", "MLP_Feature_Attention",
          "HGB_Window_SOH", "Ridge_Window_SOH")
NEURAL = MODELS[:2]
KEYS = ("dataset", "battery_id", "window_index", "cycle")
SELECTED_CASE = "Selected20_With_Source_Pretraining_Stage4"
SCRATCH_CASE = "Selected20_Without_Source_Pretraining"
BOOTSTRAPS = 1000
BASE = Path(os.environ.get(
    "CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"))
STAGE4_ROOT = BASE / "SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION"
OUTPUT_ROOT = BASE / "ABLATION_RESULTS_V6_0_CONDITIONAL"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(value: Any, path: Path) -> None:
    def encode_numpy(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError("Cannot serialize %s as JSON" % type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=encode_numpy),
                    encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_in_source(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SCRIPT_BUILD" for t in node.targets
        ):
            return str(ast.literal_eval(node.value))
    return ""


def import_matching_stage4() -> Any:
    folder = Path(__file__).resolve().parent
    explicit = os.environ.get("CRPS_STAGE4_SCRIPT")
    paths = ([Path(explicit)] if explicit else []) + sorted(
        folder.glob("04_cross_domain_ensemble_transfer_learning*.py"))
    for path in paths:
        if not path.is_file():
            continue
        try:
            script_build = build_in_source(path)
        except (OSError, SyntaxError, ValueError):
            continue
        if script_build != STAGE4_BUILD:
            continue
        spec = importlib.util.spec_from_file_location("stage4_ablation_reference", path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load Stage 4 source: " + str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if tuple(module.MODEL_NAMES) != MODELS:
            raise ValueError("Stage 4 candidates changed; do not run an old ablation")
        print("Matching Stage 4 source:", path, flush=True)
        return module
    raise FileNotFoundError("Matching Stage 4 script is required beside Stage 6; "
                            "set CRPS_STAGE4_SCRIPT to its exact path")


def accepted_stage4_run() -> Tuple[Path, Dict[str, Any]]:
    explicit = os.environ.get("CRPS_STAGE4_RUN")
    runs = [Path(explicit)] if explicit else sorted(STAGE4_ROOT.glob("run_*"), reverse=True)
    reasons: List[str] = []
    for run in runs:
        try:
            config = read_json(run / "validation_report" / "processing_config.json")
            summary = read_json(run / "validation_report" / "processing_summary.json")
            if config.get("script_build") != STAGE4_BUILD or summary.get("validation_status") != "PASS":
                raise ValueError("wrong build or validation did not pass")
            if tuple(config.get("models", ())) != MODELS or config.get("split_unit") != "battery":
                raise ValueError("model list or split unit differs")
            if any(bool(config.get(key)) for key in (
                "test_labels_used_for_training", "test_labels_used_for_early_stopping",
                "test_labels_used_for_ensemble_weights")):
                raise ValueError("test labels entered model selection")
            paths = [run / "battery_split_manifest.csv", run / "selected_features.txt"]
            for target in TARGETS:
                paths.extend([run / "predictions" / target / "validation_predictions.csv",
                              run / "predictions" / target / "test_predictions.csv"])
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError("required prediction or split artifact missing")
            return run, config
        except (OSError, KeyError, ValueError) as exc:
            reasons.append("%s: %s" % (run, exc))
    raise FileNotFoundError("No accepted matching Stage 4 run:\n" + "\n".join(reasons[:8]))


def stage2_run_for_lineage(stage3: Path) -> Path:
    config = read_json(stage3 / "validation_report" / "processing_config.json")
    stage2_recorded = config.get("stage2_run") or config.get("stage2_input")
    if not stage2_recorded:
        raise KeyError("Stage 3 processing_config.json has no stage2_run lineage")
    original = Path(str(stage2_recorded))
    override = os.environ.get("CRPS_STAGE2_RUN")
    run = Path(override) if override else original
    if not run.is_dir():
        raise FileNotFoundError("Stage 2 input missing: " + str(run))
    if run.name != original.name:
        raise ValueError("Stage 2 override is not the Stage 3 recorded run")
    summary = read_json(run / "validation_report" / "processing_summary.json")
    if summary.get("validation_status") != "PASS":
        raise ValueError("Stage 2 lineage did not pass validation")
    return run


def load_full_features(stage2: Path, stage3: Path, selected: Sequence[str],
                       stage4: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    definition = pd.read_csv(stage2 / "validation_report" / "feature_definition.csv",
                             encoding="utf-8-sig")
    if "feature" not in definition:
        raise ValueError("Stage 2 feature_definition.csv has no feature column")
    if "uses_current_cycle_capacity_label" not in definition or not definition[
        "uses_current_cycle_capacity_label"].astype(str).str.lower().eq("no").all():
        raise ValueError("One or more Stage 2 features access current-cycle SOH labels")
    names = definition["feature"].astype(str).tolist()
    if len(names) <= len(selected) or len(set(names)) != len(names) or not set(selected).issubset(names):
        raise ValueError("Full candidate set must contain all selected features, with unique names")
    indices = [names.index(name) for name in selected]
    datasets: Dict[str, Dict[str, Any]] = {}
    for domain in stage4.DATASET_ORDER:
        path = stage2 / "datasets" / (domain + "_features.pkl")
        with path.open("rb") as stream:
            payload = dict(pickle.load(stream))
        with (stage3 / "datasets" / (domain + "_selected_features.pkl")).open("rb") as stream:
            reference = pickle.load(stream)
        X = np.asarray(payload["X"])
        Y = np.asarray(reference["X"])
        if X.ndim != 2 or X.shape[1] != len(names) or list(payload["feature_names"]) != names:
            raise ValueError(domain + ": Stage 2 feature names or dimensions differ")
        if len(X) != len(Y) or not np.isfinite(X).all() or not np.allclose(
            X[:, indices], Y, rtol=0, atol=1e-6):
            raise ValueError(domain + ": Stage 2 selected columns differ from Stage 3")
        for key in ("y_soh", "battery_id", "cycle", "condition_group"):
            left, right = np.asarray(payload[key]), np.asarray(reference[key])
            same = (np.allclose(left.astype(float), right.astype(float), rtol=0, atol=1e-6)
                    if key in ("y_soh", "cycle") else np.array_equal(left.astype(str), right.astype(str)))
            if not same:
                raise ValueError(domain + ": Stage 2/3 " + key + " differs")
        if bool(payload.get("standardized")):
            raise ValueError(domain + ": Stage 2 features were already standardized")
        datasets[domain] = payload
        print("Full features:", domain, X.shape, flush=True)
    return datasets, names


def manifest_splits(manifest: pd.DataFrame, stage4: Any) -> Dict[str, Dict[str, set[str]]]:
    if manifest.duplicated(["dataset", "battery_id"]).any():
        raise ValueError("Stage 4 split manifest contains duplicate battery ids")
    expected = {"source_train", "source_validation", "target_adaptation_train",
                "target_validation", "target_test"}
    actual = set(manifest["split"].astype(str).unique())
    if not expected.issubset(actual):
        raise ValueError("Stage 4 manifest is missing essential splits: " + str(expected - actual))
    return {domain: {name: set(manifest.loc[
        manifest["dataset"].eq(domain) & manifest["split"].eq(name),
        "battery_id"].astype(str)) for name in expected}
        for domain in stage4.DATASET_ORDER}


def reference_predictions(run: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for target in TARGETS:
        frames[target] = {}
        for split, stem in (("validation", "validation"), ("test", "test")):
            frame = pd.read_csv(run / "predictions" / target /
                                (stem + "_predictions.csv"), encoding="utf-8-sig")
            required = set(KEYS) | {"condition_group", "history_start_cycle",
                                     "history_length", "y_true", *MODELS,
                                     "Adaptive_Ensemble"}
            if not required.issubset(frame.columns) or frame.duplicated(list(KEYS)).any():
                raise ValueError(target + " Stage 4 " + split + " predictions invalid")
            frames[target][split] = frame
    return frames


def validate_paired(reference: pd.DataFrame, other: pd.DataFrame,
                    description: str) -> None:
    if len(reference) != len(other):
        raise ValueError(description + ": window counts differ")
    for key in (*KEYS, "condition_group", "history_start_cycle", "history_length"):
        if not np.array_equal(reference[key].astype(str).to_numpy(),
                              other[key].astype(str).to_numpy()):
            raise ValueError(description + ": row order / " + key + " differs")
    if not np.allclose(reference["y_true"].to_numpy(float),
                       other["y_true"].to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError(description + ": true labels differ; no paired comparison is valid")


def align_frozen_branch(template: pd.DataFrame, frozen: pd.DataFrame,
                        name: str) -> pd.DataFrame:
    keys = list(KEYS)
    metadata = ["condition_group", "history_start_cycle", "history_length", "y_true"]
    if template.duplicated(keys).any() or frozen.duplicated(keys).any():
        raise ValueError("Duplicate windows when aligning frozen " + name)
    request = template[keys + metadata].copy()
    request.insert(0, "_row", np.arange(len(request)))
    aligned = request.merge(frozen[keys + metadata + [name]], on=keys, how="left",
                            validate="one_to_one", sort=False, indicator=True,
                            suffixes=("_current", "_frozen"))
    aligned = aligned.sort_values("_row", kind="stable").reset_index(drop=True)
    if len(aligned) != len(template) or not aligned["_merge"].eq("both").all():
        raise ValueError("Frozen branch " + name + " has missing window keys")
    for key in metadata[:-1]:
        if not np.array_equal(aligned[key + "_current"].astype(str).to_numpy(),
                              aligned[key + "_frozen"].astype(str).to_numpy()):
            raise ValueError("Frozen branch " + name + " differs in " + key)
    if not np.allclose(aligned["y_true_current"].to_numpy(float),
                       aligned["y_true_frozen"].to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError("Frozen branch " + name + " has different true labels")
    if not np.isfinite(aligned[name].to_numpy(float)).all():
        raise ValueError("Frozen branch " + name + " contains invalid predictions")
    result = template.copy()
    result["y_pred"] = aligned[name].to_numpy(float)
    return result


def build_model(stage4: Any, name: str, feature_count: int) -> Any:
    if name == NEURAL[0]:
        return stage4.TemporalCNNBiLSTMAttention(feature_count)
    if name == NEURAL[1]:
        return stage4.CurrentCycleFeatureAttentionMLP(feature_count)
    raise KeyError(name)


def full_window_features(dataset: Any, dimension: int) -> pd.DataFrame:
    columns = ["f_%s_%02d" % (kind, index)
               for kind in ("current", "mean5", "change20")
               for index in range(dimension)]
    blocks: List[pd.DataFrame] = []
    offset = 0
    for record in dataset.records:
        windows = np.asarray(record["windows"], dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (20, dimension):
            raise ValueError("39-feature window shape is inconsistent")
        values = np.concatenate((windows[:, -1], windows[:, -5:].mean(axis=1),
                                 windows[:, -1] - windows[:, 0]), axis=1)
        block = pd.DataFrame(values, columns=columns)
        count = len(block)
        block.insert(0, "window_index", np.arange(offset, offset + count))
        block.insert(0, "y_true", record["y"][19:].astype(float))
        block.insert(0, "cycle", record["cycle"][19:].astype(int))
        block.insert(0, "battery_id", str(record["battery_id"]))
        block.insert(0, "dataset", str(record["dataset"]))
        blocks.append(block)
        offset += count
    result = pd.concat(blocks, ignore_index=True)
    if len(result) != len(dataset) or result["window_index"].duplicated().any():
        raise ValueError("tabular summary does not match window dataset")
    return result


def fit_full_tabular(stage4: Any, dataset: Any, dimension: int,
                     name: str, seed: int) -> Any:
    frame = full_window_features(dataset, dimension)
    columns = [column for column in frame if column.startswith("f_")]
    count = frame.groupby(["dataset", "battery_id"])["battery_id"].transform("size")
    weights = len(frame) / frame["battery_id"].nunique() / count.to_numpy(float)
    if name == stage4.HGB_MODEL_NAME:
        model = HistGradientBoostingRegressor(**stage4.HGB_SETTINGS, random_state=seed)
    elif name == stage4.RIDGE_MODEL_NAME:
        model = Ridge(alpha=20.0, solver="lsqr", random_state=seed)
    else:
        raise KeyError(name)
    print("Fitting", name, "on", len(frame), "full-feature windows", flush=True)
    model.fit(frame[columns].to_numpy(np.float32),
              frame["y_true"].to_numpy(float), sample_weight=weights)
    return model


def full_tabular_prediction(stage4: Any, template: pd.DataFrame,
                            dataset: Any, dimension: int, model: Any) -> pd.DataFrame:
    features = full_window_features(dataset, dimension)
    aligned = stage4.align_window_features(template, features)
    columns = [name for name in aligned if name.startswith("f_")]
    result = template.copy()
    result["y_pred"] = model.predict(aligned[columns].to_numpy(np.float32))
    if not np.isfinite(result["y_pred"].to_numpy(float)).all():
        raise ValueError("full-feature tabular prediction is not finite")
    return result


def train_one(stage4: Any, model: Any, train: Any, validation: Any,
              checkpoint: Path, phase: str, name: str, seed: int,
              source: bool = False) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    history_path = checkpoint.with_suffix(".history.csv")
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved.get("script_build") != STAGE6_BUILD or
            saved.get("stage4_training_implementation") != STAGE4_BUILD or
            saved.get("model_name") != name or saved.get("phase") != phase):
            raise ValueError("Cannot resume checkpoint with different provenance: " + str(checkpoint))
        model.load_state_dict(saved["state_dict"], strict=True)
        model = model.to(stage4.DEVICE)
        model.eval()
        if history_path.is_file():
            history = pd.read_csv(history_path, encoding="utf-8-sig").to_dict("records")
        else:
            history = []
        info = {"best_epoch": int(saved["epoch"]),
                "best_validation_loss": float(saved["validation_loss"]),
                "epochs_completed": len(history) if history else None,
                "checkpoint": str(checkpoint), "resumed_from_checkpoint": True,
                "history_available": bool(history)}
        print(phase, name, "checkpoint restored; training skipped", flush=True)
        return model, history, info
    print(phase, name, "training started; windows", len(train), len(validation),
          flush=True)
    result = stage4.train_model(
        model, train, validation,
        stage4.MAX_EPOCHS_SOURCE if source else stage4.TARGET_MAX_EPOCHS_BY_MODEL[name],
        stage4.PATIENCE_SOURCE if source else stage4.PATIENCE_TARGET,
        stage4.SOURCE_LEARNING_RATE if source else stage4.TARGET_LEARNING_RATE,
        checkpoint, phase, name, seed)


    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["script_build"] = STAGE6_BUILD
    payload["stage4_training_implementation"] = STAGE4_BUILD
    torch.save(payload, checkpoint)
    save_csv(pd.DataFrame(result[1]), history_path)
    result[2]["resumed_from_checkpoint"] = False
    result[2]["history_available"] = True
    print(phase, name, "best epoch", result[2]["best_epoch"], flush=True)
    return result


def train_neural_control(stage4: Any, datasets: Mapping[str, Any],
                         names: Sequence[str], splits: Mapping[str, Any],
                         run: Path, case: str, references: Mapping[str, Any],
                         pretrain: bool) -> Tuple[Dict[str, Dict[str, pd.DataFrame]],
                                                  List[Dict[str, Any]], List[Dict[str, Any]]]:
    dim = len(names)
    standardizer = stage4.fit_source_standardizer(datasets, splits, names)
    stage4.write_pickle(standardizer.to_serializable_dict(),
                        run / "controls" / case / "source_standardizer.pkl")
    history: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    states: Dict[str, Any] = {}
    if pretrain:
        train = stage4.BatteryWindowDataset(datasets, {
            d: splits[d]["source_train"] for d in stage4.SOURCE_DOMAINS},
            standardizer, stage4.SEQUENCE_LENGTH)
        val = stage4.BatteryWindowDataset(datasets, {
            d: splits[d]["source_validation"] for d in stage4.SOURCE_DOMAINS},
            standardizer, stage4.SEQUENCE_LENGTH)
        for index, name in enumerate(NEURAL):
            seed = stage4.RANDOM_SEED + index
            stage4.set_seed(seed)
            model, rows, info = train_one(stage4, build_model(stage4, name, dim),
                train, val, run / "controls" / case / "source" / (name + ".pth"),
                case + "_source_pretraining", name, seed, source=True)
            states[name] = stage4.cpu_state_dict(model)
            history.extend({**row, "case": case, "target": "source"} for row in rows)
            summaries.append({**info, "case": case, "phase": "source_pretraining",
                              "target": "source", "model": name})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del train, val

    output: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ti, target in enumerate(TARGETS):
        train = stage4.BatteryWindowDataset(datasets, {
            target: splits[target]["target_adaptation_train"]},
            standardizer, stage4.SEQUENCE_LENGTH)
        val = stage4.BatteryWindowDataset(datasets, {
            target: splits[target]["target_validation"]},
            standardizer, stage4.SEQUENCE_LENGTH)
        test = stage4.BatteryWindowDataset(datasets, {
            target: splits[target]["target_test"]},
            standardizer, stage4.SEQUENCE_LENGTH)
        predicted: Dict[str, Dict[str, pd.DataFrame]] = {"validation": {}, "test": {}}
        for index, name in enumerate(NEURAL):
            seed = stage4.RANDOM_SEED + 1000 * (ti + 1) + index
            stage4.set_seed(seed)
            model = build_model(stage4, name, dim)
            if pretrain:
                model.load_state_dict(states[name], strict=True)
            model, rows, info = train_one(stage4, model, train, val,
                run / "controls" / case / target / (name + ".pth"),
                case + "_target_" + target, name, seed)
            history.extend({**row, "case": case, "target": target} for row in rows)
            summaries.append({**info, "case": case, "phase": "target_training",
                              "target": target, "model": name})
            predicted["validation"][name] = stage4.predict_model(model, val)
            predicted["test"][name] = stage4.predict_model(model, test)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if pretrain:
            print(case, target, "fitting independent HGB and Ridge", flush=True)
            for name, seed_offset in ((MODELS[2], 12000), (MODELS[3], 13000)):
                path = run / "controls" / case / target / (name + ".pkl")
                if path.is_file():
                    with path.open("rb") as stream:
                        saved = pickle.load(stream)
                    if (saved.get("script_build") != STAGE6_BUILD or
                        saved.get("model_name") != name or saved.get("target") != target or
                        saved.get("feature_count") != dim):
                        raise ValueError("Cannot resume tabular model with different provenance")
                    model = saved["regressor"]
                    print(case, target, name, "checkpoint restored; fitting skipped",
                          flush=True)
                else:
                    model = fit_full_tabular(stage4, train, dim, name,
                                             stage4.RANDOM_SEED + seed_offset + ti)
                    stage4.write_pickle({"script_build": STAGE6_BUILD, "target": target,
                                         "model_name": name, "regressor": model,
                                         "feature_count": dim}, path)
                for split, dataset in (("validation", val), ("test", test)):
                    predicted[split][name] = full_tabular_prediction(
                        stage4, predicted[split][NEURAL[0]], dataset, dim, model)
        else:


            for split in ("validation", "test"):
                for name in MODELS[2:]:
                    frozen = references[target][split]
                    predicted[split][name] = align_frozen_branch(
                        predicted[split][NEURAL[0]], frozen, name)

        output[target] = {}
        for split in ("validation", "test"):
            merged = stage4.merge_model_predictions(predicted[split])
            validate_paired(references[target][split], merged,
                            "%s %s %s" % (case, target, split))
            output[target][split] = merged
        del train, val, test, predicted
    return output, history, summaries


def fitted_fusion(stage4: Any, target: str,
                  frames: Mapping[str, pd.DataFrame], run: Path,
                  case: str) -> Dict[str, pd.DataFrame]:
    policy, diagnostics = stage4.select_validation_ensemble(frames["validation"], target)
    path = run / "controls" / case / target / "fusion_policy.json"
    save_json({"script_build": STAGE6_BUILD, "case": case, "target": target,
               "selected_models": policy["selected_models"], "policy": policy},
              path)


    saved = read_json(path)["policy"]
    if not np.allclose(stage4.fusion_row_weights(frames["validation"], policy),
                       stage4.fusion_row_weights(frames["validation"], saved),
                       atol=1e-12, rtol=0):
        raise ValueError("Serialized fusion policy does not replay")
    save_csv(diagnostics, run / "controls" / case / target / "fusion_cv.csv")
    return {split: stage4.add_ensemble_prediction(frames[split], policy)
            for split in ("validation", "test")}


def battery_mean_saved_weights(frame: pd.DataFrame) -> np.ndarray:
    columns = ["fusion_weight_" + name for name in MODELS]
    if any(column not in frame.columns for column in columns):
        raise KeyError("Fusion predictions lack one or more weight columns")
    matrix = frame[columns].to_numpy(dtype=float)
    if (not np.isfinite(matrix).all() or (matrix < -1e-8).any() or
            not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-7)):
        raise ValueError("Saved validation weights are invalid")
    means = frame.groupby(["dataset", "battery_id"], sort=True)[columns].mean()
    if means.empty:
        raise ValueError("No validation batteries for ensemble weight summary")
    return means.to_numpy(dtype=float).mean(axis=0)


def case_metric_rows(stage4: Any, target: str, case: str,
                     frame: pd.DataFrame, prediction_columns: Sequence[str]) -> List[Dict[str, Any]]:
    return [{**row, "case": case, "method": row["model"]}
            for row in stage4.metric_rows(frame, target, "final_test", prediction_columns)]


def paired_bootstrap(reference: pd.DataFrame, comparison: pd.DataFrame,
                     target: str, experiment: str, reference_name: str,
                     comparator_name: str, seed: int) -> List[Dict[str, Any]]:
    validate_paired(reference, comparison, experiment + " " + target)
    a = reference["Adaptive_Ensemble"].to_numpy(float)
    b = comparison["Adaptive_Ensemble"].to_numpy(float) if comparator_name in (
        "AllFeatures_Transfer_Adaptive", "Without_Source_Pretraining") else comparison[
        comparator_name].to_numpy(float)
    y = reference["y_true"].to_numpy(float)
    identifiers = reference["battery_id"].astype(str).to_numpy()
    groups = [np.flatnonzero(identifiers == battery) for battery in np.unique(identifiers)]
    sq_a = np.square(a - y)
    sq_b = np.square(b - y)
    pooled_a = np.asarray([sq_a[idx].sum() for idx in groups])
    pooled_b = np.asarray([sq_b[idx].sum() for idx in groups])
    counts = np.asarray([len(idx) for idx in groups])
    battery_a = np.sqrt(pooled_a / counts)
    battery_b = np.sqrt(pooled_b / counts)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(groups), size=(BOOTSTRAPS, len(groups)))
    pooled_effect = (np.sqrt(pooled_b[samples].sum(axis=1) / counts[samples].sum(axis=1))
                     - np.sqrt(pooled_a[samples].sum(axis=1) / counts[samples].sum(axis=1)))
    macro_effect = battery_b[samples].mean(axis=1) - battery_a[samples].mean(axis=1)
    observed = {
        "pooled_cycles": (np.sqrt(sq_b.mean()) - np.sqrt(sq_a.mean()), pooled_effect),
        "battery_macro": (battery_b.mean() - battery_a.mean(), macro_effect),
    }
    return [{"experiment": experiment, "target": target,
             "reference_method": reference_name, "comparator_method": comparator_name,
             "evaluation_scope": scope, "observed_rmse_difference": float(value),
             "rmse_difference_ci95_lower": float(np.percentile(draws, 2.5)),
             "rmse_difference_ci95_upper": float(np.percentile(draws, 97.5)),
             "bootstrap_replicates": BOOTSTRAPS, "resampling_unit": "paired_battery",
             "positive_means_reference_better": True}
            for scope, (value, draws) in observed.items()]


def write_report_zip(folder: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(folder))


def select_output_run(stage4_run: Path, stage3_run: Path, stage2_run: Path,
                      selected_count: int, full_count: int, sequence_length: int) -> Tuple[Path, bool]:
    explicit = os.environ.get("CRPS_STAGE6_RESUME_RUN")
    candidates = ([Path(explicit)] if explicit else
                  sorted(OUTPUT_ROOT.glob("run_*"), reverse=True))
    for run in candidates:
        try:
            config = read_json(run / "validation_report" / "processing_config.json")
            expected = {"script_build": STAGE6_BUILD, "stage4_run": str(stage4_run),
                        "stage3_run": str(stage3_run), "stage2_run": str(stage2_run),
                        "selected_feature_count": selected_count,
                        "full_feature_count": full_count,
                        "sequence_length": sequence_length}
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError("checkpoint lineage does not match")
            summary = run / "validation_report" / "processing_summary.json"
            if summary.is_file() and read_json(summary).get("validation_status") == "PASS":
                if explicit:
                    raise ValueError("selected Stage 6 run already passed")
                continue
            if not any((run / "controls").rglob("*.pth")):
                raise ValueError("no trained checkpoints to resume")
            print("Resuming incomplete Stage 6 run:", run, flush=True)
            return run, True
        except (OSError, ValueError, KeyError) as exc:
            if explicit:
                raise ValueError("CRPS_STAGE6_RESUME_RUN is incompatible: " + str(exc)) from exc
    if explicit:
        raise FileNotFoundError("Specified Stage 6 resume directory does not exist")
    return OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"), False


def finalize_report(stage4: Any, original: Path, run: Path, report: Path,
                    full_case: str, full_names: Sequence[str], references: Mapping[str, Any],
                    full: Mapping[str, Any], scratch: Mapping[str, Any],
                    full_summary: List[Dict[str, Any]], scratch_summary: List[Dict[str, Any]],
                    full_history: List[Dict[str, Any]], scratch_history: List[Dict[str, Any]],
                    resumed: bool) -> None:
    print("\n[3/3] Final-test metrics and paired battery bootstrap", flush=True)
    feature_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    fusion_rows: List[Dict[str, Any]] = []
    effects: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    control_weight_rows: List[Dict[str, Any]] = []
    for ti, target in enumerate(TARGETS):
        ref = references[target]["test"].copy()
        all_frame = full[target]["test"]
        scratch_frame = scratch[target]["test"]
        validate_paired(ref, all_frame, target + " feature ablation")
        validate_paired(ref, scratch_frame, target + " pretraining ablation")
        for case, frame in ((SELECTED_CASE, ref), (full_case, all_frame)):
            rows = case_metric_rows(stage4, target, case, frame, ["Adaptive_Ensemble"])
            for row in rows:
                row["method"] = ("Selected20_Transfer_Adaptive" if case == SELECTED_CASE
                                 else "All%d_Transfer_Adaptive" % len(full_names))
            feature_rows.extend(rows)
        for case, frame in ((SELECTED_CASE, ref), (SCRATCH_CASE, scratch_frame)):
            rows = case_metric_rows(stage4, target, case, frame, ["Adaptive_Ensemble"])
            for row in rows:
                row["method"] = ("With_Source_Pretraining" if case == SELECTED_CASE
                                 else "Without_Source_Pretraining")
            transfer_rows.extend(rows)

        policy = read_json(original / "checkpoints" / "targets" / target /
                           "fusion_policy.json")
        if "policy" in policy:
            policy = policy["policy"]
        chosen = list(policy["selected_models"])
        if not chosen or not set(chosen).issubset(MODELS):
            raise ValueError(target + ": invalid validation-selected branches")
        avg = ref.copy()
        avg["Equal_Weight_Selected_Branches"] = avg[chosen].mean(axis=1)
        avg["Equal_Weight_All_Four"] = avg[list(MODELS)].mean(axis=1)
        fusion_rows.extend(case_metric_rows(stage4, target, SELECTED_CASE, avg,
            list(MODELS) + ["Equal_Weight_Selected_Branches",
                            "Equal_Weight_All_Four", "Adaptive_Ensemble"]))
        best_val = stage4.metric_rows(references[target]["validation"], target,
                                      "validation", list(MODELS))
        best_single = min((row for row in best_val if row["evaluation_scope"] ==
                           "battery_macro"), key=lambda row: row["RMSE"])["model"]
        effects.extend(paired_bootstrap(ref, all_frame, target,
            "feature_selection", SELECTED_CASE, "AllFeatures_Transfer_Adaptive",
            450 + ti))
        effects.extend(paired_bootstrap(ref, scratch_frame, target,
            "source_pretraining", SELECTED_CASE, "Without_Source_Pretraining",
            550 + ti))
        for comparison in ("Equal_Weight_Selected_Branches", best_single):
            effects.extend(paired_bootstrap(ref, avg, target, "ensemble_fusion",
                "Adaptive_Ensemble", comparison, 650 + ti))
        for case, frame in ((SELECTED_CASE, ref), (full_case, all_frame),
                            (SCRATCH_CASE, scratch_frame)):
            save_csv(frame, report / (target + "_" + case + "_test_predictions.csv"))
        for case, frame in ((full_case, full[target]["validation"]),
                            (SCRATCH_CASE, scratch[target]["validation"])):
            save_csv(frame, run / "controls" / case / target / "validation_predictions.csv")
        weight_data = pd.read_csv(original / "metrics" / "condition_ensemble_weights.csv",
                                  encoding="utf-8-sig")
        baseline_weights = weight_data.loc[weight_data["target"].eq(target)].copy()
        baseline_weights["case"] = SELECTED_CASE
        weight_rows.extend(baseline_weights.to_dict("records"))
        for case, result in ((full_case, full), (SCRATCH_CASE, scratch)):
            saved = read_json(run / "controls" / case / target / "fusion_policy.json")
            if (saved.get("script_build") != STAGE6_BUILD or saved.get("case") != case or
                    saved.get("target") != target):
                raise ValueError("Fusion policy provenance differs from this ablation")
            policy_control = saved["policy"]
            validation_frame = result[target]["validation"]
            mean_weights = battery_mean_saved_weights(validation_frame)
            for name, weight in zip(MODELS, mean_weights):
                control_weight_rows.append({"case": case, "target": target,
                    "model": name, "weight": float(weight),
                    "selected": name in policy_control["selected_models"],
                    "fusion_mode": policy_control["mode"],
                    "weight_scope": "validation_battery_mean",
                    "learned_on": "target_validation_batteries", "test_labels_used": False})
        print(target, "selected Stage 4 branches", chosen,
              "validation best single", best_single, flush=True)

    save_csv(pd.DataFrame(feature_rows), report / "feature_selection_ablation_metrics.csv")
    save_csv(pd.DataFrame(transfer_rows), report / "source_pretraining_ablation_metrics.csv")
    save_csv(pd.DataFrame(fusion_rows), report / "ensemble_fusion_ablation_metrics.csv")
    save_csv(pd.DataFrame(effects), report / "paired_battery_bootstrap_effects.csv")
    save_csv(pd.DataFrame(weight_rows), report / "ensemble_weights.csv")
    save_csv(pd.DataFrame(control_weight_rows), report / "control_ensemble_weights.csv")
    summary_rows = full_summary + scratch_summary
    missing_history = [item for item in summary_rows if not item["history_available"]]
    history_columns = ["case", "target", "phase", "model", "epoch", "train_mse",
                       "validation_battery_domain_balanced_mse", "learning_rate"]
    save_csv(pd.DataFrame(full_history + scratch_history).reindex(columns=history_columns),
             report / "training_history.csv")
    save_csv(pd.DataFrame(summary_rows), report / "training_summary.csv")
    save_json({"validation_status": "PASS", "integrity_checks": {
        "stage4_accepted": True, "stage2_stage3_feature_and_label_alignment": True,
        "identical_final_test_window_keys": True,
        "fusion_trained_on_validation_only": True,
        "bootstrap_battery_paired": True},
        "hypothesis_outcomes_are_reported_in_metrics_not_used_for_PASS": True,
        "resumed_incomplete_run": resumed,
        "training_histories_unavailable_for_resumed_checkpoints": [
            {"case": item["case"], "target": item["target"], "model": item["model"]}
            for item in missing_history]},
        report / "processing_summary.json")
    archive = run / "STAGE6_VALIDATION_REPORT.zip"
    write_report_zip(report, archive)
    print("\nValidation status: PASS", flush=True)
    print(pd.DataFrame(effects)[["experiment", "target", "evaluation_scope",
          "observed_rmse_difference", "rmse_difference_ci95_lower",
          "rmse_difference_ci95_upper"]].to_string(index=False), flush=True)
    print("Validation report:", archive, flush=True)
    print("Note: older Figure 6 plotting code needs updated model names and cases.", flush=True)


def main() -> None:
    print("=" * 68, "\nStage 6 V6.0 matched conditional ablations started\nBuild:",
          STAGE6_BUILD, "\nDevice:", "cuda" if torch.cuda.is_available() else "cpu",
          "\n" + "=" * 68, flush=True)
    stage4 = import_matching_stage4()
    original, config = accepted_stage4_run()
    stage3_run = Path(str(config["stage3_run"]))
    stage2_run = stage2_run_for_lineage(stage3_run)
    selected_names = stage4.read_selected_feature_names(stage3_run)
    selected_datasets = stage4.load_all_stage3_datasets(stage3_run, selected_names)
    full_datasets, full_names = load_full_features(stage2_run, stage3_run,
                                                   selected_names, stage4)
    manifest = pd.read_csv(original / "battery_split_manifest.csv", encoding="utf-8-sig")
    splits = manifest_splits(manifest, stage4)
    references = reference_predictions(original)
    full_case = "All%d_With_Source_Pretraining" % len(full_names)
    print("Stage 4 accepted:", original, "\nStage 2:", stage2_run,
          "\nFeature counts:", len(selected_names), len(full_names), flush=True)

    run, resumed = select_output_run(original, stage3_run, stage2_run,
                                     len(selected_names), len(full_names),
                                     stage4.SEQUENCE_LENGTH)
    report = run / "validation_report"
    report.mkdir(parents=True, exist_ok=resumed)
    save_csv(manifest, report / "battery_split_manifest.csv")
    save_json({"schema_version": "6.0", "script_build": STAGE6_BUILD,
               "stage4_run": str(original), "stage4_build": STAGE4_BUILD,
               "stage3_run": str(stage3_run), "stage2_run": str(stage2_run),
               "selected_feature_count": len(selected_names),
               "full_feature_count": len(full_names), "sequence_length": stage4.SEQUENCE_LENGTH,
               "target_test_used_for_training_or_selection": False,
               "stage4_reference_predictions_reused": True,
               "tabular_models_independent_of_source_pretraining": True,
               "control_fusion_learned_on": "target validation batteries only",
               "evaluation_scopes": ["pooled_cycles", "battery_macro"],
               "bootstrap_resampling_unit": "battery"},
              report / "processing_config.json")

    print("\n[1/3] Full-feature source pretraining and target controls", flush=True)
    full_raw, full_history, full_summary = train_neural_control(
        stage4, full_datasets, full_names, splits, run, full_case, references, True)
    full = {target: fitted_fusion(stage4, target, full_raw[target], run, full_case)
            for target in TARGETS}
    print("\n[2/3] Selected-feature target-only scratch controls", flush=True)
    scratch_raw, scratch_history, scratch_summary = train_neural_control(
        stage4, selected_datasets, selected_names, splits, run, SCRATCH_CASE,
        references, False)
    scratch = {target: fitted_fusion(stage4, target, scratch_raw[target], run,
                                    SCRATCH_CASE) for target in TARGETS}

    finalize_report(stage4, original, run, report, full_case, full_names,
                    references, full, scratch, full_summary, scratch_summary,
                    full_history, scratch_history, resumed)

if __name__ == "__main__":
    main()
