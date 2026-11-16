from __future__ import annotations

import json
import os
import pickle
import random
import zipfile
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.optimize import minimize


STAGE4_SCHEMA_VERSION = "4.0.2"
SCRIPT_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"

EXPECTED_STAGE3_SCHEMA = "3.0.1"
EXPECTED_STAGE3_BUILD = "STAGE3_V3_0_1_FINAL_20260901"

BASE_PATH = Path(os.environ.get("CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"))
STAGE3_ROOT = BASE_PATH / "FEATURESELECTION_V3_0_1"
OUTPUT_ROOT = BASE_PATH / "SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION"


STAGE3_RUN_PATH: Optional[Path] = None

DATASET_ORDER = ("XJTU", "HUST", "TJU_NCM", "TJU_NCA")
SOURCE_DOMAINS = ("XJTU", "HUST")
TARGET_DOMAINS = ("TJU_NCM", "TJU_NCA")

FEATURE_COUNT = 20
SEQUENCE_LENGTH = 20

SOURCE_TRAIN_RATIO = 0.80
TARGET_TRAIN_RATIO = 0.60
TARGET_VALIDATION_RATIO = 0.20
TARGET_TEST_RATIO = 0.20

BATCH_SIZE = 512
MAX_EPOCHS_SOURCE = 100
MAX_EPOCHS_TARGET = 100
TARGET_MAX_EPOCHS_BY_MODEL = {
    "CNN_BiLSTM_Attention": MAX_EPOCHS_TARGET,


    "MLP_Feature_Attention": 250,
}
PATIENCE_SOURCE = 12
PATIENCE_TARGET = 8

SOURCE_LEARNING_RATE = 1e-4
TARGET_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0

HGB_MODEL_NAME = "HGB_Window_SOH"
RIDGE_MODEL_NAME = "Ridge_Window_SOH"
HGB_SETTINGS = {
    "max_iter": 140,
    "learning_rate": 0.035,
    "max_leaf_nodes": 15,
    "max_depth": 3,
    "min_samples_leaf": 80,
    "l2_regularization": 0.5,
    "max_bins": 64,
    "early_stopping": False,
}
IMPLEMENTATION_REVISION = "STAGE4_VALIDATION_CONDITION_FUSION_R2_20261006"

RANDOM_SEED = 42
NUM_WORKERS = 0
USE_MIXED_PRECISION = True

NEURAL_MODEL_NAMES = (
    "CNN_BiLSTM_Attention",
    "MLP_Feature_Attention",
)
MODEL_NAMES = NEURAL_MODEL_NAMES + (HGB_MODEL_NAME, RIDGE_MODEL_NAME)
MAX_FUSION_BRANCHES = 3


MIN_VALIDATION_GAIN_FRACTION = 0.005
MIN_MEAN_BRANCH_WEIGHT = 0.05
MIN_GROUP_TRAIN_BATTERIES = 3
GROUP_PRIOR_BATTERY_EQUIVALENTS = 1.0
MIN_CONDITION_GAIN_FRACTION = 0.002
FUSION_MACRO_SHARE_OPTIONS = (0.2, 0.5)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("not JSON serializable: " + type(value).__name__)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def zip_validation_report(report_path: Path, zip_path: Path) -> None:
    files = sorted(path for path in report_path.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(report_path)))


def find_latest_stage3_run() -> Path:
    if STAGE3_RUN_PATH is not None:
        candidates = [Path(STAGE3_RUN_PATH)]
    else:
        if not STAGE3_ROOT.exists():
            raise FileNotFoundError("Stage 3 root does not exist: " + str(STAGE3_ROOT))
        candidates = sorted(
            [path for path in STAGE3_ROOT.glob("run_*") if path.is_dir()],
            reverse=True,
        )

    rejection_reasons: List[str] = []
    for candidate in candidates:
        report_path = candidate / "validation_report"
        config_path = report_path / "processing_config.json"
        summary_path = report_path / "processing_summary.json"
        selected_path = candidate / "selected_features.txt"
        dataset_path = candidate / "datasets"

        required = [config_path, summary_path, selected_path, dataset_path]
        if any(not path.exists() for path in required):
            rejection_reasons.append(str(candidate) + ": missing required files")
            continue
        try:
            config = read_json(config_path)
            summary = read_json(summary_path)
        except Exception as exc:
            rejection_reasons.append(str(candidate) + ": invalid JSON: " + str(exc))
            continue

        if str(config.get("schema_version")) != EXPECTED_STAGE3_SCHEMA:
            rejection_reasons.append(str(candidate) + ": unexpected schema")
            continue
        if str(config.get("script_build")) != EXPECTED_STAGE3_BUILD:
            rejection_reasons.append(str(candidate) + ": unexpected build")
            continue
        if str(summary.get("validation_status")) != "PASS":
            rejection_reasons.append(str(candidate) + ": validation is not PASS")
            continue
        if summary.get("fatal_checks") or summary.get("warnings"):
            rejection_reasons.append(str(candidate) + ": checks or warnings remain")
            continue
        if int(summary.get("selected_feature_count", -1)) != FEATURE_COUNT:
            rejection_reasons.append(str(candidate) + ": feature count is not 20")
            continue
        if bool(summary.get("target_labels_used_for_selection")):
            rejection_reasons.append(str(candidate) + ": target labels entered selection")
            continue
        if float(summary.get("maximum_pairwise_redundancy_among_selected", 1.0)) >= 0.99:
            rejection_reasons.append(str(candidate) + ": redundancy threshold failed")
            continue
        if int(summary.get("redundancy_selection_success_iterations", -1)) != 300:
            rejection_reasons.append(str(candidate) + ": bootstrap selection incomplete")
            continue

        missing = [
            dataset
            for dataset in DATASET_ORDER
            if not (dataset_path / (dataset + "_selected_features.pkl")).exists()
        ]
        if missing:
            rejection_reasons.append(str(candidate) + ": missing datasets " + str(missing))
            continue
        return candidate

    detail = "\n".join(rejection_reasons[:20])
    raise FileNotFoundError(
        "No accepted Stage 3 V3.0.1 FINAL run was found.\n" + detail
    )


def read_selected_feature_names(stage3_run: Path) -> List[str]:
    path = stage3_run / "selected_features.txt"
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(names) != FEATURE_COUNT or len(set(names)) != FEATURE_COUNT:
        raise ValueError("selected_features.txt must contain 20 unique features")
    return names


def validate_stage3_payload(
    dataset: str,
    payload: Mapping[str, Any],
    feature_names: Sequence[str],
) -> None:
    required = {
        "schema_version",
        "script_build",
        "dataset",
        "feature_names",
        "feature_count",
        "standardized",
        "target_labels_used_for_selection",
        "X",
        "y_soh",
        "battery_id",
        "cycle",
        "condition_group",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(dataset + " payload missing keys: " + str(missing))
    if str(payload["schema_version"]) != EXPECTED_STAGE3_SCHEMA:
        raise ValueError(dataset + " payload has an unexpected schema")
    if str(payload["script_build"]) != EXPECTED_STAGE3_BUILD:
        raise ValueError(dataset + " payload has an unexpected build")
    if str(payload["dataset"]) != dataset:
        raise ValueError(dataset + " payload dataset mismatch")
    if bool(payload["standardized"]):
        raise ValueError(dataset + " must be unstandardized before Stage 4")
    if bool(payload["target_labels_used_for_selection"]):
        raise ValueError(dataset + " reports target-label feature-selection leakage")
    if [str(name) for name in payload["feature_names"]] != list(feature_names):
        raise ValueError(dataset + " feature names/order differ from Stage 3")

    X = np.asarray(payload["X"])
    y = np.asarray(payload["y_soh"])
    battery_id = np.asarray(payload["battery_id"], dtype=object)
    cycle = np.asarray(payload["cycle"])
    condition = np.asarray(payload["condition_group"], dtype=object)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT:
        raise ValueError(dataset + " X must have shape (N, 20)")
    if not (len(X) == len(y) == len(battery_id) == len(cycle) == len(condition)):
        raise ValueError(dataset + " sample arrays have inconsistent lengths")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(dataset + " contains non-finite X or SOH")
    cycle_float = cycle.astype(np.float64)
    if not np.all(np.isfinite(cycle_float)) or not np.allclose(cycle_float, np.rint(cycle_float)):
        raise ValueError(dataset + " cycles are not finite integers")

    keys = pd.DataFrame({
        "battery_id": battery_id.astype(str),
        "cycle": cycle_float.astype(np.int64),
    })
    if keys.duplicated(["battery_id", "cycle"]).any():
        raise ValueError(dataset + " contains duplicate battery-cycle rows")


def load_all_stage3_datasets(
    stage3_run: Path,
    feature_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    datasets: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        path = stage3_run / "datasets" / (dataset + "_selected_features.pkl")
        print("Loading:", path)
        payload = load_pickle(path)
        validate_stage3_payload(dataset, payload, feature_names)
        datasets[dataset] = dict(payload)
        print(
            dataset,
            "X=", np.asarray(payload["X"]).shape,
            "batteries=", len(set(np.asarray(payload["battery_id"]).astype(str))),
        )
    return datasets


def most_frequent_text(values: np.ndarray) -> str:
    text = pd.Series(values.astype(str))
    if len(text) == 0:
        return "unknown"
    mode = text.mode(dropna=False)
    return str(mode.iloc[0]) if len(mode) else "unknown"


def build_battery_table(
    datasets: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        payload = datasets[dataset]
        battery_ids = np.asarray(payload["battery_id"], dtype=object).astype(str)
        cycles = np.asarray(payload["cycle"], dtype=np.int64)
        conditions = np.asarray(payload["condition_group"], dtype=object)
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, battery_id in enumerate(battery_ids):
            groups[battery_id].append(index)

        for battery_id in sorted(groups):
            indices = np.asarray(groups[battery_id], dtype=np.int64)
            sample_count = len(indices)
            rows.append({
                "dataset": dataset,
                "battery_id": battery_id,
                "condition_group": most_frequent_text(conditions[indices]),
                "condition_group_count": int(len(set(conditions[indices].astype(str)))),
                "cycle_min": int(np.min(cycles[indices])),
                "cycle_max": int(np.max(cycles[indices])),
                "sample_count": sample_count,
                "window_count": max(0, sample_count - SEQUENCE_LENGTH + 1),
                "sequence_eligible": bool(sample_count >= SEQUENCE_LENGTH),
            })
    return pd.DataFrame(rows)


def optional_stratification(labels: Sequence[str], test_count: int) -> Optional[List[str]]:
    labels = [str(label) for label in labels]
    counts = pd.Series(labels).value_counts()
    class_count = len(counts)
    train_count = len(labels) - test_count
    if class_count <= 1:
        return None
    if int(counts.min()) < 2:
        return None
    if test_count < class_count or train_count < class_count:
        return None
    return labels


def split_source_batteries(
    table: pd.DataFrame,
    seed: int,
) -> Tuple[Set[str], Set[str], str]:
    eligible = table[table["sequence_eligible"]].copy()
    ids = eligible["battery_id"].astype(str).tolist()
    labels = eligible["condition_group"].astype(str).tolist()
    if len(ids) < 2:
        raise ValueError("source domain has fewer than two sequence-eligible batteries")
    validation_count = max(1, int(round(len(ids) * (1.0 - SOURCE_TRAIN_RATIO))))
    validation_count = min(validation_count, len(ids) - 1)
    stratify = optional_stratification(labels, validation_count)
    train_ids, validation_ids = train_test_split(
        ids,
        test_size=validation_count,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return set(train_ids), set(validation_ids), "stratified" if stratify else "random"


def split_target_batteries(
    table: pd.DataFrame,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str], str]:
    eligible = table[table["sequence_eligible"]].copy()
    ids = eligible["battery_id"].astype(str).tolist()
    labels = eligible["condition_group"].astype(str).tolist()
    if len(ids) < 3:
        raise ValueError("target domain has fewer than three sequence-eligible batteries")

    temporary_count = max(2, int(round(len(ids) * (1.0 - TARGET_TRAIN_RATIO))))
    temporary_count = min(temporary_count, len(ids) - 1)
    first_stratify = optional_stratification(labels, temporary_count)
    train_ids, temporary_ids, _, temporary_labels = train_test_split(
        ids,
        labels,
        test_size=temporary_count,
        random_state=seed,
        shuffle=True,
        stratify=first_stratify,
    )

    validation_fraction_of_temporary = (
        TARGET_VALIDATION_RATIO / (TARGET_VALIDATION_RATIO + TARGET_TEST_RATIO)
    )
    validation_count = max(
        1,
        int(round(len(temporary_ids) * validation_fraction_of_temporary)),
    )
    validation_count = min(validation_count, len(temporary_ids) - 1)
    second_stratify = optional_stratification(
        temporary_labels,
        len(temporary_ids) - validation_count,
    )
    validation_ids, test_ids = train_test_split(
        temporary_ids,
        train_size=validation_count,
        random_state=seed + 1,
        shuffle=True,
        stratify=second_stratify,
    )
    initial_policy = (
        "stratified"
        if first_stratify is not None and second_stratify is not None
        else "partially_stratified"
        if first_stratify is not None or second_stratify is not None
        else "random"
    )

    split_sets: Dict[str, Set[str]] = {
        "target_adaptation_train": set(train_ids),
        "target_validation": set(validation_ids),
        "target_test": set(test_ids),
    }
    condition_by_battery = dict(
        zip(
            eligible["battery_id"].astype(str),
            eligible["condition_group"].astype(str),
        )
    )
    repaired_conditions = enforce_target_condition_coverage(
        split_sets,
        condition_by_battery,
        seed + 2,
    )
    policy = (
        initial_policy + "+condition_coverage_repair"
        if repaired_conditions
        else initial_policy + "+condition_coverage_complete"
    )
    return (
        split_sets["target_adaptation_train"],
        split_sets["target_validation"],
        split_sets["target_test"],
        policy,
    )


def enforce_target_condition_coverage(
    split_sets: Dict[str, Set[str]],
    condition_by_battery: Mapping[str, str],
    seed: int,
) -> List[str]:
    split_names = (
        "target_adaptation_train",
        "target_validation",
        "target_test",
    )
    rng = np.random.default_rng(seed)
    condition_counts = pd.Series(list(condition_by_battery.values())).value_counts()
    required_conditions = sorted(
        str(condition)
        for condition, count in condition_counts.items()
        if int(count) >= len(split_names)
    )
    repaired: List[str] = []

    def batteries_for(split_name: str, condition: str) -> List[str]:
        return sorted(
            battery_id
            for battery_id in split_sets[split_name]
            if condition_by_battery[battery_id] == condition
        )

    for condition in required_conditions:
        for missing_split in split_names:
            if batteries_for(missing_split, condition):
                continue
            donor_candidates = [
                split_name
                for split_name in split_names
                if len(batteries_for(split_name, condition)) > 1
            ]
            if not donor_candidates:
                continue
            donor_candidates.sort(
                key=lambda split_name: (
                    -len(batteries_for(split_name, condition)),
                    split_name,
                )
            )
            donor_split = donor_candidates[0]
            donor_batteries = batteries_for(donor_split, condition)
            selected_index = int(rng.integers(0, len(donor_batteries)))
            selected_battery = donor_batteries[selected_index]
            split_sets[donor_split].remove(selected_battery)
            split_sets[missing_split].add(selected_battery)
            repaired.append(
                condition + ":" + donor_split + "->" + missing_split
            )
    return repaired


def create_all_splits(
    battery_table: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, str]]:
    splits: Dict[str, Dict[str, Set[str]]] = {}
    policies: Dict[str, str] = {}
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        table = battery_table[battery_table["dataset"] == dataset]
        seed = RANDOM_SEED + 100 * dataset_index
        if dataset in SOURCE_DOMAINS:
            train_ids, validation_ids, policy = split_source_batteries(table, seed)
            splits[dataset] = {
                "source_train": train_ids,
                "source_validation": validation_ids,
            }
        else:
            train_ids, validation_ids, test_ids, policy = split_target_batteries(
                table,
                seed,
            )
            splits[dataset] = {
                "target_adaptation_train": train_ids,
                "target_validation": validation_ids,
                "target_test": test_ids,
            }
        policies[dataset] = policy
    return splits, policies


def build_split_manifest(
    battery_table: pd.DataFrame,
    splits: Mapping[str, Mapping[str, Set[str]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in battery_table.to_dict(orient="records"):
        dataset = str(row["dataset"])
        battery_id = str(row["battery_id"])
        assigned = "excluded_short_history"
        for split_name, battery_ids in splits[dataset].items():
            if battery_id in battery_ids:
                assigned = split_name
                break
        output = dict(row)
        output.update({
            "split": assigned,
            "used_for_scaler": assigned == "source_train",
            "used_for_source_pretraining": assigned == "source_train",
            "used_for_early_stopping": assigned in {
                "source_validation",
                "target_validation",
            },
            "used_for_target_finetuning": assigned == "target_adaptation_train",
            "used_for_ensemble_weight_learning": assigned == "target_validation",
            "used_for_final_test": assigned == "target_test",
        })
        rows.append(output)
    return pd.DataFrame(rows)


class FrozenStandardizer:
    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        feature_names: Sequence[str],
        fitted_datasets: Sequence[str],
        fitted_batteries: Mapping[str, Sequence[str]],
    ):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.fitted_datasets = list(fitted_datasets)
        self.fitted_batteries = {
            str(key): list(value) for key, value in fitted_batteries.items()
        }
        self.fitted_on_source_train_only = True
        self.weighting = "equal domain weight and equal battery weight within domain"

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        transformed = (X - self.mean_) / self.scale_
        if not np.all(np.isfinite(transformed)):
            raise ValueError("standardizer produced non-finite values")
        return transformed.astype(np.float32)

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": STAGE4_SCHEMA_VERSION,
            "script_build": SCRIPT_BUILD,
            "mean": self.mean_,
            "scale": self.scale_,
            "feature_names": self.feature_names,
            "fitted_datasets": self.fitted_datasets,
            "fitted_batteries": self.fitted_batteries,
            "fitted_on_source_train_only": self.fitted_on_source_train_only,
            "weighting": self.weighting,
        }


def battery_indices(payload: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, battery_id in enumerate(np.asarray(payload["battery_id"]).astype(str)):
        groups[str(battery_id)].append(index)
    return {
        battery_id: np.asarray(indices, dtype=np.int64)
        for battery_id, indices in groups.items()
    }


def fit_source_standardizer(
    datasets: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, Mapping[str, Set[str]]],
    feature_names: Sequence[str],
) -> FrozenStandardizer:
    domain_means: List[np.ndarray] = []
    domain_second_moments: List[np.ndarray] = []
    fitted_batteries: Dict[str, List[str]] = {}

    for dataset in SOURCE_DOMAINS:
        payload = datasets[dataset]
        X = np.asarray(payload["X"], dtype=np.float64)
        groups = battery_indices(payload)
        train_ids = sorted(splits[dataset]["source_train"])
        fitted_batteries[dataset] = train_ids
        battery_means: List[np.ndarray] = []
        battery_seconds: List[np.ndarray] = []
        for battery_id in train_ids:
            values = X[groups[battery_id]]
            battery_means.append(np.mean(values, axis=0))
            battery_seconds.append(np.mean(np.square(values), axis=0))
        if not battery_means:
            raise ValueError(dataset + " has no source-training batteries")
        domain_means.append(np.mean(np.vstack(battery_means), axis=0))
        domain_second_moments.append(np.mean(np.vstack(battery_seconds), axis=0))

    mean = np.mean(np.vstack(domain_means), axis=0)
    second_moment = np.mean(np.vstack(domain_second_moments), axis=0)
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return FrozenStandardizer(
        mean,
        scale,
        feature_names,
        SOURCE_DOMAINS,
        fitted_batteries,
    )


class BatteryWindowDataset(Dataset):
    def __init__(
        self,
        datasets: Mapping[str, Mapping[str, Any]],
        selected_batteries: Mapping[str, Set[str]],
        standardizer: FrozenStandardizer,
        sequence_length: int,
    ):
        self.sequence_length = int(sequence_length)
        self.records: List[Dict[str, Any]] = []
        self.index_map: List[Tuple[int, int]] = []

        for dataset in DATASET_ORDER:
            if dataset not in selected_batteries:
                continue
            payload = datasets[dataset]
            X = np.asarray(payload["X"], dtype=np.float64)
            y = np.asarray(payload["y_soh"], dtype=np.float32)
            cycles = np.asarray(payload["cycle"], dtype=np.int64)
            battery_ids = np.asarray(payload["battery_id"]).astype(str)
            conditions = np.asarray(payload["condition_group"]).astype(str)
            groups = battery_indices(payload)

            for battery_id in sorted(selected_batteries[dataset]):
                indices = groups[battery_id]
                order = np.argsort(cycles[indices], kind="stable")
                indices = indices[order]
                if len(indices) < self.sequence_length:
                    continue
                record_index = len(self.records)
                transformed = standardizer.transform(X[indices])
                windows = np.lib.stride_tricks.sliding_window_view(
                    transformed, self.sequence_length, axis=0,
                ).transpose(0, 2, 1)
                self.records.append({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "condition_group": most_frequent_text(conditions[indices]),
                    "X": transformed,
                    "windows": np.ascontiguousarray(windows, dtype=np.float32),
                    "y": y[indices].astype(np.float32, copy=False),
                    "cycle": cycles[indices].astype(np.int64, copy=False),
                    "window_count": len(indices) - self.sequence_length + 1,
                })
                for end_index in range(self.sequence_length - 1, len(indices)):
                    self.index_map.append((record_index, end_index))

        if not self.index_map:
            raise ValueError("window dataset contains no samples")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        record_index, end_index = self.index_map[index]
        record = self.records[record_index]
        start_index = end_index - self.sequence_length + 1
        window = record["windows"][start_index]
        target = float(record["y"][end_index])
        return (
            torch.from_numpy(window),
            torch.tensor(target, dtype=torch.float32),
            int(index),
        )

    def metadata(self, index: int) -> Dict[str, Any]:
        record_index, end_index = self.index_map[index]
        record = self.records[record_index]
        start_index = end_index - self.sequence_length + 1
        return {
            "dataset": record["dataset"],
            "battery_id": record["battery_id"],
            "condition_group": record["condition_group"],
            "history_start_cycle": int(record["cycle"][start_index]),
            "cycle": int(record["cycle"][end_index]),
            "history_length": self.sequence_length,
        }

    def balanced_sampler_weights(self) -> np.ndarray:
        domain_records: Dict[str, List[int]] = defaultdict(list)
        for record_index, record in enumerate(self.records):
            domain_records[str(record["dataset"])].append(record_index)
        domain_count = len(domain_records)
        record_weight: Dict[int, float] = {}
        for dataset, record_indices in domain_records.items():
            battery_count = len(record_indices)
            for record_index in record_indices:
                window_count = int(self.records[record_index]["window_count"])
                record_weight[record_index] = (
                    1.0 / domain_count / battery_count / window_count
                )
        weights = np.asarray(
            [record_weight[record_index] for record_index, _ in self.index_map],
            dtype=np.float64,
        )
        weights /= np.sum(weights)
        return weights


def make_train_loader(dataset: BatteryWindowDataset, seed: int) -> DataLoader:
    weights = torch.as_tensor(dataset.balanced_sampler_weights(), dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )


def make_eval_loader(dataset: BatteryWindowDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )


class TemporalAttention(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.Tanh(),
            nn.Linear(dimension // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class TemporalCNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.attention = TemporalAttention(128)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.regressor(x).squeeze(-1)


class CurrentCycleFeatureAttentionMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        summary_dim = input_dim * 3
        self.feature_gate = nn.Sequential(
            nn.Linear(summary_dim, summary_dim),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(summary_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_cycle = x[:, -1, :]
        last_five_mean = x[:, -5:, :].mean(dim=1)
        history_change = current_cycle - x[:, 0, :]
        summary = torch.cat((current_cycle, last_five_mean, history_change), dim=-1)
        summary = summary * self.feature_gate(summary)
        return self.regressor(summary).squeeze(-1)


def build_model(name: str) -> nn.Module:
    if name == "CNN_BiLSTM_Attention":
        return TemporalCNNBiLSTMAttention(FEATURE_COUNT)
    if name == "MLP_Feature_Attention":
        return CurrentCycleFeatureAttentionMLP(FEATURE_COUNT)
    raise KeyError("unknown model: " + name)


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def predict_model(
    model: nn.Module,
    dataset: BatteryWindowDataset,
) -> pd.DataFrame:
    loader = make_eval_loader(dataset)
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for X, y, indices in loader:
            X = X.to(DEVICE, non_blocking=True)
            prediction = model(X).detach().cpu().numpy()
            y_array = y.numpy()
            for local_index, dataset_index in enumerate(indices.numpy().tolist()):
                metadata = dataset.metadata(int(dataset_index))
                metadata.update({
                    "window_index": int(dataset_index),
                    "y_true": float(y_array[local_index]),
                    "y_pred": float(prediction[local_index]),
                })
                rows.append(metadata)
    frame = pd.DataFrame(rows)
    if len(frame) != len(dataset):
        raise AssertionError("prediction count differs from dataset length")
    if not np.all(np.isfinite(frame[["y_true", "y_pred"]].to_numpy())):
        raise ValueError("prediction output contains non-finite values")
    return frame


def balanced_row_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)
    domains = sorted(frame["dataset"].astype(str).unique())
    for dataset in domains:
        domain_mask = frame["dataset"].astype(str).to_numpy() == dataset
        batteries = sorted(frame.loc[domain_mask, "battery_id"].astype(str).unique())
        for battery_id in batteries:
            mask = domain_mask & (
                frame["battery_id"].astype(str).to_numpy() == battery_id
            )
            count = int(mask.sum())
            weights[mask] = 1.0 / len(domains) / len(batteries) / count
    weights /= np.sum(weights)
    return weights


def balanced_mse(frame: pd.DataFrame) -> float:
    error_squared = np.square(frame["y_pred"].to_numpy() - frame["y_true"].to_numpy())
    return float(np.sum(balanced_row_weights(frame) * error_squared))


def train_model(
    model: nn.Module,
    train_dataset: BatteryWindowDataset,
    validation_dataset: BatteryWindowDataset,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    checkpoint_path: Path,
    phase: str,
    model_name: str,
    seed: int,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(seed)
    model = model.to(DEVICE)
    train_loader = make_train_loader(train_dataset, seed)
    validation_loader = make_eval_loader(validation_dataset)
    validation_weights = validation_dataset.balanced_sampler_weights()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
        min_lr=1e-6,
    )
    amp_enabled = bool(USE_MIXED_PRECISION and DEVICE.type == "cuda")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_index, (X, y, _) in enumerate(train_loader, start=1):
            if epoch == 1 and batch_index == 1:
                print(phase, model_name, "first training batch started", flush=True)
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(X)
                loss = criterion(prediction, y)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if epoch == 1 and batch_index == 1:
                print(phase, model_name, "first training batch completed", flush=True)
            loss_sum += float(loss.item()) * len(y)
            sample_count += len(y)

        train_loss = loss_sum / max(sample_count, 1)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for X, y, indices in validation_loader:
                prediction = model(X.to(DEVICE)).detach().cpu().numpy()
                errors = prediction.astype(np.float64) - y.numpy().astype(np.float64)
                validation_loss += float(np.sum(
                    validation_weights[indices.numpy()] * np.square(errors)
                ))
        scheduler.step(validation_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "phase": phase,
            "model": model_name,
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_battery_domain_balanced_mse": validation_loss,
            "learning_rate": current_lr,
        })

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = cpu_state_dict(model)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epochs_without_improvement >= patience:
            print(
                phase,
                model_name,
                "epoch", epoch,
                "train", train_loss,
                "validation", validation_loss,
                "patience", str(epochs_without_improvement) + "/" + str(patience),
                flush=True,
            )
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("no best model state was recorded")
    model.load_state_dict(best_state)
    torch.save({
        "schema_version": STAGE4_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "phase": phase,
        "model_name": model_name,
        "epoch": best_epoch,
        "validation_loss": best_loss,
        "state_dict": best_state,
    }, checkpoint_path)
    model.eval()
    information = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "epochs_completed": int(len(history)),
        "checkpoint": str(checkpoint_path),
    }
    return model, history, information


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def window_feature_frame(dataset: BatteryWindowDataset) -> pd.DataFrame:
    blocks: List[pd.DataFrame] = []
    next_window_index = 0
    names = [
        "f_%s_%02d" % (kind, index)
        for kind in ("current", "mean5", "change20")
        for index in range(FEATURE_COUNT)
    ]
    for record in dataset.records:
        windows = np.asarray(record["windows"], dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
            raise ValueError("invalid cached window shape")
        current = windows[:, -1, :]
        average = windows[:, -5:, :].mean(axis=1)
        change = current - windows[:, 0, :]
        block = pd.DataFrame(
            np.concatenate((current, average, change), axis=1), columns=names
        )
        block.insert(
            0, "window_index",
            np.arange(next_window_index, next_window_index + len(block), dtype=int),
        )
        next_window_index += len(block)
        block.insert(0, "y_true", record["y"][SEQUENCE_LENGTH - 1:].astype(float))
        block.insert(0, "cycle", record["cycle"][SEQUENCE_LENGTH - 1:].astype(int))
        block.insert(0, "battery_id", str(record["battery_id"]))
        block.insert(0, "dataset", str(record["dataset"]))
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    if len(frame) != len(dataset) or frame["window_index"].duplicated().any():
        raise ValueError("tabular features do not match window keys")
    return frame


def fit_independent_hgb(
    dataset: BatteryWindowDataset, seed: int
) -> HistGradientBoostingRegressor:
    frame = window_feature_frame(dataset)
    columns = [column for column in frame if column.startswith("f_")]
    counts = frame.groupby(["dataset", "battery_id"])["battery_id"].transform("size")
    weights = len(frame) / frame["battery_id"].nunique() / counts.to_numpy(float)
    print("Fitting independent HGB from", len(frame), "target adaptation windows",
          flush=True)
    model = HistGradientBoostingRegressor(**HGB_SETTINGS, random_state=seed)
    model.fit(
        frame[columns].to_numpy(np.float32),
        frame["y_true"].to_numpy(float),
        sample_weight=weights,
    )
    return model


def fit_independent_ridge(
    dataset: BatteryWindowDataset, seed: int
) -> Ridge:
    frame = window_feature_frame(dataset)
    columns = [column for column in frame if column.startswith("f_")]
    counts = frame.groupby(["dataset", "battery_id"])["battery_id"].transform("size")
    weights = len(frame) / frame["battery_id"].nunique() / counts.to_numpy(float)
    print("Fitting independent ridge from", len(frame),
          "target adaptation windows", flush=True)
    model = Ridge(alpha=20.0, solver="lsqr", random_state=seed)
    model.fit(frame[columns].to_numpy(np.float32),
              frame["y_true"].to_numpy(float), sample_weight=weights)
    return model


def tabular_prediction_frame(
    template: pd.DataFrame,
    dataset: BatteryWindowDataset,
    model: Any,
) -> pd.DataFrame:
    feature_frame = window_feature_frame(dataset)
    aligned = align_window_features(template, feature_frame)
    columns = [column for column in aligned if column.startswith("f_")]
    result = template.copy()
    result["y_pred"] = model.predict(
        aligned[columns].to_numpy(np.float32)
    )
    if not np.isfinite(result["y_pred"].to_numpy(float)).all():
        raise ValueError("tabular predictions contain non-finite values")
    return result


def align_window_features(
    template: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    if "window_index" not in template or "window_index" not in features:
        raise ValueError("missing window_index; use this build's predict_model")
    if template["window_index"].duplicated().any():
        raise ValueError("neural predictions contain duplicate window indices")
    if features["window_index"].duplicated().any():
        raise ValueError("tabular features contain duplicate window indices")
    ordered = template[[
        "window_index", "dataset", "battery_id", "cycle", "y_true"
    ]].copy()
    ordered.insert(0, "_template_row", np.arange(len(ordered), dtype=int))
    aligned = ordered.merge(
        features, on="window_index", how="left", sort=False,
        validate="one_to_one", suffixes=("_prediction", "_feature"),
        indicator=True,
    ).sort_values("_template_row", kind="stable").reset_index(drop=True)
    if len(aligned) != len(template) or not aligned["_merge"].eq("both").all():
        raise ValueError("HGB feature join lost prediction windows")
    for key in ("dataset", "battery_id", "cycle"):
        left = aligned[key + "_prediction"].astype(str).to_numpy()
        right = aligned[key + "_feature"].astype(str).to_numpy()
        if not np.array_equal(left, right):
            bad = int(np.flatnonzero(left != right)[0])
            raise ValueError(
                "HGB window key mismatch at row %d: %s %r versus %r"
                % (bad, key, left[bad], right[bad])
            )
    labels = aligned["y_true_prediction"].to_numpy(float)
    feature_labels = aligned["y_true_feature"].to_numpy(float)
    if not np.allclose(labels, feature_labels, atol=1e-6, rtol=0.0):
        bad = int(np.flatnonzero(np.abs(labels - feature_labels) > 1e-6)[0])
        raise ValueError(
            "HGB SOH label mismatch at row %d: prediction %.9g versus feature %.9g"
            % (bad, labels[bad], feature_labels[bad])
        )
    return aligned


def merge_model_predictions(
    predictions: Mapping[str, pd.DataFrame],
    names: Sequence[str] = MODEL_NAMES,
) -> pd.DataFrame:
    key_columns = [
        "dataset",
        "battery_id",
        "window_index",
        "condition_group",
        "history_start_cycle",
        "cycle",
        "history_length",
        "y_true",
    ]
    merged: Optional[pd.DataFrame] = None
    for model_name in names:
        frame = predictions[model_name].copy()
        frame = frame.rename(columns={"y_pred": model_name})
        if merged is None:
            merged = frame[key_columns + [model_name]]
        else:
            merged = merged.merge(
                frame[key_columns + [model_name]],
                on=key_columns,
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise ValueError("no model predictions to merge")
    if len(merged) != len(next(iter(predictions.values()))):
        raise ValueError("model prediction keys are inconsistent")
    return merged.sort_values(["dataset", "battery_id", "cycle"], kind="stable").reset_index(drop=True)


def fusion_rmse_scores(frame: pd.DataFrame, prediction: np.ndarray) -> Dict[str, float]:
    error = np.asarray(prediction, dtype=float) - frame["y_true"].to_numpy(float)
    groups = list(frame.groupby(["dataset", "battery_id"], sort=True).indices.values())
    return {
        "pooled_cycles": float(np.sqrt(np.mean(error * error))),
        "battery_macro": float(np.mean([
            np.sqrt(np.mean(error[indices] ** 2)) for indices in groups
        ])),
    }


def fusion_score(scores: Mapping[str, float], scales: Mapping[str, float]) -> float:
    return 0.5 * (
        scores["pooled_cycles"] / scales["pooled_cycles"]
        + scores["battery_macro"] / scales["battery_macro"]
    )


def fit_validation_weights(
    frame: pd.DataFrame,
    names: Sequence[str],
    scales: Mapping[str, float],
) -> np.ndarray:
    if len(names) == 1:
        return np.ones(1, dtype=float)
    matrix = frame[list(names)].to_numpy(float)
    label = frame["y_true"].to_numpy(float)
    groups = list(frame.groupby(["dataset", "battery_id"], sort=True).indices.values())

    def value_and_gradient(weights: np.ndarray) -> Tuple[float, np.ndarray]:
        errors = matrix @ weights - label
        pooled = max(float(np.sqrt(np.mean(errors ** 2))), 1e-12)
        pooled_grad = (matrix.T @ errors) / (len(errors) * pooled)
        macro = 0.0
        macro_grad = np.zeros(len(names), dtype=float)
        for indices in groups:
            local = errors[indices]
            local_rmse = max(float(np.sqrt(np.mean(local ** 2))), 1e-12)
            macro += local_rmse / len(groups)
            macro_grad += (matrix[indices].T @ local) / (
                len(groups) * len(indices) * local_rmse
            )
        score = 0.5 * (
            pooled / scales["pooled_cycles"] + macro / scales["battery_macro"]
        )
        gradient = 0.5 * (
            pooled_grad / scales["pooled_cycles"]
            + macro_grad / scales["battery_macro"]
        )
        return score, gradient

    solution = minimize(
        value_and_gradient,
        np.full(len(names), 1.0 / len(names)),
        method="SLSQP", jac=True,
        bounds=[(0.0, 1.0)] * len(names),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 250, "ftol": 1e-11},
    )
    if (not np.isfinite(solution.x).all()
            or np.min(solution.x) < -1e-7
            or abs(float(np.sum(solution.x)) - 1.0) > 1e-6):
        raise RuntimeError("validation fusion optimization produced invalid weights")
    weights = np.maximum(solution.x, 0.0)
    weights /= weights.sum()
    candidates = [weights] + [np.eye(len(names))[i] for i in range(len(names))]
    return min(candidates, key=lambda w: value_and_gradient(w)[0]).copy()


def fit_fusion_policy(
    frame: pd.DataFrame,
    names: Sequence[str],
    mode: str,
    scales: Mapping[str, float],
) -> Dict[str, Any]:
    if mode not in ("global", "condition"):
        raise ValueError("unknown fusion mode: " + mode)
    local_global = fit_validation_weights(frame, names, scales)
    global_weights = np.zeros(len(MODEL_NAMES), dtype=float)
    for name, weight in zip(names, local_global):
        global_weights[MODEL_NAMES.index(name)] = weight
    conditional: Dict[str, np.ndarray] = {}
    if mode == "condition" and len(names) > 1:
        for group_name, group in frame.groupby("condition_group", sort=True):
            battery_count = group["battery_id"].nunique()
            if battery_count < MIN_GROUP_TRAIN_BATTERIES:
                continue
            local = fit_validation_weights(group, names, scales)
            learned = np.zeros(len(MODEL_NAMES), dtype=float)
            for name, weight in zip(names, local):
                learned[MODEL_NAMES.index(name)] = weight


            strength = battery_count / (
                battery_count + GROUP_PRIOR_BATTERY_EQUIVALENTS
            )
            conditional[str(group_name)] = (
                (1.0 - strength) * global_weights + strength * learned
            )
    return {
        "mode": mode,
        "selected_models": list(names),
        "global_weights": global_weights,
        "condition_weights": conditional,
    }


def fusion_row_weights(
    frame: pd.DataFrame, policy: Mapping[str, Any],
) -> np.ndarray:
    base = np.asarray(policy["global_weights"], dtype=float)
    weights = np.tile(base, (len(frame), 1))
    conditions = frame["condition_group"].astype(str).to_numpy()
    for group_name, value in policy["condition_weights"].items():
        weights[conditions == group_name] = np.asarray(value, dtype=float)
    if (
        not np.isfinite(weights).all()
        or (weights < -1e-8).any()
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7)
    ):
        raise ValueError("fusion produced invalid per-window weights")
    return weights


def policy_predict(
    frame: pd.DataFrame, policy: Mapping[str, Any],
) -> np.ndarray:
    weights = fusion_row_weights(frame, policy)
    predictors = frame[list(MODEL_NAMES)].to_numpy(dtype=float)
    return np.sum(predictors * weights, axis=1)


def battery_average_weights(
    frame: pd.DataFrame, policy: Mapping[str, Any],
) -> np.ndarray:
    row_weights = fusion_row_weights(frame, policy)
    battery_means = [
        row_weights[positions].mean(axis=0)
        for positions in frame.groupby(
            ["dataset", "battery_id"], sort=True
        ).indices.values()
    ]
    return np.mean(battery_means, axis=0)


def battery_cv_fusion(
    frame: pd.DataFrame,
    names: Sequence[str],
    mode: str,
    scales: Mapping[str, float],
) -> Dict[str, float]:
    groups = list(
        frame.groupby(["dataset", "battery_id"], sort=True).indices.values()
    )
    if len(groups) < 4:
        raise ValueError("need at least four validation batteries for fusion CV")
    predictions = np.empty(len(frame), dtype=float)
    for held_indices in groups:
        keep = np.ones(len(frame), dtype=bool)
        keep[held_indices] = False
        policy = fit_fusion_policy(frame.iloc[keep], names, mode, scales)
        predictions[held_indices] = policy_predict(
            frame.iloc[held_indices], policy
        )
    return fusion_rmse_scores(frame, predictions)


def select_validation_ensemble(
    frame: pd.DataFrame, target: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    diagnostics: List[Dict[str, Any]] = []
    by_candidate: Dict[Tuple[Tuple[str, ...], str, float], float] = {}
    scope_scores: Dict[Tuple[Tuple[str, ...], str, float], Dict[str, float]] = {}
    candidate_scales: Dict[float, Dict[str, float]] = {
        share: {
            "pooled_cycles": 0.5 / (1.0 - share),
            "battery_macro": 0.5 / share,
        }
        for share in FUSION_MACRO_SHARE_OPTIONS
    }
    enough_groups = (
        frame.groupby("condition_group")["battery_id"].nunique()
        .ge(MIN_GROUP_TRAIN_BATTERIES + 1).sum() >= 2
    )
    for size in range(1, min(MAX_FUSION_BRANCHES, len(MODEL_NAMES)) + 1):
        for names in combinations(MODEL_NAMES, size):
            modes = ("global", "condition") if size > 1 and enough_groups else ("global",)
            for mode in modes:
                for share in (FUSION_MACRO_SHARE_OPTIONS if size > 1 else (0.5,)):
                    scales = candidate_scales[share]
                    scores = battery_cv_fusion(frame, names, mode, scales)
                    candidate = (names, mode, share)


                    joint = 0.5 * (
                        scores["pooled_cycles"] + scores["battery_macro"]
                    )
                    by_candidate[candidate] = joint
                    scope_scores[candidate] = scores
                    diagnostics.append({
                        "target": target,
                        "candidate_models": "|".join(names),
                        "model_count": size,
                        "fusion_mode": mode,
                        "fit_macro_share": share,
                        "battery_cv_pooled_RMSE": scores["pooled_cycles"],
                        "battery_cv_macro_RMSE": scores["battery_macro"],
                        "joint_cv_score": joint,
                        "test_labels_used": False,
                    })
            print(target, "fusion candidates", size, "branch(es) evaluated", flush=True)

    def best_reduced(names: Tuple[str, ...], modes: Sequence[str]) -> float:
        return min(
            score for (subset, candidate_mode, _), score in by_candidate.items()
            if subset == names and candidate_mode in modes
        )

    best_single_scopes = {
        scope: min(
            scope_scores[((name,), "global", 0.5)][scope]
            for name in MODEL_NAMES
        )
        for scope in ("pooled_cycles", "battery_macro")
    }
    eligible: List[Tuple[Tuple[str, ...], str, float]] = []
    for candidate, scores in scope_scores.items():
        names, mode, share = candidate
        if len(names) < 2:
            continue
        beats_singles = all(
            scores[scope] <= best_single_scopes[scope] * (
                1.0 - MIN_VALIDATION_GAIN_FRACTION
            )
            for scope in ("pooled_cycles", "battery_macro")
        )
        each_branch_helps = all(
            best_reduced(without, ("global", "condition") if len(without) > 1 else ("global",))
            - by_candidate[candidate] >= (
                MIN_VALIDATION_GAIN_FRACTION
                * best_reduced(without, ("global", "condition") if len(without) > 1 else ("global",))
            )
            for without in combinations(names, len(names) - 1)
        )
        mode_adds_value = (
            mode != "condition"
            or best_reduced(names, ("global",)) - by_candidate[candidate]
            >= MIN_CONDITION_GAIN_FRACTION * best_reduced(names, ("global",))
        )
        if beats_singles and each_branch_helps and mode_adds_value:
            eligible.append(candidate)

    selected: Optional[Tuple[Tuple[str, ...], str, float]] = None
    policy: Optional[Dict[str, Any]] = None
    for candidate in sorted(
        eligible, key=lambda c: (by_candidate[c], len(c[0]), c[1], c[2])
    ):
        names, mode, share = candidate
        fitted = fit_fusion_policy(frame, names, mode, candidate_scales[share])
        mean_weights = battery_average_weights(frame, fitted)
        if all(
            mean_weights[MODEL_NAMES.index(name)] >= MIN_MEAN_BRANCH_WEIGHT
            for name in names
        ):
            selected, policy = candidate, fitted
            break
    if selected is None:
        selected = min(
            (((name,), "global", 0.5) for name in MODEL_NAMES),
            key=lambda c: by_candidate[c]
        )
        policy = fit_fusion_policy(frame, selected[0], selected[1], candidate_scales[0.5])
    assert policy is not None
    names, mode, share = selected
    policy["fit_macro_share"] = share
    selected_score = by_candidate[selected]
    best_single_score = min(
        by_candidate[((name,), "global", 0.5)] for name in MODEL_NAMES
    )
    drop_gains = {
        name: (
            best_reduced(
                tuple(other for other in names if other != name),
                ("global", "condition") if len(names) > 2 else ("global",),
            ) - selected_score
            if name in names and len(names) >= 2 else float("nan")
        )
        for name in MODEL_NAMES
    }
    for row in diagnostics:
        candidate = (
            tuple(row["candidate_models"].split("|")),
            row["fusion_mode"],
            row["fit_macro_share"],
        )
        row["eligible_cv_contribution"] = candidate in eligible
        row["selected"] = candidate == selected
        row["selected_cv_score"] = selected_score
        row["best_single_cv_score"] = best_single_score
        row["incremental_cv_gain_if_selected"] = float("nan")
        for name in MODEL_NAMES:
            row["drop_%s_cv_RMSE_gain" % name] = (
                drop_gains[name] if row["selected"] else float("nan")
            )
        if row["selected"] and len(names) >= 2:
            row["incremental_cv_gain_if_selected"] = min(
                drop_gains[name] for name in names
            )
    mean_weights = battery_average_weights(frame, policy)
    print(
        target, "validation-selected", selected,
        "battery-mean weights", dict(zip(MODEL_NAMES, mean_weights)),
        "CV score", selected_score, "best single", best_single_score,
        flush=True,
    )
    return policy, pd.DataFrame(diagnostics)


def add_ensemble_prediction(
    frame: pd.DataFrame, policy: Mapping[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    row_weights = fusion_row_weights(output, policy)
    output["Adaptive_Ensemble"] = np.sum(
        output[list(MODEL_NAMES)].to_numpy(float) * row_weights, axis=1
    )
    for index, name in enumerate(MODEL_NAMES):
        output["fusion_weight_" + name] = row_weights[:, index]
    return output


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_percent = float(
        100.0 * np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))
    )
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape_percent,
        "R2": r2,
    }


def metric_rows(
    frame: pd.DataFrame,
    target: str,
    split: str,
    prediction_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name in prediction_columns:
        pooled = regression_metrics(
            frame["y_true"].to_numpy(),
            frame[model_name].to_numpy(),
        )
        rows.append({
            "target": target,
            "split": split,
            "model": model_name,
            "evaluation_scope": "pooled_cycles",
            "sample_count": len(frame),
            "battery_count": frame["battery_id"].nunique(),
            **pooled,
        })

        battery_metrics: List[Dict[str, float]] = []
        for _, battery_frame in frame.groupby(["dataset", "battery_id"], sort=False):
            battery_metrics.append(
                regression_metrics(
                    battery_frame["y_true"].to_numpy(),
                    battery_frame[model_name].to_numpy(),
                )
            )
        macro = {
            key: float(np.nanmean([item[key] for item in battery_metrics]))
            for key in ("MAE", "RMSE", "MAPE_percent", "R2")
        }
        rows.append({
            "target": target,
            "split": split,
            "model": model_name,
            "evaluation_scope": "battery_macro",
            "sample_count": len(frame),
            "battery_count": frame["battery_id"].nunique(),
            **macro,
        })
    return rows


def report_ensemble_vs_single(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in TARGET_DOMAINS:
        for scope in ("pooled_cycles", "battery_macro"):
            block = metrics.loc[
                metrics["target"].eq(target)
                & metrics["evaluation_scope"].eq(scope)
            ]
            single = block.loc[block["model"].isin(MODEL_NAMES)].sort_values(
                ["RMSE", "model"], kind="stable"
            ).iloc[0]
            ensemble = block.loc[block["model"].eq("Adaptive_Ensemble")].iloc[0]
            difference = float(single["RMSE"] - ensemble["RMSE"])
            rows.append({
                "target": target,
                "evaluation_scope": scope,
                "best_single_model_on_test_for_reporting_only": str(single["model"]),
                "best_single_RMSE": float(single["RMSE"]),
                "adaptive_ensemble_RMSE": float(ensemble["RMSE"]),
                "RMSE_improvement_vs_best_single": difference,
                "ensemble_lower_RMSE_than_every_single": bool(difference > 0.0),
                "best_branch_for_reporting_only": str(single["model"]),
                "best_branch_RMSE": float(single["RMSE"]),
                "ensemble_lower_RMSE_than_every_branch": bool(difference > 0.0),
                "test_labels_used_to_choose_weights_or_model_subset": False,
            })
    return pd.DataFrame(rows)


def split_overlap_checks(
    splits: Mapping[str, Mapping[str, Set[str]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        names = list(splits[dataset])
        for left_index in range(len(names)):
            for right_index in range(left_index + 1, len(names)):
                left = names[left_index]
                right = names[right_index]
                overlap = sorted(splits[dataset][left] & splits[dataset][right])
                rows.append({
                    "dataset": dataset,
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": len(overlap),
                    "overlap_batteries": ";".join(overlap),
                    "check_pass": len(overlap) == 0,
                })
    return rows


def target_condition_coverage_checks(
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    target_splits = (
        "target_adaptation_train",
        "target_validation",
        "target_test",
    )
    for dataset in TARGET_DOMAINS:
        eligible = split_manifest[
            (split_manifest["dataset"] == dataset)
            & (split_manifest["sequence_eligible"].astype(bool))
        ].copy()
        counts = eligible["condition_group"].astype(str).value_counts()
        for condition, total_count in counts.items():
            coverage_required = int(total_count) >= len(target_splits)
            for split_name in target_splits:
                count = int(
                    (
                        (eligible["condition_group"].astype(str) == str(condition))
                        & (eligible["split"] == split_name)
                    ).sum()
                )
                rows.append({
                    "dataset": dataset,
                    "condition_group": str(condition),
                    "eligible_battery_count": int(total_count),
                    "split": split_name,
                    "split_battery_count": count,
                    "coverage_required": coverage_required,
                    "check_pass": (not coverage_required) or count >= 1,
                })
    return pd.DataFrame(rows)


def training_budget_checks(training_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in training_summary.to_dict(orient="records"):
        if str(row["phase"]) == "source_pretraining":
            configured_max_epochs = MAX_EPOCHS_SOURCE
        else:
            configured_max_epochs = TARGET_MAX_EPOCHS_BY_MODEL[str(row["model"])]
        best_epoch = int(row["best_epoch"])
        epochs_completed = int(row["epochs_completed"])
        exhausted = (
            epochs_completed >= configured_max_epochs
            and best_epoch >= configured_max_epochs
        )
        rows.append({
            "phase": str(row["phase"]),
            "target": str(row["target"]),
            "model": str(row["model"]),
            "configured_max_epochs": configured_max_epochs,
            "best_epoch": best_epoch,
            "epochs_completed": epochs_completed,
            "training_budget_exhausted_at_best_epoch": exhausted,
            "check_pass": not exhausted,
        })
    return pd.DataFrame(rows)


def validate_outputs(
    splits: Mapping[str, Mapping[str, Set[str]]],
    split_manifest: pd.DataFrame,
    standardizer: FrozenStandardizer,
    target_predictions: Mapping[str, pd.DataFrame],
    target_validation_predictions: Mapping[str, pd.DataFrame],
    ensemble_weights: Mapping[str, np.ndarray],
    ensemble_policies: Mapping[str, Mapping[str, Any]],
    target_validation_metrics: pd.DataFrame,
    metrics: pd.DataFrame,
    training_history: pd.DataFrame,
    training_summary: pd.DataFrame,
) -> Tuple[
    List[str],
    List[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    fatal_checks: List[str] = []
    warnings: List[str] = []
    leakage_frame = pd.DataFrame(split_overlap_checks(splits))
    condition_coverage_frame = target_condition_coverage_checks(split_manifest)
    training_budget_frame = training_budget_checks(training_summary)
    if not leakage_frame["check_pass"].all():
        fatal_checks.append("one or more battery splits overlap")
    if not condition_coverage_frame["check_pass"].all():
        fatal_checks.append(
            "a target condition with at least three eligible batteries is "
            "missing from target train, validation or final test"
        )
    if not training_budget_frame["check_pass"].all():
        fatal_checks.append(
            "one or more models reached their epoch ceiling at the best "
            "validation checkpoint; convergence is not established"
        )

    if set(standardizer.fitted_datasets) != set(SOURCE_DOMAINS):
        fatal_checks.append("standardizer was not fitted only on source domains")
    for dataset in SOURCE_DOMAINS:
        if set(standardizer.fitted_batteries[dataset]) != splits[dataset]["source_train"]:
            fatal_checks.append(dataset + " scaler battery set differs from source_train")
    if not np.all(np.isfinite(standardizer.mean_)) or not np.all(np.isfinite(standardizer.scale_)):
        fatal_checks.append("standardizer parameters are non-finite")
    if np.any(standardizer.scale_ <= 0.0):
        fatal_checks.append("standardizer contains non-positive scales")

    for target in TARGET_DOMAINS:
        test_frame = target_predictions[target]
        validation_frame = target_validation_predictions[target]
        expected_test = splits[target]["target_test"]
        expected_validation = splits[target]["target_validation"]
        if set(test_frame["battery_id"].astype(str).unique()) != expected_test:
            fatal_checks.append(target + " test predictions use unexpected batteries")
        if set(validation_frame["battery_id"].astype(str).unique()) != expected_validation:
            fatal_checks.append(target + " validation predictions use unexpected batteries")
        if expected_test & splits[target]["target_adaptation_train"]:
            fatal_checks.append(target + " test/adaptation battery overlap")
        if expected_test & expected_validation:
            fatal_checks.append(target + " test/validation battery overlap")

        prediction_columns = ["y_true"] + list(MODEL_NAMES) + ["Adaptive_Ensemble"]
        if not np.all(np.isfinite(test_frame[prediction_columns].to_numpy())):
            fatal_checks.append(target + " test predictions contain non-finite values")
        weights = np.asarray(ensemble_weights[target], dtype=np.float64)
        if (len(weights) != len(MODEL_NAMES)
                or not np.isfinite(weights).all()
                or np.any(weights < -1e-9)
                or np.any(weights > 1.0 + 1e-9)):
            fatal_checks.append(target + " ensemble weights are invalid")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            fatal_checks.append(target + " ensemble weights do not sum to one")
        for split_name, predicted in (("validation", validation_frame),
                                      ("final_test", test_frame)):
            policy_weights = fusion_row_weights(
                predicted, ensemble_policies[target]
            )
            stored_weights = predicted[
                ["fusion_weight_" + name for name in MODEL_NAMES]
            ].to_numpy(float)
            if not np.allclose(
                policy_weights, stored_weights, atol=1e-10, rtol=0.0
            ):
                fatal_checks.append(target + " " + split_name
                                    + " per-row fusion weights differ")
            replay = np.sum(
                predicted[list(MODEL_NAMES)].to_numpy(float) * policy_weights,
                axis=1,
            )
            if not np.allclose(replay,
                               predicted["Adaptive_Ensemble"].to_numpy(np.float64),
                               atol=1e-10, rtol=0.0):
                fatal_checks.append(target + " " + split_name + " fusion replay mismatch")

    all_metrics = pd.concat(
        [target_validation_metrics, metrics],
        axis=0,
        ignore_index=True,
    )
    numeric_metrics = all_metrics[
        ["MAE", "RMSE", "MAPE_percent", "R2"]
    ].to_numpy()
    if not np.all(np.isfinite(numeric_metrics)):
        fatal_checks.append("metrics contain non-finite values")
    numeric_history = training_history[
        ["train_mse", "validation_battery_domain_balanced_mse", "learning_rate"]
    ].to_numpy()
    if not np.all(np.isfinite(numeric_history)):
        fatal_checks.append("training history contains non-finite values")

    if split_manifest["used_for_final_test"].sum() == 0:
        fatal_checks.append("no final-test batteries were assigned")
    if (
        split_manifest["used_for_final_test"]
        & (
            split_manifest["used_for_scaler"]
            | split_manifest["used_for_source_pretraining"]
            | split_manifest["used_for_target_finetuning"]
            | split_manifest["used_for_early_stopping"]
            | split_manifest["used_for_ensemble_weight_learning"]
        )
    ).any():
        fatal_checks.append("a final-test battery was used before final evaluation")

    mixed_condition_count = int(
        (split_manifest["condition_group_count"].astype(int) > 1).sum()
    )
    if mixed_condition_count:
        warnings.append(
            str(mixed_condition_count)
            + " batteries contain more than one condition_group label"
        )

    return (
        fatal_checks,
        warnings,
        leakage_frame,
        condition_coverage_frame,
        training_budget_frame,
    )


def main() -> None:
    print("============================================================")
    print(" Stage 4 V4.0.2 validation-learned selective ensemble started")
    print(" Script build:", SCRIPT_BUILD)
    print(" Device:", DEVICE)
    print("============================================================")

    stage3_run = find_latest_stage3_run()
    print("Stage 3 input:", stage3_run)
    feature_names = read_selected_feature_names(stage3_run)
    datasets = load_all_stage3_datasets(stage3_run, feature_names)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_path = OUTPUT_ROOT / run_name
    checkpoint_root = run_path / "checkpoints"
    prediction_root = run_path / "predictions"
    metric_root = run_path / "metrics"
    report_path = run_path / "validation_report"
    for path in (checkpoint_root, prediction_root, metric_root, report_path):
        path.mkdir(parents=True, exist_ok=False)
    (run_path / "selected_features.txt").write_text(
        "\n".join(feature_names) + "\n",
        encoding="utf-8",
    )

    battery_table = build_battery_table(datasets)
    splits, split_policies = create_all_splits(battery_table)
    split_manifest = build_split_manifest(battery_table, splits)
    write_csv(split_manifest, run_path / "battery_split_manifest.csv")

    standardizer = fit_source_standardizer(
        datasets,
        splits,
        feature_names,
    )


    write_pickle(
        standardizer.to_serializable_dict(),
        run_path / "source_standardizer.pkl",
    )

    source_train_selection = {
        dataset: splits[dataset]["source_train"] for dataset in SOURCE_DOMAINS
    }
    source_validation_selection = {
        dataset: splits[dataset]["source_validation"] for dataset in SOURCE_DOMAINS
    }
    source_train_dataset = BatteryWindowDataset(
        datasets,
        source_train_selection,
        standardizer,
        SEQUENCE_LENGTH,
    )
    source_validation_dataset = BatteryWindowDataset(
        datasets,
        source_validation_selection,
        standardizer,
        SEQUENCE_LENGTH,
    )

    print(
        "Source windows:",
        "train=", len(source_train_dataset),
        "validation=", len(source_validation_dataset),
    )

    all_history_rows: List[Dict[str, Any]] = []
    training_summary_rows: List[Dict[str, Any]] = []
    source_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
    source_validation_frames: Dict[str, pd.DataFrame] = {}
    model_parameters: Dict[str, int] = {}

    print("\n================ Source pre-training ================")
    for model_index, model_name in enumerate(NEURAL_MODEL_NAMES):
        set_seed(RANDOM_SEED + model_index)
        model = build_model(model_name)
        model_parameters[model_name] = model_parameter_count(model)
        checkpoint = checkpoint_root / "source" / (model_name + "_best.pth")
        model, history, information = train_model(
            model,
            source_train_dataset,
            source_validation_dataset,
            MAX_EPOCHS_SOURCE,
            PATIENCE_SOURCE,
            SOURCE_LEARNING_RATE,
            checkpoint,
            "source_pretraining",
            model_name,
            RANDOM_SEED + model_index,
        )
        all_history_rows.extend(history)
        training_summary_rows.append({
            "phase": "source_pretraining",
            "target": "source",
            "model": model_name,
            **information,
        })
        source_state_dicts[model_name] = cpu_state_dict(model)
        source_validation_frames[model_name] = predict_model(
            model,
            source_validation_dataset,
        )

    source_merged = merge_model_predictions(source_validation_frames, NEURAL_MODEL_NAMES)
    source_metric_rows = metric_rows(
        source_merged,
        "XJTU+HUST",
        "source_validation",
        NEURAL_MODEL_NAMES,
    )

    target_test_predictions: Dict[str, pd.DataFrame] = {}
    target_validation_predictions: Dict[str, pd.DataFrame] = {}
    ensemble_weights: Dict[str, np.ndarray] = {}
    ensemble_policies: Dict[str, Dict[str, Any]] = {}
    target_validation_metric_rows: List[Dict[str, Any]] = []
    target_metric_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    ensemble_diagnostics: List[pd.DataFrame] = []

    for target_index, target in enumerate(TARGET_DOMAINS):
        print("\n================ Target:", target, "================")
        train_dataset = BatteryWindowDataset(
            datasets,
            {target: splits[target]["target_adaptation_train"]},
            standardizer,
            SEQUENCE_LENGTH,
        )
        validation_dataset = BatteryWindowDataset(
            datasets,
            {target: splits[target]["target_validation"]},
            standardizer,
            SEQUENCE_LENGTH,
        )
        test_dataset = BatteryWindowDataset(
            datasets,
            {target: splits[target]["target_test"]},
            standardizer,
            SEQUENCE_LENGTH,
        )
        print(
            target,
            "windows train/validation/test =",
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset),
        )

        validation_frames: Dict[str, pd.DataFrame] = {}
        test_frames: Dict[str, pd.DataFrame] = {}
        for model_index, model_name in enumerate(NEURAL_MODEL_NAMES):


            set_seed(RANDOM_SEED + 1000 * (target_index + 1) + model_index)
            model = build_model(model_name)
            model.load_state_dict(source_state_dicts[model_name])
            checkpoint = (
                checkpoint_root / "targets" / target / (model_name + "_best.pth")
            )
            model, history, information = train_model(
                model,
                train_dataset,
                validation_dataset,
                TARGET_MAX_EPOCHS_BY_MODEL[model_name],
                PATIENCE_TARGET,
                TARGET_LEARNING_RATE,
                checkpoint,
                "target_finetuning_" + target,
                model_name,
                RANDOM_SEED + 1000 * (target_index + 1) + model_index,
            )
            all_history_rows.extend(history)
            training_summary_rows.append({
                "phase": "target_finetuning",
                "target": target,
                "model": model_name,
                **information,
            })
            validation_frames[model_name] = predict_model(model, validation_dataset)
            test_frames[model_name] = predict_model(model, test_dataset)

        print(target, "fitting independent HGB branch", flush=True)
        hgb = fit_independent_hgb(
            train_dataset, RANDOM_SEED + 12000 + target_index,
        )
        template_validation = validation_frames[NEURAL_MODEL_NAMES[0]]
        template_test = test_frames[NEURAL_MODEL_NAMES[0]]
        validation_frames[HGB_MODEL_NAME] = tabular_prediction_frame(
            template_validation, validation_dataset, hgb,
        )
        test_frames[HGB_MODEL_NAME] = tabular_prediction_frame(
            template_test, test_dataset, hgb,
        )
        write_pickle({
            "script_build": SCRIPT_BUILD,
            "target": target,
            "model_name": HGB_MODEL_NAME,
            "regressor": hgb,
            "feature_columns": [
                "f_%s_%02d" % (kind, index)
                for kind in ("current", "mean5", "change20")
                for index in range(FEATURE_COUNT)
            ],
            "trained_on": "target adaptation-training batteries, independent SOH labels",
            "test_labels_used": False,
        }, checkpoint_root / "targets" / target / (HGB_MODEL_NAME + "_best.pkl"))

        ridge = fit_independent_ridge(
            train_dataset, RANDOM_SEED + 13000 + target_index,
        )
        validation_frames[RIDGE_MODEL_NAME] = tabular_prediction_frame(
            template_validation, validation_dataset, ridge,
        )
        test_frames[RIDGE_MODEL_NAME] = tabular_prediction_frame(
            template_test, test_dataset, ridge,
        )
        write_pickle({
            "script_build": SCRIPT_BUILD,
            "target": target,
            "model_name": RIDGE_MODEL_NAME,
            "regressor": ridge,
            "feature_columns": [
                "f_%s_%02d" % (kind, index)
                for kind in ("current", "mean5", "change20")
                for index in range(FEATURE_COUNT)
            ],
            "trained_on": "target adaptation-training batteries, independent SOH labels",
            "test_labels_used": False,
        }, checkpoint_root / "targets" / target / (RIDGE_MODEL_NAME + "_best.pkl"))

        validation_merged = merge_model_predictions(validation_frames)
        policy, diagnostics = select_validation_ensemble(validation_merged, target)
        ensemble_diagnostics.append(diagnostics)
        chosen_diagnostics = diagnostics.loc[diagnostics["selected"]].iloc[0]
        validation_merged = add_ensemble_prediction(validation_merged, policy)
        test_merged = add_ensemble_prediction(
            merge_model_predictions(test_frames), policy,
        )
        target_validation_predictions[target] = validation_merged
        target_test_predictions[target] = test_merged
        ensemble_policies[target] = policy
        ensemble_weights[target] = battery_average_weights(
            validation_merged, policy
        )
        weight_views: List[Tuple[str, str, np.ndarray]] = [
            ("global", "ALL", np.asarray(policy["global_weights"])),
            ("validation_battery_mean", "ALL", ensemble_weights[target]),
            ("test_battery_mean", "ALL",
             battery_average_weights(test_merged, policy)),
        ]
        weight_views.extend(
            ("condition", group, np.asarray(group_weights))
            for group, group_weights in policy["condition_weights"].items()
        )
        for scope, condition, vector in weight_views:
            for model_name, weight in zip(MODEL_NAMES, vector):
                weight_rows.append({
                    "target": target,
                    "fusion_mode": policy["mode"],
                    "weight_scope": scope,
                    "condition_group": condition,
                    "model": model_name,
                    "weight": float(weight),
                    "selected": model_name in policy["selected_models"],
                    "learned_on": "target validation batteries, battery-CV subset selection",
                    "battery_cv_RMSE_gain_from_including_model": chosen_diagnostics[
                        "drop_%s_cv_RMSE_gain" % model_name
                    ],
                    "test_labels_used": False,
                })
        write_json({
            "script_build": SCRIPT_BUILD,
            "target": target,
            "mode": policy["mode"],
            "selected_models": policy["selected_models"],
            "fit_macro_share": policy["fit_macro_share"],
            "global_weights": policy["global_weights"],
            "condition_weights": policy["condition_weights"],
            "test_labels_used": False,
        }, checkpoint_root / "targets" / target / "fusion_policy.json")

        prediction_target_path = prediction_root / target
        write_csv(
            validation_merged,
            prediction_target_path / "validation_predictions.csv",
        )
        write_csv(
            test_merged,
            prediction_target_path / "test_predictions.csv",
        )
        target_validation_metric_rows.extend(
            metric_rows(
                validation_merged,
                target,
                "ensemble_calibration_validation",
                list(MODEL_NAMES) + ["Adaptive_Ensemble"],
            )
        )
        target_metric_rows.extend(
            metric_rows(
                test_merged,
                target,
                "final_test",
                list(MODEL_NAMES) + ["Adaptive_Ensemble"],
            )
        )

    training_history = pd.DataFrame(all_history_rows)
    training_summary = pd.DataFrame(training_summary_rows)
    source_metrics = pd.DataFrame(source_metric_rows)
    target_validation_metrics = pd.DataFrame(target_validation_metric_rows)
    target_metrics = pd.DataFrame(target_metric_rows)
    ensemble_comparison = report_ensemble_vs_single(target_metrics)
    desired_ranking_met = bool(
        ensemble_comparison["ensemble_lower_RMSE_than_every_single"].all()
    )
    selected_model_count = {
        target: int(np.count_nonzero(ensemble_weights[target] > 1e-6))
        for target in TARGET_DOMAINS
    }
    three_branch_goal_met = all(
        count == MAX_FUSION_BRANCHES for count in selected_model_count.values()
    )
    contributing_models_across_targets = sorted(set().union(*(
        set(ensemble_policies[target]["selected_models"])
        for target in TARGET_DOMAINS
    )))
    condition_weights_frame = pd.DataFrame(weight_rows)
    weights_frame = condition_weights_frame.loc[
        condition_weights_frame["weight_scope"].eq("validation_battery_mean")
    ].drop(columns=["weight_scope", "condition_group"]).reset_index(drop=True)
    diagnostics_frame = pd.concat(ensemble_diagnostics, ignore_index=True)

    write_csv(training_history, metric_root / "training_history.csv")
    write_csv(training_summary, metric_root / "training_summary.csv")
    write_csv(source_metrics, metric_root / "source_validation_metrics.csv")
    write_csv(
        target_validation_metrics,
        metric_root / "target_validation_metrics.csv",
    )
    write_csv(target_metrics, metric_root / "target_test_metrics.csv")
    write_csv(ensemble_comparison, metric_root / "ensemble_vs_single_test.csv")
    write_csv(weights_frame, metric_root / "ensemble_weights.csv")
    write_csv(condition_weights_frame, metric_root / "condition_ensemble_weights.csv")
    write_csv(diagnostics_frame, metric_root / "ensemble_validation_selection.csv")

    (
        fatal_checks,
        warnings,
        leakage_frame,
        condition_coverage_frame,
        training_budget_frame,
    ) = validate_outputs(
        splits,
        split_manifest,
        standardizer,
        target_test_predictions,
        target_validation_predictions,
        ensemble_weights,
        ensemble_policies,
        target_validation_metrics,
        target_metrics,
        training_history,
        training_summary,
    )
    validation_status = "PASS" if not fatal_checks else "FAIL"


    write_csv(split_manifest, report_path / "battery_split_manifest.csv")
    write_csv(leakage_frame, report_path / "battery_overlap_checks.csv")
    write_csv(
        condition_coverage_frame,
        report_path / "target_condition_coverage_checks.csv",
    )
    write_csv(
        training_budget_frame,
        report_path / "training_budget_checks.csv",
    )
    write_csv(training_summary, report_path / "training_summary.csv")
    write_csv(source_metrics, report_path / "source_validation_metrics.csv")
    write_csv(
        target_validation_metrics,
        report_path / "target_validation_metrics.csv",
    )
    write_csv(target_metrics, report_path / "target_test_metrics.csv")
    write_csv(ensemble_comparison, report_path / "ensemble_vs_single_test.csv")
    write_csv(weights_frame, report_path / "ensemble_weights.csv")
    write_csv(condition_weights_frame, report_path / "condition_ensemble_weights.csv")
    write_csv(diagnostics_frame, report_path / "ensemble_validation_selection.csv")
    write_csv(training_history, report_path / "training_history.csv")
    for target in TARGET_DOMAINS:
        write_csv(
            target_validation_predictions[target],
            report_path / (target + "_validation_predictions.csv"),
        )
        write_csv(
            target_test_predictions[target],
            report_path / (target + "_test_predictions.csv"),
        )

    split_counts = (
        split_manifest.groupby(["dataset", "split"])
        .agg(battery_count=("battery_id", "nunique"), window_count=("window_count", "sum"))
        .reset_index()
    )
    write_csv(split_counts, report_path / "split_counts.csv")

    config = {
        "schema_version": STAGE4_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage3_run": str(stage3_run),
        "output_run": str(run_path),
        "device": str(DEVICE),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_requested": True,
        "source_domains": list(SOURCE_DOMAINS),
        "target_domains": list(TARGET_DOMAINS),
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "source_train_ratio": SOURCE_TRAIN_RATIO,
        "target_train_ratio": TARGET_TRAIN_RATIO,
        "target_validation_ratio": TARGET_VALIDATION_RATIO,
        "target_test_ratio": TARGET_TEST_RATIO,
        "split_unit": "battery",
        "split_policies": split_policies,
        "standardizer_fit": "source-training batteries only",
        "standardizer_weighting": standardizer.weighting,
        "target_standardizer_refit": False,
        "source_max_epochs": MAX_EPOCHS_SOURCE,
        "target_max_epochs": MAX_EPOCHS_TARGET,
        "target_max_epochs_by_model": TARGET_MAX_EPOCHS_BY_MODEL,
        "source_patience": PATIENCE_SOURCE,
        "target_patience": PATIENCE_TARGET,
        "batch_size": BATCH_SIZE,
        "source_learning_rate": SOURCE_LEARNING_RATE,
        "target_learning_rate": TARGET_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "models": list(MODEL_NAMES),
        "independent_comparison_models": list(MODEL_NAMES),
        "mlp_window_summary_inputs": "current_cycle,last_five_mean,current_minus_first",
        "neural_models_with_source_pretraining": list(NEURAL_MODEL_NAMES),
        "model_parameter_counts": model_parameters,
        "hgb_model": HGB_MODEL_NAME,
        "hgb_settings": HGB_SETTINGS,
        "ridge_model": RIDGE_MODEL_NAME,
        "ridge_alpha": 20.0,
        "maximum_fusion_branches": MAX_FUSION_BRANCHES,
        "minimum_validation_gain_fraction": MIN_VALIDATION_GAIN_FRACTION,
        "minimum_selected_battery_mean_weight": MIN_MEAN_BRANCH_WEIGHT,
        "condition_min_training_batteries": MIN_GROUP_TRAIN_BATTERIES,
        "condition_shrinkage_prior_batteries": GROUP_PRIOR_BATTERY_EQUIVALENTS,
        "minimum_condition_mode_gain_fraction": MIN_CONDITION_GAIN_FRACTION,
        "fusion_fit_macro_share_options": FUSION_MACRO_SHARE_OPTIONS,
        "ensemble_weight_fit_split": "target_validation_batteries",
        "ensemble_weight_objective": "validation CV selects between 20 and 50 percent battery-macro shares; candidate ranking uses equal-mean pooled and macro RMSE",
        "ensemble_weight_selection": "leave-one-validation-battery-out subset and fusion-mode selection",
        "ensemble_candidate_subsets": (
            "all subsets of one, two or three of four independent candidates, "
            "global and condition-specific fusion for multi-branch candidates"
        ),
        "ensemble_selection_rule": (
            "choose lowest joint battery-CV score among policies beating every "
            "individual model under both RMSE scopes, beating each same-mode "
            "best one-branch-removed subset across fit objectives by at least 0.5 percent, granting "
            "selected branches battery-mean learned weights of at least 5 percent, "
            "and requiring condition mode to improve its global counterpart "
            "by at least 0.2 percent; "
            "otherwise choose the best singleton"
        ),
        "positive_weights_forced": False,
        "validation_reused_for_neural_early_stopping": True,
        "ensemble_has_additive_nonlinear_corrector": False,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "test_labels_used_for_training": False,
        "test_labels_used_for_early_stopping": False,
        "test_labels_used_for_ensemble_weights": False,
        "target_experiments_initialized_independently_from_same_source_checkpoints": True,
        "target_condition_coverage_rule": (
            "every condition with at least three eligible batteries must appear "
            "in adaptation-train, validation and final-test"
        ),
        "random_seed": RANDOM_SEED,
    }

    summary = {
        "schema_version": STAGE4_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "validation_status": validation_status,
        "fatal_checks": fatal_checks,
        "warnings": warnings,
        "feature_count": FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "source_train_window_count": len(source_train_dataset),
        "source_validation_window_count": len(source_validation_dataset),
        "test_labels_used_before_final_evaluation": False,
        "target_experiments_independent": True,
        "selected_model_count_by_target": selected_model_count,
        "contributing_models_across_targets": contributing_models_across_targets,
        "fusion_fit_macro_share_by_target": {
            target: ensemble_policies[target]["fit_macro_share"]
            for target in TARGET_DOMAINS
        },
        "fusion_mode_by_target": {
            target: ensemble_policies[target]["mode"]
            for target in TARGET_DOMAINS
        },
        "three_selected_branches_on_both_targets": three_branch_goal_met,
        "ensemble_strictly_best_in_both_test_scopes_all_targets": desired_ranking_met,
        "both_requested_outcomes_observed": bool(
            three_branch_goal_met and desired_ranking_met
        ),
        "ensemble_weights_learned_on_validation_only": True,
        "ensemble_base_weights_predeclared_positive": False,
        "ensemble_has_additive_nonlinear_corrector": False,
        "hgb_independently_fitted_on_adaptation_train": True,
        "ensemble_weight_objective": "battery-CV selects pooled/macro fitting share; all candidates ranked by their equal-mean CV RMSE",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "target_condition_coverage_pass": bool(
            condition_coverage_frame["check_pass"].all()
        ),
        "training_budget_check_pass": bool(
            training_budget_frame["check_pass"].all()
        ),
        "datasets": {
            row["dataset"] + ":" + row["split"]: {
                "battery_count": int(row["battery_count"]),
                "window_count": int(row["window_count"]),
            }
            for row in split_counts.to_dict(orient="records")
        },
        "ensemble_weights": {
            target: {
                model: float(weight)
                for model, weight in zip(MODEL_NAMES, ensemble_weights[target])
            }
            for target in TARGET_DOMAINS
        },
        "ensemble_condition_weights": {
            target: {
                group: {
                    model: float(value)
                    for model, value in zip(MODEL_NAMES, vector)
                }
                for group, vector in ensemble_policies[target][
                    "condition_weights"
                ].items()
            }
            for target in TARGET_DOMAINS
        },
        "ensemble_selection_on_validation_batteries": [
            {
                key: (None if isinstance(value, float) and np.isnan(value)
                      else value)
                for key, value in record.items()
            }
            for record in diagnostics_frame.loc[
                diagnostics_frame["selected"]
            ].to_dict(orient="records")
        ],
        "ensemble_vs_single_test_reporting_only": ensemble_comparison.to_dict(
            orient="records"
        ),
        "target_final_test_pooled_metrics": {
            target: {
                row["model"]: {
                    "MAE": float(row["MAE"]),
                    "RMSE": float(row["RMSE"]),
                    "MAPE_percent": float(row["MAPE_percent"]),
                    "R2": float(row["R2"]),
                }
                for row in target_metrics[
                    (target_metrics["target"] == target)
                    & (target_metrics["evaluation_scope"] == "pooled_cycles")
                ].to_dict(orient="records")
            }
            for target in TARGET_DOMAINS
        },
    }
    write_json(config, report_path / "processing_config.json")
    write_json(summary, report_path / "processing_summary.json")

    readme = (
        "Stage 4 V4.0.2 conditional-contribution ensemble report\n"
        "=======================================\n\n"
        "Validation status: " + validation_status + "\n"
        "Script build: " + SCRIPT_BUILD + "\n"
        "Stage 3 input: " + str(stage3_run) + "\n"
        "Stage 4 output: " + str(run_path) + "\n\n"
        "All splits are battery-level.\n"
        "The standardizer was fitted only on source-training batteries.\n"
        "Target validation batteries control neural early stopping.\n"
        "The same validation split calibrates the fusion; the subset selection\n"
        "cross-validates coefficients across batteries, but is not a fully\n"
        "independent second validation set for the neural checkpoints.\n"
        "HGB and ridge are independently fitted on adaptation-training batteries.\n"
        "Global and condition-specific policies are compared using\n"
        "leave-one-validation-battery-out predictions. Every selected branch\n"
        "must add 0.5 percent CV improvement and have at least 5 percent\n"
        "battery-mean learned weight; its coefficient itself is not forced.\n"
        "If no candidate beats every single under both validation-CV scopes,\n"
        "a single predictor is selected; no ensemble advantage is claimed.\n"
        "This build requires downstream Stage 5-7 replay adaptations.\n"
        "Every target condition represented by at least three eligible batteries\n"
        "is covered by adaptation-train, validation and final-test splits.\n"
        "Final-test batteries were not used for training, scaling, early stopping,\n"
        "or ensemble-weight optimization.\n"
        "Return STAGE4_VALIDATION_REPORT.zip for model and integrity analysis.\n"
    )
    (report_path / "README.txt").write_text(readme, encoding="utf-8")

    zip_path = run_path / "STAGE4_VALIDATION_REPORT.zip"
    zip_validation_report(report_path, zip_path)

    print("============================================================")
    print("Validation status:", validation_status)
    print(target_metrics[target_metrics["evaluation_scope"] == "pooled_cycles"])
    print("Battery macro RMSE:")
    print(target_metrics.loc[
        target_metrics["evaluation_scope"] == "battery_macro",
        ["target", "model", "RMSE"],
    ].to_string(index=False))
    print("Adaptive fusion vs best single model (test reporting only):")
    print(ensemble_comparison.to_string(index=False))
    print("Ensemble strictly best in both test scopes and both targets:",
          desired_ranking_met)
    print("Three positively contributing branches on both targets:",
          three_branch_goal_met, selected_model_count)
    print("Distinct contributing models across targets:",
          contributing_models_across_targets)
    print("Both requested outcomes observed:",
          bool(three_branch_goal_met and desired_ranking_met))
    print("Fusion modes:", {
        target: ensemble_policies[target]["mode"]
        for target in TARGET_DOMAINS
    })
    print("Validation battery-mean learned weights:", {
        target: dict(zip(MODEL_NAMES, ensemble_weights[target]))
        for target in TARGET_DOMAINS
    })
    print("Validation report:", zip_path)
    print("============================================================")

    if fatal_checks:
        raise RuntimeError(
            "Stage 4 validation failed:\n- " + "\n- ".join(fatal_checks)
        )


if __name__ == "__main__":
    main()
