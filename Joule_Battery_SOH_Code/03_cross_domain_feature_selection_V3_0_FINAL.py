from __future__ import annotations

import os

import json
import pickle
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


STAGE3_SCHEMA_VERSION = "3.0.1"
SCRIPT_BUILD = "STAGE3_V3_0_1_FINAL_20260901"

EXPECTED_STAGE2_SCHEMA = "2.0.2"
EXPECTED_STAGE2_BUILD = "STAGE2_V2_0_2_FINAL_20260901"

BASE_PATH = Path(os.environ.get("CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"))
STAGE2_ROOT = BASE_PATH / "FEATUREPROCESSING_V2_0"
OUTPUT_ROOT = BASE_PATH / "FEATURESELECTION_V3_0_1"


STAGE2_RUN_PATH: Optional[Path] = None

DATASET_ORDER = ("XJTU", "HUST", "TJU_NCM", "TJU_NCA")
SOURCE_DOMAINS = ("XJTU", "HUST")
TARGET_DOMAINS = ("TJU_NCM", "TJU_NCA")


TOP_K = 20
ALPHA = 0.50

MIN_BATTERY_SAMPLES = 20
MIN_VALID_BATTERIES_PER_DOMAIN = 10
CORRELATION_EPS = 1e-12

BOOTSTRAP_ITERATIONS = 300
RANDOM_SEED = 42


REDUNDANCY_PENALTY = 0.50
REDUNDANCY_MAX_CYCLES_PER_BATTERY = 200


MAX_SELECTED_REDUNDANCY = 0.99


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


def find_latest_stage2_run() -> Path:
    if STAGE2_RUN_PATH is not None:
        candidates = [Path(STAGE2_RUN_PATH)]
    else:
        if not STAGE2_ROOT.exists():
            raise FileNotFoundError("Stage 2 root does not exist: " + str(STAGE2_ROOT))
        candidates = sorted(
            [path for path in STAGE2_ROOT.glob("run_*") if path.is_dir()],
            reverse=True,
        )

    rejection_reasons: List[str] = []
    for candidate in candidates:
        report_path = candidate / "validation_report"
        config_path = report_path / "processing_config.json"
        summary_path = report_path / "processing_summary.json"
        definition_path = report_path / "feature_definition.csv"
        dataset_path = candidate / "datasets"

        required_paths = [
            config_path,
            summary_path,
            definition_path,
            dataset_path,
        ]
        if any(not path.exists() for path in required_paths):
            rejection_reasons.append(
                str(candidate) + ": missing required Stage 2 output"
            )
            continue

        try:
            config = read_json(config_path)
            summary = read_json(summary_path)
        except Exception as exc:
            rejection_reasons.append(
                str(candidate) + ": invalid JSON: " + str(exc)
            )
            continue

        if str(config.get("schema_version")) != EXPECTED_STAGE2_SCHEMA:
            rejection_reasons.append(
                str(candidate) + ": schema=" + str(config.get("schema_version"))
            )
            continue
        if str(config.get("script_build")) != EXPECTED_STAGE2_BUILD:
            rejection_reasons.append(
                str(candidate) + ": build=" + str(config.get("script_build"))
            )
            continue
        if str(summary.get("validation_status")) != "PASS":
            rejection_reasons.append(str(candidate) + ": validation is not PASS")
            continue
        if summary.get("fatal_checks"):
            rejection_reasons.append(str(candidate) + ": fatal checks are present")
            continue
        if int(summary.get("feature_count", -1)) != 39:
            rejection_reasons.append(str(candidate) + ": feature count is not 39")
            continue
        if bool(config.get("standardization_performed")):
            rejection_reasons.append(str(candidate) + ": Stage 2 data are standardized")
            continue

        missing_dataset = [
            name
            for name in DATASET_ORDER
            if not (dataset_path / (name + "_features.pkl")).exists()
        ]
        if missing_dataset:
            rejection_reasons.append(
                str(candidate) + ": missing datasets " + str(missing_dataset)
            )
            continue
        return candidate

    detail = "\n".join(rejection_reasons[:20])
    raise FileNotFoundError(
        "No valid Stage 2 V2.0.2 FINAL run was found.\n" + detail
    )


def load_feature_definitions(stage2_run: Path) -> pd.DataFrame:
    path = stage2_run / "validation_report" / "feature_definition.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "feature",
        "group",
        "unit",
        "description",
        "uses_current_cycle_capacity_label",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError("feature_definition.csv is missing columns: " + str(missing))
    if len(frame) != 39 or frame["feature"].nunique() != 39:
        raise ValueError("Stage 2 feature definitions must contain 39 unique rows")
    if frame["feature"].astype(str).str.contains("ica", case=False).any():
        raise ValueError("obsolete ICA features were found in Stage 2 definitions")
    if not frame["uses_current_cycle_capacity_label"].astype(str).str.lower().eq("no").all():
        raise ValueError("one or more Stage 2 features use the capacity label")
    frame = frame.copy()
    frame.insert(0, "stage2_feature_index", np.arange(len(frame), dtype=int))
    return frame


def validate_stage2_payload(
    dataset: str,
    payload: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "script_build",
        "dataset",
        "feature_names",
        "feature_count",
        "standardized",
        "X",
        "y_soh",
        "battery_id",
        "cycle",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(dataset + " payload missing keys: " + str(missing))
    if str(payload["schema_version"]) != EXPECTED_STAGE2_SCHEMA:
        raise ValueError(dataset + " payload has an unexpected schema")
    if str(payload["script_build"]) != EXPECTED_STAGE2_BUILD:
        raise ValueError(dataset + " payload has an unexpected build")
    if str(payload["dataset"]) != dataset:
        raise ValueError(dataset + " payload dataset name mismatch")
    if bool(payload["standardized"]):
        raise ValueError(dataset + " payload must not be standardized")

    X = np.asarray(payload["X"])
    y = np.asarray(payload["y_soh"])
    battery_id = np.asarray(payload["battery_id"], dtype=object)
    cycle = np.asarray(payload["cycle"])
    feature_names = [str(name) for name in payload["feature_names"]]

    if X.ndim != 2 or X.shape[1] != 39:
        raise ValueError(dataset + " X must have shape (N, 39)")
    if len(feature_names) != 39 or len(set(feature_names)) != 39:
        raise ValueError(dataset + " must contain 39 unique feature names")
    if int(payload["feature_count"]) != 39:
        raise ValueError(dataset + " feature_count is not 39")
    if not (len(y) == len(battery_id) == len(cycle) == len(X)):
        raise ValueError(dataset + " sample-level arrays have inconsistent lengths")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(dataset + " contains non-finite X or SOH")
    if not np.all(np.isfinite(cycle.astype(np.float64))):
        raise ValueError(dataset + " contains non-finite cycles")
    if not np.allclose(cycle.astype(np.float64), np.rint(cycle.astype(np.float64))):
        raise ValueError(dataset + " contains non-integer cycles")

    keys = pd.DataFrame({
        "battery_id": battery_id.astype(str),
        "cycle": cycle.astype(np.int64),
    })
    if keys.duplicated(["battery_id", "cycle"]).any():
        raise ValueError(dataset + " contains duplicate battery-cycle keys")


def load_all_stage2_datasets(
    stage2_run: Path,
    feature_definitions: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    datasets: Dict[str, Dict[str, Any]] = {}
    expected_names = feature_definitions["feature"].astype(str).tolist()

    for dataset in DATASET_ORDER:
        path = stage2_run / "datasets" / (dataset + "_features.pkl")
        print("Loading:", path)
        payload = load_pickle(path)
        validate_stage2_payload(dataset, payload)
        names = [str(name) for name in payload["feature_names"]]
        if names != expected_names:
            raise ValueError(dataset + " feature order differs from feature_definition.csv")
        datasets[dataset] = dict(payload)
        print(
            dataset,
            "X=", np.asarray(payload["X"]).shape,
            "batteries=", len(set(np.asarray(payload["battery_id"]).astype(str))),
        )
    return datasets


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < MIN_BATTERY_SAMPLES:
        return float("nan")
    if np.std(x) < CORRELATION_EPS or np.std(y) < CORRELATION_EPS:
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def safe_spearman(x: np.ndarray, progression: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    progression = np.asarray(progression, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(progression)
    x = x[finite]
    progression = progression[finite]
    if len(x) < MIN_BATTERY_SAMPLES:
        return float("nan")
    if np.std(x) < CORRELATION_EPS or np.std(progression) < CORRELATION_EPS:
        return float("nan")
    x_rank = rankdata(x, method="average")
    progression_rank = rankdata(progression, method="average")
    value = float(np.corrcoef(x_rank, progression_rank)[0, 1])
    return value if np.isfinite(value) else float("nan")


def battery_index_groups(battery_ids: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, battery_id in enumerate(battery_ids.astype(str)):
        groups[str(battery_id)].append(index)
    return [
        (battery_id, np.asarray(groups[battery_id], dtype=np.int64))
        for battery_id in sorted(groups)
    ]


def calculate_battery_correlations(
    datasets: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for dataset in SOURCE_DOMAINS:
        payload = datasets[dataset]
        X = np.asarray(payload["X"], dtype=np.float64)
        y = np.asarray(payload["y_soh"], dtype=np.float64)
        cycles = np.asarray(payload["cycle"], dtype=np.float64)
        battery_ids = np.asarray(payload["battery_id"], dtype=object)

        for battery_id, indices in battery_index_groups(battery_ids):
            order = np.argsort(cycles[indices], kind="stable")
            indices = indices[order]
            y_battery = y[indices]
            cycle_battery = cycles[indices]
            X_battery = X[indices]

            for feature_index, feature_name in enumerate(feature_names):
                feature = X_battery[:, feature_index]
                rows.append({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "sample_count": len(indices),
                    "cycle_min": int(np.min(cycle_battery)),
                    "cycle_max": int(np.max(cycle_battery)),
                    "feature_index": feature_index,
                    "feature": feature_name,
                    "pearson_soh": safe_pearson(feature, y_battery),
                    "spearman_cycle": safe_spearman(feature, cycle_battery),
                })

    frame = pd.DataFrame(rows)
    expected_rows = sum(
        len(set(np.asarray(datasets[name]["battery_id"]).astype(str)))
        for name in SOURCE_DOMAINS
    ) * len(feature_names)
    if len(frame) != expected_rows:
        raise AssertionError(
            "unexpected battery-correlation row count: " + str(len(frame))
        )
    return frame


def summarize_correlations(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    valid_count = len(values)
    if valid_count == 0:
        return {
            "valid_battery_count": 0,
            "median_signed_correlation": 0.0,
            "median_absolute_correlation": 0.0,
            "correlation_q25": 0.0,
            "correlation_q75": 0.0,
            "positive_fraction": 0.0,
            "direction_consistency": 0.0,
            "stable_strength": 0.0,
        }

    median_signed = float(np.median(values))
    median_absolute = float(np.median(np.abs(values)))
    positive_fraction = float(np.mean(values > 0.0))
    direction_consistency = float(np.abs(np.mean(np.sign(values))))
    stable_strength = median_absolute * direction_consistency
    if valid_count < MIN_VALID_BATTERIES_PER_DOMAIN:
        stable_strength = 0.0

    return {
        "valid_battery_count": int(valid_count),
        "median_signed_correlation": median_signed,
        "median_absolute_correlation": median_absolute,
        "correlation_q25": float(np.quantile(values, 0.25)),
        "correlation_q75": float(np.quantile(values, 0.75)),
        "positive_fraction": positive_fraction,
        "direction_consistency": direction_consistency,
        "stable_strength": float(stable_strength),
    }


def calculate_domain_feature_scores(
    battery_correlations: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in SOURCE_DOMAINS:
        domain_frame = battery_correlations[
            battery_correlations["dataset"] == dataset
        ]
        for feature_index, feature_name in enumerate(feature_names):
            feature_frame = domain_frame[
                domain_frame["feature_index"] == feature_index
            ]
            pc = summarize_correlations(feature_frame["pearson_soh"].to_numpy())
            tc = summarize_correlations(feature_frame["spearman_cycle"].to_numpy())
            row: Dict[str, Any] = {
                "dataset": dataset,
                "feature_index": feature_index,
                "feature": feature_name,
            }
            for key, value in pc.items():
                row["pc_" + key] = value
            for key, value in tc.items():
                row["tc_" + key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def harmonic_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        return 0.0
    return float(len(array) / np.sum(1.0 / array))


def cross_domain_direction(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    denominator = float(np.sum(np.abs(array)))
    if len(array) == 0 or denominator <= CORRELATION_EPS:
        return 0.0
    return float(np.abs(np.sum(array)) / denominator)


def calculate_cross_domain_scores(
    domain_scores: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        frame = domain_scores[
            domain_scores["feature_index"] == feature_index
        ].set_index("dataset")
        missing = [name for name in SOURCE_DOMAINS if name not in frame.index]
        if missing:
            raise ValueError(feature_name + " missing domain scores: " + str(missing))

        pc_strength = [
            float(frame.loc[name, "pc_stable_strength"])
            for name in SOURCE_DOMAINS
        ]
        tc_strength = [
            float(frame.loc[name, "tc_stable_strength"])
            for name in SOURCE_DOMAINS
        ]
        pc_signed = [
            float(frame.loc[name, "pc_median_signed_correlation"])
            for name in SOURCE_DOMAINS
        ]
        tc_signed = [
            float(frame.loc[name, "tc_median_signed_correlation"])
            for name in SOURCE_DOMAINS
        ]

        pc_harmonic = harmonic_mean(pc_strength)
        tc_harmonic = harmonic_mean(tc_strength)
        pc_direction = cross_domain_direction(pc_signed)
        tc_direction = cross_domain_direction(tc_signed)
        pc_balanced = pc_harmonic * pc_direction
        tc_balanced = tc_harmonic * tc_direction
        raw_score = ALPHA * pc_balanced + (1.0 - ALPHA) * tc_balanced

        row: Dict[str, Any] = {
            "feature_index": feature_index,
            "feature": feature_name,
            "pc_harmonic_strength": pc_harmonic,
            "pc_cross_domain_direction": pc_direction,
            "pc_balanced_score": pc_balanced,
            "tc_harmonic_strength": tc_harmonic,
            "tc_cross_domain_direction": tc_direction,
            "tc_balanced_score": tc_balanced,
            "raw_pc_tc_score": raw_score,
        }
        for dataset in SOURCE_DOMAINS:
            row[dataset + "_pc_stable_strength"] = float(
                frame.loc[dataset, "pc_stable_strength"]
            )
            row[dataset + "_pc_median_signed"] = float(
                frame.loc[dataset, "pc_median_signed_correlation"]
            )
            row[dataset + "_tc_stable_strength"] = float(
                frame.loc[dataset, "tc_stable_strength"]
            )
            row[dataset + "_tc_median_signed"] = float(
                frame.loc[dataset, "tc_median_signed_correlation"]
            )
        rows.append(row)

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["raw_pc_tc_score", "feature_index"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "raw_rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def correlation_matrices_by_domain(
    battery_correlations: pd.DataFrame,
    value_column: str,
    feature_count: int,
) -> Dict[str, np.ndarray]:
    matrices: Dict[str, np.ndarray] = {}
    for dataset in SOURCE_DOMAINS:
        frame = battery_correlations[
            battery_correlations["dataset"] == dataset
        ]
        pivot = frame.pivot(
            index="battery_id",
            columns="feature_index",
            values=value_column,
        ).reindex(columns=np.arange(feature_count))
        matrices[dataset] = pivot.to_numpy(dtype=np.float64)
    return matrices


def stable_strength_vector(
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(matrix)
    valid_count = finite.sum(axis=0)
    with np.errstate(invalid="ignore"):
        median_signed = np.nanmedian(matrix, axis=0)
        median_absolute = np.nanmedian(np.abs(matrix), axis=0)
    direction = np.abs(
        np.divide(
            np.nansum(np.sign(matrix), axis=0),
            valid_count,
            out=np.zeros(matrix.shape[1], dtype=np.float64),
            where=valid_count > 0,
        )
    )
    stable = np.nan_to_num(median_absolute * direction, nan=0.0)
    stable[valid_count < MIN_VALID_BATTERIES_PER_DOMAIN] = 0.0
    median_signed = np.nan_to_num(median_signed, nan=0.0)
    return stable, median_signed


def balanced_metric_vector(
    stable_vectors: Sequence[np.ndarray],
    signed_vectors: Sequence[np.ndarray],
) -> np.ndarray:
    stable_stack = np.vstack(stable_vectors)
    signed_stack = np.vstack(signed_vectors)

    positive = np.all(stable_stack > 0.0, axis=0)
    harmonic = np.zeros(stable_stack.shape[1], dtype=np.float64)
    harmonic[positive] = stable_stack.shape[0] / np.sum(
        1.0 / stable_stack[:, positive],
        axis=0,
    )
    denominator = np.sum(np.abs(signed_stack), axis=0)
    direction = np.divide(
        np.abs(np.sum(signed_stack, axis=0)),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > CORRELATION_EPS,
    )
    return harmonic * direction


def bootstrap_feature_stability(
    battery_correlations: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    feature_count = len(feature_names)
    pc_matrices = correlation_matrices_by_domain(
        battery_correlations,
        "pearson_soh",
        feature_count,
    )
    tc_matrices = correlation_matrices_by_domain(
        battery_correlations,
        "spearman_cycle",
        feature_count,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    scores = np.zeros((BOOTSTRAP_ITERATIONS, feature_count), dtype=np.float64)
    top_counts = np.zeros(feature_count, dtype=np.int64)

    for iteration in range(BOOTSTRAP_ITERATIONS):
        pc_stable: List[np.ndarray] = []
        pc_signed: List[np.ndarray] = []
        tc_stable: List[np.ndarray] = []
        tc_signed: List[np.ndarray] = []

        for dataset in SOURCE_DOMAINS:
            row_count = pc_matrices[dataset].shape[0]
            sampled = rng.integers(0, row_count, size=row_count)
            pc_s, pc_m = stable_strength_vector(pc_matrices[dataset][sampled])
            tc_s, tc_m = stable_strength_vector(tc_matrices[dataset][sampled])
            pc_stable.append(pc_s)
            pc_signed.append(pc_m)
            tc_stable.append(tc_s)
            tc_signed.append(tc_m)

        pc_balanced = balanced_metric_vector(pc_stable, pc_signed)
        tc_balanced = balanced_metric_vector(tc_stable, tc_signed)
        iteration_score = ALPHA * pc_balanced + (1.0 - ALPHA) * tc_balanced
        scores[iteration] = iteration_score
        order = np.lexsort((np.arange(feature_count), -iteration_score))
        top_counts[order[:TOP_K]] += 1

    rows: List[Dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        values = scores[:, feature_index]
        rows.append({
            "feature_index": feature_index,
            "feature": feature_name,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "score_mean": float(np.mean(values)),
            "score_standard_deviation": float(np.std(values)),
            "score_ci025": float(np.quantile(values, 0.025)),
            "score_median": float(np.median(values)),
            "score_ci975": float(np.quantile(values, 0.975)),
            "raw_top_k_frequency": float(top_counts[feature_index] / BOOTSTRAP_ITERATIONS),
        })
    return pd.DataFrame(rows), scores


def within_battery_spearman_matrix(
    X: np.ndarray,
    cycles: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.float64)
    order = np.argsort(cycles, kind="stable")
    X = X[order]

    if len(X) > REDUNDANCY_MAX_CYCLES_PER_BATTERY:
        indices = np.linspace(
            0,
            len(X) - 1,
            REDUNDANCY_MAX_CYCLES_PER_BATTERY,
        ).astype(np.int64)
        indices = np.unique(indices)
        X = X[indices]

    feature_count = X.shape[1]
    result = np.full((feature_count, feature_count), np.nan, dtype=np.float64)
    ranked = np.zeros_like(X, dtype=np.float64)
    valid_columns: List[int] = []

    for feature_index in range(feature_count):
        values = X[:, feature_index]
        if not np.all(np.isfinite(values)) or np.std(values) < CORRELATION_EPS:
            continue
        ranks = rankdata(values, method="average")
        ranks = (ranks - np.mean(ranks)) / np.std(ranks)
        ranked[:, feature_index] = ranks
        valid_columns.append(feature_index)

    if valid_columns:
        valid_array = np.asarray(valid_columns, dtype=np.int64)
        correlation = (
            ranked[:, valid_array].T @ ranked[:, valid_array]
        ) / len(ranked)
        result[np.ix_(valid_array, valid_array)] = np.clip(
            np.abs(correlation),
            0.0,
            1.0,
        )
    return result


def calculate_redundancy_matrix(
    datasets: Mapping[str, Mapping[str, Any]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    domain_matrices: List[np.ndarray] = []
    battery_counts: Dict[str, int] = {}

    for dataset in SOURCE_DOMAINS:
        payload = datasets[dataset]
        X = np.asarray(payload["X"], dtype=np.float64)
        cycles = np.asarray(payload["cycle"], dtype=np.float64)
        battery_ids = np.asarray(payload["battery_id"], dtype=object)
        battery_matrices: List[np.ndarray] = []

        for _, indices in battery_index_groups(battery_ids):
            if len(indices) < MIN_BATTERY_SAMPLES:
                continue
            matrix = within_battery_spearman_matrix(X[indices], cycles[indices])
            battery_matrices.append(matrix)

        if not battery_matrices:
            raise ValueError(dataset + " has no batteries for redundancy analysis")
        with np.errstate(invalid="ignore"):
            domain_matrix = np.nanmedian(np.stack(battery_matrices), axis=0)
        domain_matrices.append(domain_matrix)
        battery_counts[dataset] = len(battery_matrices)

    with np.errstate(invalid="ignore"):
        redundancy = np.nanmax(np.stack(domain_matrices), axis=0)
    redundancy = np.nan_to_num(redundancy, nan=0.0, posinf=0.0, neginf=0.0)
    redundancy = np.clip(redundancy, 0.0, 1.0)
    redundancy = np.maximum(redundancy, redundancy.T)
    np.fill_diagonal(redundancy, 1.0)
    return redundancy, battery_counts


def redundancy_frame(
    redundancy: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(redundancy, columns=list(feature_names))
    frame.insert(0, "feature", list(feature_names))
    return frame


def greedy_select_feature_indices(
    scores: np.ndarray,
    redundancy: np.ndarray,
) -> Tuple[List[int], List[Dict[str, float]]]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != redundancy.shape[0]:
        raise ValueError("score vector and redundancy matrix dimensions differ")
    if not np.all(np.isfinite(scores)):
        raise ValueError("feature scores contain non-finite values")

    remaining = list(range(len(scores)))
    selected: List[int] = []
    steps: List[Dict[str, float]] = []

    for selection_rank in range(1, TOP_K + 1):
        candidates: List[Tuple[float, float, int, float]] = []
        for feature_index in remaining:
            if selected:
                max_redundancy = float(np.max(redundancy[feature_index, selected]))
            else:
                max_redundancy = 0.0


            if max_redundancy >= MAX_SELECTED_REDUNDANCY:
                continue

            adjusted = float(scores[feature_index]) * (
                1.0 - REDUNDANCY_PENALTY * max_redundancy
            )
            candidates.append(
                (
                    adjusted,
                    float(scores[feature_index]),
                    -feature_index,
                    max_redundancy,
                )
            )

        if not candidates:
            break

        best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        feature_index = -int(best[2])
        selected.append(feature_index)
        remaining.remove(feature_index)
        steps.append({
            "selection_rank": float(selection_rank),
            "feature_index": float(feature_index),
            "raw_score": float(best[1]),
            "maximum_redundancy": float(best[3]),
            "adjusted_score": float(best[0]),
        })

    return selected, steps


def select_top_features(
    ranking: pd.DataFrame,
    redundancy: np.ndarray,
) -> pd.DataFrame:
    score_by_index = {
        int(row["feature_index"]): float(row["raw_pc_tc_score"])
        for row in ranking.to_dict(orient="records")
    }
    name_by_index = {
        int(row["feature_index"]): str(row["feature"])
        for row in ranking.to_dict(orient="records")
    }

    feature_count = len(score_by_index)
    if sorted(score_by_index) != list(range(feature_count)):
        raise ValueError("feature indices must be contiguous from zero")
    score_array = np.asarray(
        [score_by_index[index] for index in range(feature_count)],
        dtype=np.float64,
    )
    selected, steps = greedy_select_feature_indices(score_array, redundancy)
    if len(selected) != TOP_K:
        raise ValueError(
            "Hard redundancy threshold "
            + str(MAX_SELECTED_REDUNDANCY)
            + " permits only "
            + str(len(selected))
            + " features; TOP_K="
            + str(TOP_K)
        )

    rows: List[Dict[str, Any]] = []
    for step in steps:
        feature_index = int(step["feature_index"])
        rows.append({
            "selection_rank": int(step["selection_rank"]),
            "stage2_feature_index": feature_index,
            "feature": name_by_index[feature_index],
            "raw_pc_tc_score": score_by_index[feature_index],
            "maximum_redundancy_with_earlier_selection": float(
                step["maximum_redundancy"]
            ),
            "redundancy_adjusted_selection_score": float(step["adjusted_score"]),
        })
    return pd.DataFrame(rows)


def redundancy_constrained_bootstrap_frequency(
    bootstrap_scores: np.ndarray,
    redundancy: np.ndarray,
) -> Tuple[np.ndarray, int]:
    bootstrap_scores = np.asarray(bootstrap_scores, dtype=np.float64)
    counts = np.zeros(bootstrap_scores.shape[1], dtype=np.int64)
    successful_iterations = 0

    for scores in bootstrap_scores:
        selected, _ = greedy_select_feature_indices(scores, redundancy)
        if len(selected) != TOP_K:
            continue
        counts[np.asarray(selected, dtype=np.int64)] += 1
        successful_iterations += 1

    frequency = counts.astype(np.float64) / float(len(bootstrap_scores))
    return frequency, successful_iterations


def build_selected_manifest(
    selected: pd.DataFrame,
    feature_definitions: pd.DataFrame,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    definition_columns = [
        "stage2_feature_index",
        "feature",
        "group",
        "unit",
        "description",
        "uses_current_cycle_capacity_label",
    ]
    ranking_columns = [
        "feature",
        "raw_rank",
        "pc_balanced_score",
        "tc_balanced_score",
        "pc_cross_domain_direction",
        "tc_cross_domain_direction",
    ]
    bootstrap_columns = [
        "feature",
        "score_mean",
        "score_standard_deviation",
        "score_ci025",
        "score_median",
        "score_ci975",
        "raw_top_k_frequency",
        "redundancy_constrained_top_k_frequency",
        "redundancy_selection_success_iterations",
    ]
    manifest = selected.merge(
        feature_definitions[definition_columns],
        on=["stage2_feature_index", "feature"],
        how="left",
        validate="one_to_one",
    )
    manifest = manifest.merge(
        ranking[ranking_columns],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    manifest = manifest.merge(
        bootstrap[bootstrap_columns],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    manifest.insert(1, "selected_feature_index", np.arange(len(manifest), dtype=int))
    return manifest


def selected_feature_groups(selected_manifest: pd.DataFrame) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for row in selected_manifest.to_dict(orient="records"):
        groups[str(row["group"])].append(int(row["selected_feature_index"]))
    return dict(groups)


def save_selected_datasets(
    datasets: Mapping[str, Mapping[str, Any]],
    selected_manifest: pd.DataFrame,
    stage2_run: Path,
    output_path: Path,
) -> Dict[str, Dict[str, Any]]:
    selected_indices = selected_manifest["stage2_feature_index"].to_numpy(dtype=int)
    selected_names = selected_manifest["feature"].astype(str).tolist()
    groups = selected_feature_groups(selected_manifest)
    summaries: Dict[str, Dict[str, Any]] = {}

    for dataset in DATASET_ORDER:
        source_payload = datasets[dataset]
        X_source = np.asarray(source_payload["X"])
        X_selected = X_source[:, selected_indices].astype(np.float32, copy=False)

        payload = dict(source_payload)
        payload.update({
            "schema_version": STAGE3_SCHEMA_VERSION,
            "script_build": SCRIPT_BUILD,
            "source_stage2_schema": EXPECTED_STAGE2_SCHEMA,
            "source_stage2_build": EXPECTED_STAGE2_BUILD,
            "source_stage2_run": str(stage2_run),
            "feature_selection_fitted_on": list(SOURCE_DOMAINS),
            "target_labels_used_for_selection": False,
            "selected_from_feature_count": int(X_source.shape[1]),
            "feature_count": len(selected_names),
            "feature_names": selected_names,
            "feature_groups": groups,
            "standardized": False,
            "X": X_selected,
        })

        output_file = output_path / "datasets" / (dataset + "_selected_features.pkl")
        write_pickle(payload, output_file)

        summaries[dataset] = {
            "sample_count": int(len(X_selected)),
            "battery_count": int(len(set(np.asarray(payload["battery_id"]).astype(str)))),
            "feature_shape": list(X_selected.shape),
            "feature_count": len(selected_names),
            "finite_feature_count": int(np.isfinite(X_selected).sum()),
            "expected_finite_feature_count": int(X_selected.size),
            "standardized": False,
            "output_file": str(output_file.relative_to(output_path)),
        }
    return summaries


def validate_stage3_outputs(
    feature_names: Sequence[str],
    battery_correlations: pd.DataFrame,
    domain_scores: pd.DataFrame,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    redundancy: np.ndarray,
    selected_manifest: pd.DataFrame,
    dataset_summaries: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], List[str]]:
    fatal_checks: List[str] = []
    warnings: List[str] = []

    if len(feature_names) != 39 or len(set(feature_names)) != 39:
        fatal_checks.append("input feature names are not 39 unique values")
    if len(ranking) != 39 or ranking["feature"].nunique() != 39:
        fatal_checks.append("feature ranking is not 39 unique rows")
    if len(selected_manifest) != TOP_K:
        fatal_checks.append("selected feature count is not " + str(TOP_K))
    if selected_manifest["feature"].nunique() != TOP_K:
        fatal_checks.append("selected feature names are not unique")
    if not set(selected_manifest["feature"]).issubset(set(feature_names)):
        fatal_checks.append("selected features are not a subset of Stage 2 features")
    selected_raw_scores = pd.to_numeric(
        selected_manifest["raw_pc_tc_score"],
        errors="coerce",
    ).to_numpy()
    if not np.all(np.isfinite(selected_raw_scores)):
        fatal_checks.append("selected features contain non-finite PC-TC scores")
    elif np.any(selected_raw_scores <= 0.0):
        fatal_checks.append(
            "fewer than TOP_K candidates have positive cross-domain PC-TC scores"
        )
    if set(battery_correlations["dataset"].astype(str)) != set(SOURCE_DOMAINS):
        fatal_checks.append("battery correlation table contains non-source domains")
    if set(domain_scores["dataset"].astype(str)) != set(SOURCE_DOMAINS):
        fatal_checks.append("domain score table contains non-source domains")

    correlation_columns = ["pearson_soh", "spearman_cycle"]
    for column in correlation_columns:
        values = pd.to_numeric(battery_correlations[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) and ((finite < -1.0000001) | (finite > 1.0000001)).any():
            fatal_checks.append(column + " contains values outside [-1, 1]")

    score_columns = [
        "pc_balanced_score",
        "tc_balanced_score",
        "raw_pc_tc_score",
    ]
    for column in score_columns:
        values = pd.to_numeric(ranking[column], errors="coerce").to_numpy()
        if not np.all(np.isfinite(values)):
            fatal_checks.append(column + " contains non-finite values")
        elif np.any(values < -1e-12) or np.any(values > 1.0000001):
            fatal_checks.append(column + " contains values outside [0, 1]")

    bootstrap_numeric = bootstrap.select_dtypes(include=[np.number]).to_numpy()
    if not np.all(np.isfinite(bootstrap_numeric)):
        fatal_checks.append("bootstrap stability contains non-finite values")
    success_values = pd.to_numeric(
        bootstrap["redundancy_selection_success_iterations"],
        errors="coerce",
    ).dropna().unique()
    if len(success_values) != 1 or int(success_values[0]) != BOOTSTRAP_ITERATIONS:
        fatal_checks.append(
            "hard-gated Top-K selection did not succeed in every bootstrap iteration"
        )
    if redundancy.shape != (39, 39):
        fatal_checks.append("redundancy matrix does not have shape (39, 39)")
    elif not np.all(np.isfinite(redundancy)):
        fatal_checks.append("redundancy matrix contains non-finite values")
    elif not np.allclose(redundancy, redundancy.T, atol=1e-10):
        fatal_checks.append("redundancy matrix is not symmetric")

    selected_indices = selected_manifest["stage2_feature_index"].to_numpy(dtype=int)
    if len(selected_indices) > 1:
        submatrix = redundancy[np.ix_(selected_indices, selected_indices)].copy()
        np.fill_diagonal(submatrix, 0.0)
        maximum_selected_redundancy = float(np.max(submatrix))
        if maximum_selected_redundancy >= MAX_SELECTED_REDUNDANCY:
            fatal_checks.append(
                "selected features violate the hard redundancy threshold: "
                + str(maximum_selected_redundancy)
            )

    minimum_frequency = float(
        selected_manifest["redundancy_constrained_top_k_frequency"].min()
    )
    if minimum_frequency < 0.20:
        warnings.append(
            "at least one selected feature has redundancy-constrained bootstrap "
            "Top-K frequency below 0.20; inspect the stability report"
        )

    for dataset in DATASET_ORDER:
        summary = dataset_summaries[dataset]
        if list(summary["feature_shape"])[1] != TOP_K:
            fatal_checks.append(dataset + " selected X does not have TOP_K columns")
        if summary["finite_feature_count"] != summary["expected_finite_feature_count"]:
            fatal_checks.append(dataset + " selected X contains non-finite values")
        if bool(summary["standardized"]):
            fatal_checks.append(dataset + " selected X was standardized")

    return fatal_checks, warnings


def main() -> None:
    print("============================================================")
    print(" Stage 3 V3.0.1 FINAL cross-domain feature selection started")
    print(" Script build:", SCRIPT_BUILD)
    print("============================================================")

    stage2_run = find_latest_stage2_run()
    print("Stage 2 input:", stage2_run)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_path = OUTPUT_ROOT / run_name
    report_path = run_path / "validation_report"
    dataset_path = run_path / "datasets"
    report_path.mkdir(parents=True, exist_ok=False)
    dataset_path.mkdir(parents=True, exist_ok=False)

    feature_definitions = load_feature_definitions(stage2_run)
    feature_names = feature_definitions["feature"].astype(str).tolist()
    datasets = load_all_stage2_datasets(stage2_run, feature_definitions)

    print("\nCalculating within-battery PC and TC...")
    battery_correlations = calculate_battery_correlations(datasets, feature_names)

    print("Calculating battery-balanced source-domain scores...")
    domain_scores = calculate_domain_feature_scores(
        battery_correlations,
        feature_names,
    )
    ranking = calculate_cross_domain_scores(domain_scores, feature_names)

    print("Running battery-level bootstrap stability analysis...")
    bootstrap, bootstrap_scores = bootstrap_feature_stability(
        battery_correlations,
        feature_names,
    )

    print("Calculating battery-balanced feature redundancy...")
    redundancy, redundancy_battery_counts = calculate_redundancy_matrix(datasets)
    constrained_frequency, bootstrap_selection_successes = (
        redundancy_constrained_bootstrap_frequency(
            bootstrap_scores,
            redundancy,
        )
    )
    bootstrap["redundancy_constrained_top_k_frequency"] = constrained_frequency
    bootstrap["redundancy_selection_success_iterations"] = (
        bootstrap_selection_successes
    )
    ranking = ranking.merge(
        bootstrap,
        on=["feature_index", "feature"],
        how="left",
        validate="one_to_one",
    )
    selected = select_top_features(ranking, redundancy)
    selected_manifest = build_selected_manifest(
        selected,
        feature_definitions,
        ranking,
        bootstrap,
    )

    selected_names = selected_manifest["feature"].astype(str).tolist()
    dataset_summaries = save_selected_datasets(
        datasets,
        selected_manifest,
        stage2_run,
        run_path,
    )


    selected_text = "\n".join(selected_names) + "\n"
    (run_path / "selected_features.txt").write_text(selected_text, encoding="utf-8")
    write_csv(selected_manifest, run_path / "selected_feature_manifest.csv")
    write_csv(ranking, run_path / "feature_ranking.csv")


    write_csv(battery_correlations, report_path / "battery_correlations.csv")
    write_csv(domain_scores, report_path / "domain_feature_scores.csv")
    write_csv(ranking, report_path / "feature_ranking.csv")
    write_csv(bootstrap, report_path / "bootstrap_stability.csv")
    write_csv(selected_manifest, report_path / "selected_feature_manifest.csv")
    write_csv(
        redundancy_frame(redundancy, feature_names),
        report_path / "redundancy_matrix.csv",
    )
    (report_path / "selected_features.txt").write_text(
        selected_text,
        encoding="utf-8",
    )

    fatal_checks, warnings = validate_stage3_outputs(
        feature_names,
        battery_correlations,
        domain_scores,
        ranking,
        bootstrap,
        redundancy,
        selected_manifest,
        dataset_summaries,
    )
    validation_status = "PASS" if not fatal_checks else "FAIL"

    selected_indices = selected_manifest["stage2_feature_index"].to_numpy(dtype=int)
    selected_redundancy = redundancy[np.ix_(selected_indices, selected_indices)].copy()
    np.fill_diagonal(selected_redundancy, 0.0)
    maximum_selected_redundancy = float(np.max(selected_redundancy))

    config = {
        "schema_version": STAGE3_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage2_run": str(stage2_run),
        "output_run": str(run_path),
        "expected_stage2_schema": EXPECTED_STAGE2_SCHEMA,
        "expected_stage2_build": EXPECTED_STAGE2_BUILD,
        "source_domains_used_for_selection": list(SOURCE_DOMAINS),
        "target_domains_excluded_from_scoring": list(TARGET_DOMAINS),
        "target_labels_used_for_selection": False,
        "candidate_feature_count": 39,
        "selected_feature_count": TOP_K,
        "alpha_pc": ALPHA,
        "alpha_tc": 1.0 - ALPHA,
        "minimum_battery_samples": MIN_BATTERY_SAMPLES,
        "minimum_valid_batteries_per_domain": MIN_VALID_BATTERIES_PER_DOMAIN,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "random_seed": RANDOM_SEED,
        "redundancy_penalty": REDUNDANCY_PENALTY,
        "hard_redundancy_threshold": MAX_SELECTED_REDUNDANCY,
        "redundancy_max_cycles_per_battery": REDUNDANCY_MAX_CYCLES_PER_BATTERY,
        "standardization_performed": False,
        "pc_definition": "within-battery Pearson(feature, SOH)",
        "tc_definition": "within-battery Spearman(feature, cycle progression)",
        "domain_score_definition": (
            "median(abs(correlation)) * abs(mean(sign(correlation)))"
        ),
        "cross_domain_score_definition": (
            "harmonic_mean(domain stable strengths) * "
            "abs(sum(domain median signed correlations)) / "
            "sum(abs(domain median signed correlations))"
        ),
        "fusion_score_definition": (
            "ALPHA * PC_balanced + (1 - ALPHA) * TC_balanced"
        ),
        "selection_definition": (
            "first require maximum redundancy with every earlier selection "
            "< MAX_SELECTED_REDUNDANCY; then greedily maximize raw_score * "
            "(1 - REDUNDANCY_PENALTY * maximum redundancy)"
        ),
        "bootstrap_selection_definition": (
            "repeat the complete hard-gated redundancy-aware Top-K selector "
            "for every battery-bootstrap score vector"
        ),
    }

    summary = {
        "schema_version": STAGE3_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "validation_status": validation_status,
        "fatal_checks": fatal_checks,
        "warnings": warnings,
        "source_domains_used_for_selection": list(SOURCE_DOMAINS),
        "target_labels_used_for_selection": False,
        "candidate_feature_count": 39,
        "selected_feature_count": len(selected_names),
        "selected_features": selected_names,
        "selected_feature_groups": (
            selected_manifest.groupby("group").size().astype(int).to_dict()
        ),
        "source_battery_counts_for_redundancy": redundancy_battery_counts,
        "hard_redundancy_threshold": MAX_SELECTED_REDUNDANCY,
        "maximum_pairwise_redundancy_among_selected": maximum_selected_redundancy,
        "minimum_selected_raw_bootstrap_top_k_frequency": float(
            selected_manifest["raw_top_k_frequency"].min()
        ),
        "minimum_selected_redundancy_constrained_bootstrap_top_k_frequency": float(
            selected_manifest["redundancy_constrained_top_k_frequency"].min()
        ),
        "redundancy_selection_success_iterations": int(
            bootstrap_selection_successes
        ),
        "datasets": dataset_summaries,
    }
    write_json(config, report_path / "processing_config.json")
    write_json(summary, report_path / "processing_summary.json")

    readme = (
        "Stage 3 V3.0.1 FINAL validation report\n"
        "=======================================\n\n"
        "Validation status: " + validation_status + "\n"
        "Script build: " + SCRIPT_BUILD + "\n"
        "Stage 2 input: " + str(stage2_run) + "\n"
        "Stage 3 output: " + str(run_path) + "\n\n"
        "Feature selection was fitted only on XJTU and HUST.\n"
        "TJU_NCM and TJU_NCA labels were not used for scoring.\n"
        "Every selected feature pair must have redundancy below 0.99.\n"
        "Bootstrap stability repeats the complete redundancy-aware selector.\n"
        "No feature standardization was performed in Stage 3.\n"
        "Please return STAGE3_VALIDATION_REPORT.zip for independent review.\n"
        "Do not start Stage 4 until the validation report has been reviewed.\n"
    )
    (report_path / "README.txt").write_text(readme, encoding="utf-8")

    zip_path = run_path / "STAGE3_VALIDATION_REPORT.zip"
    zip_validation_report(report_path, zip_path)

    print("============================================================")
    print("Validation status:", validation_status)
    print("Selected feature count:", len(selected_names))
    for rank, name in enumerate(selected_names, 1):
        print(str(rank).rjust(2), name)
    print("Validation report:", zip_path)
    print("============================================================")

    if fatal_checks:
        raise RuntimeError(
            "Stage 3 validation failed:\n- " + "\n- ".join(fatal_checks)
        )


if __name__ == "__main__":
    main()
