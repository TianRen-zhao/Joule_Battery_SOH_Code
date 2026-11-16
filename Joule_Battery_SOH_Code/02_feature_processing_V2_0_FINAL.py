from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


STAGE2_SCHEMA_VERSION = "2.0.2"
SCRIPT_BUILD = "STAGE2_V2_0_2_FINAL_20260901"

EXPECTED_STAGE1_SCHEMA = "2.2"
EXPECTED_STAGE1_BUILD = "STAGE1_V2_2_FINAL_20260901"

BASE_PATH = Path(os.environ.get("CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"))
STAGE1_ROOT = BASE_PATH / "DATAPROCESSING_V2_2"
OUTPUT_ROOT = BASE_PATH / "FEATUREPROCESSING_V2_0"


STAGE1_RUN_PATH: Optional[Path] = None

DATASET_ORDER = ("XJTU", "HUST", "TJU_NCM", "TJU_NCA")


NORMALIZED_CAPACITY_AXIS_POINTS = 101
MIN_UNIQUE_CAPACITY_POINTS = 16
MIN_CAPACITY_SPAN_AH = 1e-8

SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 2
GRADIENT_EDGE_POINTS = 5

SOH_WARNING_LOWER = 0.50
SOH_WARNING_UPPER = 1.20

Q_AXIS = np.linspace(
    0.0,
    1.0,
    NORMALIZED_CAPACITY_AXIS_POINTS,
    dtype=np.float64,
)


def build_feature_definitions() -> List[Dict[str, str]]:
    definitions: List[Dict[str, str]] = []

    voltage_specs = [
        ("voltage_mean_v", "Mean voltage on normalized throughput axis", "V"),
        ("voltage_std_v", "Voltage standard deviation on normalized throughput axis", "V"),
        ("voltage_range_v", "Maximum minus minimum voltage", "V"),
        ("voltage_q10_v", "Voltage at 10% normalized phase throughput", "V"),
        ("voltage_q25_v", "Voltage at 25% normalized phase throughput", "V"),
        ("voltage_q50_v", "Voltage at 50% normalized phase throughput", "V"),
        ("voltage_q75_v", "Voltage at 75% normalized phase throughput", "V"),
        ("voltage_q90_v", "Voltage at 90% normalized phase throughput", "V"),
        ("voltage_iqr_v", "Voltage at 75% minus voltage at 25% throughput", "V"),
        (
            "voltage_early_slope_v_per_q",
            "Voltage slope from 10% to 25% normalized throughput",
            "V/q",
        ),
        (
            "voltage_mid_slope_v_per_q",
            "Voltage slope from 25% to 75% normalized throughput",
            "V/q",
        ),
        (
            "voltage_late_slope_v_per_q",
            "Voltage slope from 75% to 90% normalized throughput",
            "V/q",
        ),
        (
            "voltage_curvature_v_per_q",
            "Late voltage slope minus early voltage slope",
            "V/q",
        ),
        (
            "voltage_gradient_abs_p90_v_per_q",
            "90th percentile of absolute smoothed dV/dQ_norm",
            "V/q",
        ),
        (
            "voltage_gradient_iqr_v_per_q",
            "Interquartile range of smoothed dV/dQ_norm",
            "V/q",
        ),
    ]

    for phase in ("charge", "discharge"):
        for suffix, description, unit in voltage_specs:
            definitions.append({
                "feature": f"{phase}_{suffix}",
                "group": f"{phase}_voltage_shape",
                "unit": unit,
                "description": description,
                "uses_current_cycle_capacity_label": "no",
            })

    current_specs = [
        (
            "c_rate_mean",
            "Mean absolute current divided by the fixed initial capacity Q0",
            "1/h",
        ),
        (
            "c_rate_std",
            "Standard deviation of absolute current divided by Q0",
            "1/h",
        ),
        (
            "c_rate_peak_factor",
            "Maximum absolute C-rate divided by RMS C-rate",
            "dimensionless",
        ),
    ]
    for phase in ("charge", "discharge"):
        for suffix, description, unit in current_specs:
            definitions.append({
                "feature": f"{phase}_{suffix}",
                "group": "current_stress",
                "unit": unit,
                "description": description,
                "uses_current_cycle_capacity_label": "no",
            })

    hysteresis_specs = [
        (
            "voltage_hysteresis_low_soc_v",
            "Charge voltage at q=0.25 minus discharge voltage at q=0.75",
        ),
        (
            "voltage_hysteresis_mid_soc_v",
            "Charge voltage at q=0.50 minus discharge voltage at q=0.50",
        ),
        (
            "voltage_hysteresis_high_soc_v",
            "Charge voltage at q=0.75 minus discharge voltage at q=0.25",
        ),
    ]
    for name, description in hysteresis_specs:
        definitions.append({
            "feature": name,
            "group": "charge_discharge_relation",
            "unit": "V",
            "description": description,
            "uses_current_cycle_capacity_label": "no",
        })

    names = [item["feature"] for item in definitions]
    if len(names) != 39:
        raise AssertionError(f"expected 39 features, found {len(names)}")
    if len(names) != len(set(names)):
        raise AssertionError("candidate feature names are not unique")
    return definitions


FEATURE_DEFINITIONS = build_feature_definitions()
FEATURE_NAMES = [item["feature"] for item in FEATURE_DEFINITIONS]
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

FEATURE_GROUPS: Dict[str, List[int]] = defaultdict(list)
for feature_index, definition in enumerate(FEATURE_DEFINITIONS):
    FEATURE_GROUPS[definition["group"]].append(feature_index)
FEATURE_GROUPS = dict(FEATURE_GROUPS)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


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


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def normalize_relative_path(relative_path: Any) -> Path:
    text = str(relative_path).strip().replace("\\", "/")
    return Path(*[part for part in text.split("/") if part])


def finite_positive(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric > 0.0)


def find_latest_stage1_run() -> Path:
    if STAGE1_RUN_PATH is not None:
        candidates = [Path(STAGE1_RUN_PATH)]
    else:
        if not STAGE1_ROOT.exists():
            raise FileNotFoundError(
                f"Stage 1 root does not exist: {STAGE1_ROOT}"
            )
        candidates = sorted(
            [path for path in STAGE1_ROOT.glob("run_*") if path.is_dir()],
            reverse=True,
        )

    rejection_reasons: List[str] = []
    for candidate in candidates:
        config_path = candidate / "processing_config.json"
        battery_manifest = candidate / "manifests" / "battery_manifest.csv"
        processed_path = candidate / "processed_batteries"
        if not config_path.exists() or not battery_manifest.exists() or not processed_path.exists():
            rejection_reasons.append(f"{candidate}: missing required Stage 1 output")
            continue
        try:
            with config_path.open("r", encoding="utf-8-sig") as file:
                config = json.load(file)
        except Exception as exc:
            rejection_reasons.append(f"{candidate}: invalid config: {exc}")
            continue
        if str(config.get("schema_version")) != EXPECTED_STAGE1_SCHEMA:
            rejection_reasons.append(
                f"{candidate}: schema={config.get('schema_version')}"
            )
            continue
        if str(config.get("script_build")) != EXPECTED_STAGE1_BUILD:
            rejection_reasons.append(
                f"{candidate}: build={config.get('script_build')}"
            )
            continue
        return candidate

    detail = "\n".join(rejection_reasons[:10])
    raise FileNotFoundError(
        "No valid Stage 1 V2.2 FINAL run was found.\n" + detail
    )


def read_battery_manifest(stage1_run: Path) -> pd.DataFrame:
    path = stage1_run / "manifests" / "battery_manifest.csv"
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "dataset",
        "chemistry",
        "battery_id",
        "valid_cycles",
        "soh_label_eligible_cycles",
        "curve_quality_eligible_cycles",
        "processed_file",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Stage 1 battery manifest is missing columns: {missing}")
    if frame.duplicated(["dataset", "battery_id"]).any():
        raise ValueError("Stage 1 battery manifest contains duplicate battery IDs")
    unexpected = sorted(set(frame["dataset"].astype(str)) - set(DATASET_ORDER))
    if unexpected:
        raise ValueError(f"unexpected datasets in Stage 1 manifest: {unexpected}")
    return frame


def clean_numeric_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{name} cannot be converted to float array: {exc}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def smooth_voltage(voltage: np.ndarray) -> np.ndarray:
    window = min(SAVGOL_WINDOW, len(voltage))
    if window % 2 == 0:
        window -= 1
    if window <= SAVGOL_POLYORDER:
        return voltage.copy()
    return savgol_filter(
        voltage,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        mode="interp",
    )


def phase_on_normalized_capacity_axis(
    phase_data: Mapping[str, Any],
    phase_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    voltage = clean_numeric_array(
        phase_data.get("voltage_v"),
        f"{phase_name}.voltage_v",
    )
    current = clean_numeric_array(
        phase_data.get("current_a"),
        f"{phase_name}.current_a",
    )
    throughput = clean_numeric_array(
        phase_data.get("capacity_throughput_ah"),
        f"{phase_name}.capacity_throughput_ah",
    )

    length = min(len(voltage), len(current), len(throughput))
    if length < MIN_UNIQUE_CAPACITY_POINTS:
        raise ValueError(
            f"{phase_name} has only {length} aligned resampled points"
        )

    voltage = voltage[:length]
    current = current[:length]
    throughput = throughput[:length]
    finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(throughput)
    if int(finite.sum()) < MIN_UNIQUE_CAPACITY_POINTS:
        raise ValueError(f"{phase_name} has insufficient finite points")

    voltage = voltage[finite]
    current = current[finite]
    throughput = np.maximum.accumulate(throughput[finite])
    span = float(throughput[-1] - throughput[0])
    if not np.isfinite(span) or span <= MIN_CAPACITY_SPAN_AH:
        raise ValueError(f"{phase_name} has zero capacity-throughput span")

    normalized_q = (throughput - throughput[0]) / span
    unique_q, unique_indices = np.unique(normalized_q, return_index=True)
    if len(unique_q) < MIN_UNIQUE_CAPACITY_POINTS:
        raise ValueError(
            f"{phase_name} has only {len(unique_q)} unique throughput points"
        )

    voltage_unique = voltage[unique_indices]
    voltage_q = np.interp(Q_AXIS, unique_q, voltage_unique)
    if not np.all(np.isfinite(voltage_q)):
        raise ValueError(f"{phase_name} voltage interpolation is non-finite")

    finite_current = current[np.isfinite(current)]
    if len(finite_current) < MIN_UNIQUE_CAPACITY_POINTS:
        raise ValueError(f"{phase_name} current has insufficient finite points")
    return voltage_q.astype(np.float64), finite_current.astype(np.float64)


def voltage_shape_features(
    phase_name: str,
    voltage_smooth: np.ndarray,
) -> Dict[str, float]:
    v10 = float(np.interp(0.10, Q_AXIS, voltage_smooth))
    v25 = float(np.interp(0.25, Q_AXIS, voltage_smooth))
    v50 = float(np.interp(0.50, Q_AXIS, voltage_smooth))
    v75 = float(np.interp(0.75, Q_AXIS, voltage_smooth))
    v90 = float(np.interp(0.90, Q_AXIS, voltage_smooth))

    early_slope = (v25 - v10) / 0.15
    mid_slope = (v75 - v25) / 0.50
    late_slope = (v90 - v75) / 0.15

    dv_dq = np.gradient(voltage_smooth, Q_AXIS)
    if len(dv_dq) > 2 * GRADIENT_EDGE_POINTS:
        valid_indices = np.arange(
            GRADIENT_EDGE_POINTS,
            len(dv_dq) - GRADIENT_EDGE_POINTS,
        )
    else:
        valid_indices = np.arange(len(dv_dq))
    valid_indices = valid_indices[np.isfinite(dv_dq[valid_indices])]
    if len(valid_indices) == 0:
        raise ValueError(f"{phase_name} has no valid voltage gradient")

    gradient_values = dv_dq[valid_indices]
    gradient_abs_p90 = float(np.quantile(np.abs(gradient_values), 0.90))
    gradient_iqr = float(
        np.quantile(gradient_values, 0.75)
        - np.quantile(gradient_values, 0.25)
    )

    prefix = f"{phase_name}_"
    return {
        prefix + "voltage_mean_v": float(np.mean(voltage_smooth)),
        prefix + "voltage_std_v": float(np.std(voltage_smooth)),
        prefix + "voltage_range_v": float(np.max(voltage_smooth) - np.min(voltage_smooth)),
        prefix + "voltage_q10_v": v10,
        prefix + "voltage_q25_v": v25,
        prefix + "voltage_q50_v": v50,
        prefix + "voltage_q75_v": v75,
        prefix + "voltage_q90_v": v90,
        prefix + "voltage_iqr_v": v75 - v25,
        prefix + "voltage_early_slope_v_per_q": early_slope,
        prefix + "voltage_mid_slope_v_per_q": mid_slope,
        prefix + "voltage_late_slope_v_per_q": late_slope,
        prefix + "voltage_curvature_v_per_q": late_slope - early_slope,
        prefix + "voltage_gradient_abs_p90_v_per_q": gradient_abs_p90,
        prefix + "voltage_gradient_iqr_v_per_q": gradient_iqr,
    }


def current_stress_features(
    phase_name: str,
    current_a: np.ndarray,
    reference_capacity_ah: float,
) -> Dict[str, float]:
    if not finite_positive(reference_capacity_ah):
        raise ValueError("reference capacity is not positive")
    c_rate = np.abs(current_a) / float(reference_capacity_ah)
    c_rate = c_rate[np.isfinite(c_rate)]
    if len(c_rate) == 0:
        raise ValueError(f"{phase_name} C-rate is empty")
    rms = float(np.sqrt(np.mean(np.square(c_rate))))
    peak_factor = float(np.max(c_rate) / max(rms, 1e-12))
    return {
        f"{phase_name}_c_rate_mean": float(np.mean(c_rate)),
        f"{phase_name}_c_rate_std": float(np.std(c_rate)),
        f"{phase_name}_c_rate_peak_factor": peak_factor,
    }


def extract_cycle_features(
    cycle: Mapping[str, Any],
    reference_capacity_ah: float,
) -> np.ndarray:
    charge_voltage, charge_current = phase_on_normalized_capacity_axis(
        cycle["charge"],
        "charge",
    )
    discharge_voltage, discharge_current = phase_on_normalized_capacity_axis(
        cycle["discharge"],
        "discharge",
    )


    charge_voltage_smooth = smooth_voltage(charge_voltage)
    discharge_voltage_smooth = smooth_voltage(discharge_voltage)

    feature_map: Dict[str, float] = {}
    feature_map.update(voltage_shape_features("charge", charge_voltage_smooth))
    feature_map.update(
        voltage_shape_features("discharge", discharge_voltage_smooth)
    )
    feature_map.update(
        current_stress_features(
            "charge",
            charge_current,
            reference_capacity_ah,
        )
    )
    feature_map.update(
        current_stress_features(
            "discharge",
            discharge_current,
            reference_capacity_ah,
        )
    )

    charge_v25 = float(np.interp(0.25, Q_AXIS, charge_voltage_smooth))
    charge_v50 = float(np.interp(0.50, Q_AXIS, charge_voltage_smooth))
    charge_v75 = float(np.interp(0.75, Q_AXIS, charge_voltage_smooth))
    discharge_v25 = float(np.interp(0.25, Q_AXIS, discharge_voltage_smooth))
    discharge_v50 = float(np.interp(0.50, Q_AXIS, discharge_voltage_smooth))
    discharge_v75 = float(np.interp(0.75, Q_AXIS, discharge_voltage_smooth))

    feature_map["voltage_hysteresis_low_soc_v"] = charge_v25 - discharge_v75
    feature_map["voltage_hysteresis_mid_soc_v"] = charge_v50 - discharge_v50
    feature_map["voltage_hysteresis_high_soc_v"] = charge_v75 - discharge_v25

    missing = [name for name in FEATURE_NAMES if name not in feature_map]
    extra = sorted(set(feature_map) - set(FEATURE_NAMES))
    if missing or extra:
        raise AssertionError(f"feature map mismatch; missing={missing}, extra={extra}")

    vector = np.asarray([feature_map[name] for name in FEATURE_NAMES], dtype=np.float64)
    if vector.shape != (39,):
        raise AssertionError(f"feature vector shape is {vector.shape}, expected (39,)")
    if not np.all(np.isfinite(vector)):
        indices = np.where(~np.isfinite(vector))[0].tolist()
        names = [FEATURE_NAMES[index] for index in indices]
        raise ValueError(f"non-finite features: {names}")
    return vector.astype(np.float32)


def stage1_joint_eligible(cycle: Mapping[str, Any]) -> bool:
    return bool(cycle.get("soh_label_eligible")) and bool(
        cycle.get("curve_quality_eligible")
    )


def validate_stage1_payload(
    payload: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "dataset",
        "chemistry",
        "battery_id",
        "condition_group",
        "cycles",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(f"Stage 1 battery payload is missing keys: {missing}")
    if str(payload["schema_version"]) != EXPECTED_STAGE1_SCHEMA:
        raise ValueError(
            f"battery payload schema={payload['schema_version']}, "
            f"expected={EXPECTED_STAGE1_SCHEMA}"
        )
    if str(payload["dataset"]) != str(manifest_row["dataset"]):
        raise ValueError("payload dataset does not match battery manifest")
    if str(payload["battery_id"]) != str(manifest_row["battery_id"]):
        raise ValueError("payload battery_id does not match battery manifest")
    if not isinstance(payload["cycles"], Sequence):
        raise TypeError("payload['cycles'] must be a sequence")


def choose_reference_cycle(
    cycles: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    eligible = [cycle for cycle in cycles if stage1_joint_eligible(cycle)]
    eligible.sort(key=lambda item: int(item["cycle"]))
    if not eligible:
        raise ValueError("battery has no Stage-1-qualified cycle for Q0")
    reference = eligible[0]
    if not finite_positive(reference.get("capacity_ah")):
        raise ValueError("first qualified cycle has invalid capacity")
    return reference


def safe_optional_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


SAMPLE_MANIFEST_FIELDS = [
    "dataset",
    "chemistry",
    "battery_id",
    "cycle",
    "condition_group",
    "time_source",
    "capacity_ah",
    "reference_capacity_ah",
    "reference_cycle",
    "soh",
    "ambient_temperature_c",
    "temperature_available",
    "dataset_row_index",
    "output_dataset_file",
]

BATTERY_REFERENCE_FIELDS = [
    "dataset",
    "chemistry",
    "battery_id",
    "condition_group",
    "stage1_valid_cycles",
    "stage1_joint_eligible_cycles",
    "stage1_label_excluded_cycles",
    "stage1_curve_excluded_cycles",
    "reference_cycle",
    "reference_capacity_ah",
    "reference_capacity_source",
    "extracted_samples",
    "feature_extraction_failures",
]

QUALITY_ISSUE_FIELDS = [
    "dataset",
    "battery_id",
    "cycle",
    "reason",
    "detail",
]

FEATURE_STATISTIC_FIELDS = [
    "dataset",
    "feature_index",
    "feature",
    "group",
    "unit",
    "sample_count",
    "finite_count",
    "minimum",
    "percentile_01",
    "median",
    "percentile_99",
    "maximum",
    "mean",
    "standard_deviation",
    "constant_feature",
]


def create_empty_buffers() -> Dict[str, Dict[str, Any]]:
    buffers: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        buffers[dataset] = {
            "X_chunks": [],
            "y_soh": [],
            "capacity_ah": [],
            "reference_capacity_ah": [],
            "reference_cycle": [],
            "battery_id": [],
            "cycle": [],
            "condition_group": [],
            "time_source": [],
            "ambient_temperature_c": [],
            "temperature_available": [],
            "chemistry_values": set(),
        }
    return buffers


def finalize_dataset_payload(
    dataset: str,
    buffer: Dict[str, Any],
    stage1_run: Path,
) -> Dict[str, Any]:
    chunks: List[np.ndarray] = buffer["X_chunks"]
    if chunks:
        X = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    else:
        X = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray(buffer["y_soh"], dtype=np.float32)

    if X.shape != (len(y), len(FEATURE_NAMES)):
        raise AssertionError(
            f"{dataset} X shape {X.shape} does not match y length {len(y)}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{dataset} contains non-finite features")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{dataset} contains non-finite SOH labels")

    chemistry_values = sorted(str(item) for item in buffer["chemistry_values"])
    chemistry: Any = chemistry_values[0] if len(chemistry_values) == 1 else chemistry_values

    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "source_stage1_schema": EXPECTED_STAGE1_SCHEMA,
        "source_stage1_build": EXPECTED_STAGE1_BUILD,
        "source_stage1_run": str(stage1_run),
        "dataset": dataset,
        "chemistry": chemistry,
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": dict(FEATURE_GROUPS),
        "feature_count": len(FEATURE_NAMES),
        "standardized": False,
        "label_definition": "SOH_i = capacity_ah_i / first_qualified_capacity_ah_Q0",
        "sample_filter": "soh_label_eligible AND curve_quality_eligible",
        "X": X,
        "y_soh": y,
        "capacity_ah": np.asarray(buffer["capacity_ah"], dtype=np.float32),
        "reference_capacity_ah": np.asarray(
            buffer["reference_capacity_ah"], dtype=np.float32
        ),
        "reference_cycle": np.asarray(buffer["reference_cycle"], dtype=np.int32),
        "battery_id": np.asarray(buffer["battery_id"], dtype=object),
        "cycle": np.asarray(buffer["cycle"], dtype=np.int32),
        "condition_group": np.asarray(buffer["condition_group"], dtype=object),
        "time_source": np.asarray(buffer["time_source"], dtype=object),
        "ambient_temperature_c": np.asarray(
            buffer["ambient_temperature_c"], dtype=np.float32
        ),
        "temperature_available": np.asarray(
            buffer["temperature_available"], dtype=bool
        ),
    }


def feature_statistics_rows(
    dataset: str,
    X: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    definition_by_name = {
        definition["feature"]: definition for definition in FEATURE_DEFINITIONS
    }
    for index, name in enumerate(FEATURE_NAMES):
        values = X[:, index].astype(np.float64, copy=False)
        finite = values[np.isfinite(values)]
        definition = definition_by_name[name]
        rows.append({
            "dataset": dataset,
            "feature_index": index,
            "feature": name,
            "group": definition["group"],
            "unit": definition["unit"],
            "sample_count": len(values),
            "finite_count": len(finite),
            "minimum": float(np.min(finite)) if len(finite) else "",
            "percentile_01": float(np.quantile(finite, 0.01)) if len(finite) else "",
            "median": float(np.median(finite)) if len(finite) else "",
            "percentile_99": float(np.quantile(finite, 0.99)) if len(finite) else "",
            "maximum": float(np.max(finite)) if len(finite) else "",
            "mean": float(np.mean(finite)) if len(finite) else "",
            "standard_deviation": float(np.std(finite)) if len(finite) else "",
            "constant_feature": bool(len(finite) and np.std(finite) < 1e-12),
        })
    return rows


def zip_validation_report(report_path: Path, zip_path: Path) -> None:
    files = sorted(path for path in report_path.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(report_path)))


def main() -> None:
    print("============================================================")
    print(" Stage 2 V2.0.2 FINAL feature processing started")
    print(" Script build:", SCRIPT_BUILD)
    print("============================================================")

    stage1_run = find_latest_stage1_run()
    print("Stage 1 input:", stage1_run)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_path = OUTPUT_ROOT / run_name
    dataset_path = run_path / "datasets"
    report_path = run_path / "validation_report"
    dataset_path.mkdir(parents=True, exist_ok=False)
    report_path.mkdir(parents=True, exist_ok=False)

    battery_manifest = read_battery_manifest(stage1_run)
    buffers = create_empty_buffers()

    dataset_counters: Dict[str, Counter] = {
        dataset: Counter() for dataset in DATASET_ORDER
    }
    dataset_batteries: Dict[str, set] = {
        dataset: set() for dataset in DATASET_ORDER
    }
    sample_keys: set = set()
    duplicate_sample_count = 0

    sample_manifest_path = report_path / "sample_manifest.csv"
    battery_reference_path = report_path / "battery_reference_manifest.csv"
    quality_issue_path = report_path / "feature_quality_issues.csv"


    with ExitStack() as stack:
        sample_file = stack.enter_context(
            sample_manifest_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
        )
        battery_file = stack.enter_context(
            battery_reference_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
        )
        issue_file = stack.enter_context(
            quality_issue_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
        )
        sample_writer = csv.DictWriter(sample_file, fieldnames=SAMPLE_MANIFEST_FIELDS)
        battery_writer = csv.DictWriter(battery_file, fieldnames=BATTERY_REFERENCE_FIELDS)
        issue_writer = csv.DictWriter(issue_file, fieldnames=QUALITY_ISSUE_FIELDS)
        sample_writer.writeheader()
        battery_writer.writeheader()
        issue_writer.writeheader()

        ordered_manifest = battery_manifest.copy()
        ordered_manifest["dataset"] = ordered_manifest["dataset"].astype(str)
        ordered_manifest["battery_id"] = ordered_manifest["battery_id"].astype(str)
        ordered_manifest["dataset_order"] = ordered_manifest["dataset"].map(
            {name: index for index, name in enumerate(DATASET_ORDER)}
        )
        ordered_manifest = ordered_manifest.sort_values(
            ["dataset_order", "battery_id"],
            kind="stable",
        )

        for manifest_row in ordered_manifest.to_dict(orient="records"):
            dataset = str(manifest_row["dataset"])
            battery_id = str(manifest_row["battery_id"])
            relative_file = normalize_relative_path(manifest_row["processed_file"])
            payload_path = stage1_run / relative_file
            print(f"[{dataset}] {battery_id}")

            if not payload_path.exists():
                issue_writer.writerow({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle": "",
                    "reason": "missing_stage1_battery_file",
                    "detail": str(payload_path),
                })
                dataset_counters[dataset]["battery_file_failures"] += 1
                continue

            try:
                payload = load_pickle(payload_path)
                validate_stage1_payload(payload, manifest_row)
            except Exception as exc:
                issue_writer.writerow({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle": "",
                    "reason": "invalid_stage1_battery_payload",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                dataset_counters[dataset]["battery_file_failures"] += 1
                continue

            dataset_batteries[dataset].add(battery_id)
            dataset_counters[dataset]["battery_count"] += 1
            cycles = list(payload["cycles"])
            cycles.sort(key=lambda item: int(item["cycle"]))
            dataset_counters[dataset]["stage1_valid_cycles"] += len(cycles)

            label_excluded = sum(
                not bool(cycle.get("soh_label_eligible")) for cycle in cycles
            )
            curve_excluded = sum(
                not bool(cycle.get("curve_quality_eligible")) for cycle in cycles
            )
            joint_eligible = [cycle for cycle in cycles if stage1_joint_eligible(cycle)]
            dataset_counters[dataset]["stage1_label_excluded_cycles"] += label_excluded
            dataset_counters[dataset]["stage1_curve_excluded_cycles"] += curve_excluded
            dataset_counters[dataset]["stage1_joint_eligible_cycles"] += len(joint_eligible)

            try:
                reference_cycle = choose_reference_cycle(cycles)
                reference_capacity_ah = float(reference_cycle["capacity_ah"])
                reference_cycle_number = int(reference_cycle["cycle"])
                reference_capacity_source = str(reference_cycle.get("capacity_source", ""))
            except Exception as exc:
                issue_writer.writerow({
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "cycle": "",
                    "reason": "reference_capacity_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                dataset_counters[dataset]["reference_failures"] += 1
                battery_writer.writerow({
                    "dataset": dataset,
                    "chemistry": payload.get("chemistry", ""),
                    "battery_id": battery_id,
                    "condition_group": payload.get("condition_group", ""),
                    "stage1_valid_cycles": len(cycles),
                    "stage1_joint_eligible_cycles": len(joint_eligible),
                    "stage1_label_excluded_cycles": label_excluded,
                    "stage1_curve_excluded_cycles": curve_excluded,
                    "reference_cycle": "",
                    "reference_capacity_ah": "",
                    "reference_capacity_source": "",
                    "extracted_samples": 0,
                    "feature_extraction_failures": len(joint_eligible),
                })
                continue

            battery_vectors: List[np.ndarray] = []
            extracted_samples = 0
            extraction_failures = 0
            output_relative = str(
                (Path("datasets") / f"{dataset}_features.pkl")
            )

            for cycle in joint_eligible:
                cycle_number = int(cycle["cycle"])
                capacity_ah = float(cycle["capacity_ah"])
                soh = capacity_ah / reference_capacity_ah
                if not np.isfinite(soh) or soh <= 0:
                    issue_writer.writerow({
                        "dataset": dataset,
                        "battery_id": battery_id,
                        "cycle": cycle_number,
                        "reason": "invalid_soh_label",
                        "detail": f"capacity={capacity_ah}, Q0={reference_capacity_ah}",
                    })
                    extraction_failures += 1
                    dataset_counters[dataset]["feature_extraction_failures"] += 1
                    continue

                try:
                    vector = extract_cycle_features(cycle, reference_capacity_ah)
                except Exception as exc:
                    issue_writer.writerow({
                        "dataset": dataset,
                        "battery_id": battery_id,
                        "cycle": cycle_number,
                        "reason": "feature_extraction_failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    })
                    extraction_failures += 1
                    dataset_counters[dataset]["feature_extraction_failures"] += 1
                    continue

                sample_key = (dataset, battery_id, cycle_number)
                if sample_key in sample_keys:
                    duplicate_sample_count += 1
                    issue_writer.writerow({
                        "dataset": dataset,
                        "battery_id": battery_id,
                        "cycle": cycle_number,
                        "reason": "duplicate_sample_key",
                        "detail": str(sample_key),
                    })
                    extraction_failures += 1
                    dataset_counters[dataset]["feature_extraction_failures"] += 1
                    continue
                sample_keys.add(sample_key)

                buffer = buffers[dataset]
                row_index = len(buffer["y_soh"])
                battery_vectors.append(vector.reshape(1, -1))
                buffer["y_soh"].append(float(soh))
                buffer["capacity_ah"].append(capacity_ah)
                buffer["reference_capacity_ah"].append(reference_capacity_ah)
                buffer["reference_cycle"].append(reference_cycle_number)
                buffer["battery_id"].append(battery_id)
                buffer["cycle"].append(cycle_number)
                buffer["condition_group"].append(str(cycle.get("condition_group", "")))
                buffer["time_source"].append(str(cycle.get("time_source", "")))
                buffer["ambient_temperature_c"].append(
                    safe_optional_float(cycle.get("ambient_temperature_c"))
                )
                buffer["temperature_available"].append(
                    bool(cycle.get("temperature_available"))
                )
                buffer["chemistry_values"].add(str(cycle.get("chemistry", "")))

                sample_writer.writerow({
                    "dataset": dataset,
                    "chemistry": cycle.get("chemistry", ""),
                    "battery_id": battery_id,
                    "cycle": cycle_number,
                    "condition_group": cycle.get("condition_group", ""),
                    "time_source": cycle.get("time_source", ""),
                    "capacity_ah": capacity_ah,
                    "reference_capacity_ah": reference_capacity_ah,
                    "reference_cycle": reference_cycle_number,
                    "soh": soh,
                    "ambient_temperature_c": (
                        ""
                        if not np.isfinite(
                            safe_optional_float(cycle.get("ambient_temperature_c"))
                        )
                        else float(cycle["ambient_temperature_c"])
                    ),
                    "temperature_available": bool(cycle.get("temperature_available")),
                    "dataset_row_index": row_index,
                    "output_dataset_file": output_relative,
                })

                extracted_samples += 1
                dataset_counters[dataset]["extracted_samples"] += 1
                if soh < SOH_WARNING_LOWER or soh > SOH_WARNING_UPPER:
                    dataset_counters[dataset]["soh_warning_count"] += 1

            if battery_vectors:
                buffers[dataset]["X_chunks"].append(
                    np.concatenate(battery_vectors, axis=0).astype(np.float32, copy=False)
                )

            battery_writer.writerow({
                "dataset": dataset,
                "chemistry": payload.get("chemistry", ""),
                "battery_id": battery_id,
                "condition_group": payload.get("condition_group", ""),
                "stage1_valid_cycles": len(cycles),
                "stage1_joint_eligible_cycles": len(joint_eligible),
                "stage1_label_excluded_cycles": label_excluded,
                "stage1_curve_excluded_cycles": curve_excluded,
                "reference_cycle": reference_cycle_number,
                "reference_capacity_ah": reference_capacity_ah,
                "reference_capacity_source": reference_capacity_source,
                "extracted_samples": extracted_samples,
                "feature_extraction_failures": extraction_failures,
            })

    dataset_payloads: Dict[str, Dict[str, Any]] = {}
    feature_stat_rows: List[Dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        payload = finalize_dataset_payload(dataset, buffers[dataset], stage1_run)
        dataset_payloads[dataset] = payload
        output_file = dataset_path / f"{dataset}_features.pkl"
        write_pickle(payload, output_file)
        feature_stat_rows.extend(feature_statistics_rows(dataset, payload["X"]))
        print(
            f"Saved {dataset}: X={payload['X'].shape}, "
            f"y={payload['y_soh'].shape}, standardized={payload['standardized']}"
        )

    pd.DataFrame(FEATURE_DEFINITIONS).to_csv(
        report_path / "feature_definition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(feature_stat_rows, columns=FEATURE_STATISTIC_FIELDS).to_csv(
        report_path / "feature_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_datasets: Dict[str, Any] = {}
    fatal_checks: List[str] = []
    for dataset in DATASET_ORDER:
        counter = dataset_counters[dataset]
        payload = dataset_payloads[dataset]
        X = payload["X"]
        y = payload["y_soh"]
        extracted = int(len(y))
        expected = int(counter["stage1_joint_eligible_cycles"])
        failures = int(counter["feature_extraction_failures"])
        if extracted + failures != expected:
            fatal_checks.append(
                f"{dataset}: extracted({extracted}) + failures({failures}) "
                f"!= joint eligible({expected})"
            )
        if X.shape != (extracted, 39):
            fatal_checks.append(f"{dataset}: unexpected X shape {X.shape}")
        if not np.all(np.isfinite(X)):
            fatal_checks.append(f"{dataset}: non-finite X")
        if not np.all(np.isfinite(y)):
            fatal_checks.append(f"{dataset}: non-finite y")

        constant_features = [
            FEATURE_NAMES[index]
            for index in range(39)
            if extracted and float(np.std(X[:, index])) < 1e-12
        ]
        summary_datasets[dataset] = {
            "battery_count": int(counter["battery_count"]),
            "stage1_valid_cycle_count": int(counter["stage1_valid_cycles"]),
            "stage1_label_excluded_cycle_count": int(
                counter["stage1_label_excluded_cycles"]
            ),
            "stage1_curve_excluded_cycle_count": int(
                counter["stage1_curve_excluded_cycles"]
            ),
            "stage1_joint_eligible_cycle_count": expected,
            "extracted_sample_count": extracted,
            "feature_extraction_failure_count": failures,
            "reference_failure_count": int(counter["reference_failures"]),
            "battery_file_failure_count": int(counter["battery_file_failures"]),
            "feature_shape": list(X.shape),
            "feature_count": 39,
            "constant_features": constant_features,
            "soh_min": float(np.min(y)) if len(y) else None,
            "soh_max": float(np.max(y)) if len(y) else None,
            "soh_mean": float(np.mean(y)) if len(y) else None,
            "soh_warning_range": [SOH_WARNING_LOWER, SOH_WARNING_UPPER],
            "soh_outside_warning_range_count": int(counter["soh_warning_count"]),
            "standardized": False,
        }

    if duplicate_sample_count:
        fatal_checks.append(f"duplicate sample keys: {duplicate_sample_count}")
    if any(dataset_counters[name]["battery_file_failures"] for name in DATASET_ORDER):
        fatal_checks.append("one or more Stage 1 battery files could not be loaded")
    if any(dataset_counters[name]["reference_failures"] for name in DATASET_ORDER):
        fatal_checks.append("one or more batteries have no valid Q0 reference")

    validation_status = "PASS" if not fatal_checks else "FAIL"
    config = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage1_run": str(stage1_run),
        "output_run": str(run_path),
        "expected_stage1_schema": EXPECTED_STAGE1_SCHEMA,
        "expected_stage1_build": EXPECTED_STAGE1_BUILD,
        "sample_filter": "soh_label_eligible AND curve_quality_eligible",
        "label_definition": "SOH_i = Q_i / Q0",
        "reference_policy": "first_eligible_high_quality_cycle_per_battery",
        "feature_count": 39,
        "normalized_capacity_axis_points": NORMALIZED_CAPACITY_AXIS_POINTS,
        "standardization_performed": False,
        "excluded_from_features": [
            "capacity_ah",
            "integrated_discharge_capacity_ah",
            "raw_discharge_capacity_span_ah",
            "SOH",
            "cycle number",
            "absolute phase duration",
            "temperature",
        ],
        "temperature_policy": (
            "not used in common feature set; missing values are retained as missing metadata"
        ),
        "time_policy": (
            "voltage alignment uses normalized capacity throughput; no absolute duration feature"
        ),
    }
    summary = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "validation_status": validation_status,
        "fatal_checks": fatal_checks,
        "duplicate_sample_key_count": duplicate_sample_count,
        "total_battery_count": int(sum(len(value) for value in dataset_batteries.values())),
        "total_extracted_sample_count": int(
            sum(len(dataset_payloads[name]["y_soh"]) for name in DATASET_ORDER)
        ),
        "feature_count": 39,
        "datasets": summary_datasets,
    }
    write_json(config, report_path / "processing_config.json")
    write_json(summary, report_path / "processing_summary.json")

    readme = (
        "Stage 2 V2.0.2 FINAL validation report\n"
        "=====================================\n\n"
        f"Validation status: {validation_status}\n"
        f"Script build: {SCRIPT_BUILD}\n"
        f"Stage 1 input: {stage1_run}\n"
        f"Stage 2 output: {run_path}\n\n"
        "Please return STAGE2_VALIDATION_REPORT.zip for independent review.\n"
        "Do not use the feature files in model training until the report passes.\n"
        "No feature standardization has been performed in Stage 2.\n"
    )
    with (report_path / "README.txt").open("w", encoding="utf-8") as file:
        file.write(readme)

    zip_path = run_path / "STAGE2_VALIDATION_REPORT.zip"
    zip_validation_report(report_path, zip_path)

    print("============================================================")
    print("Validation status:", validation_status)
    print("Total extracted samples:", summary["total_extracted_sample_count"])
    print("Feature count:", summary["feature_count"])
    print("Validation report:", zip_path)
    print("============================================================")

    if fatal_checks:
        raise RuntimeError(
            "Stage 2 validation failed:\n- " + "\n- ".join(fatal_checks)
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nFATAL ERROR:", type(error).__name__, str(error), file=sys.stderr)
        raise
