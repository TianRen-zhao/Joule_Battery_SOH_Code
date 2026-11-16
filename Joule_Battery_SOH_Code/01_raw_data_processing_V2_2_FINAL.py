from __future__ import annotations

import csv
import gc
import json
import os
import pickle
import re
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio


BASE_PATH = Path(os.environ.get("CRPS_BASE_PATH", Path(__file__).resolve().parent / "workspace"))

XJTU_PATH = BASE_PATH / "Battery Dataset"
HUST_PATH = BASE_PATH / "our_data"
TJU_NCM_PATH = BASE_PATH / "TJU" / "Dataset_2_NCM_battery"
TJU_NCA_PATH = BASE_PATH / "TJU" / "Dataset_1_NCA_battery"

OUTPUT_ROOT = BASE_PATH / "DATAPROCESSING_V2_2"


RESAMPLE_POINTS = 128
MIN_PHASE_POINTS = 10


CURRENT_ZERO_THRESHOLD_A = 0.01


ACTIVE_TIME_GAP_FACTOR = 10.0
MIN_ACTIVE_TIME_GAP_LIMIT_S = 60.0


REPEATED_SPARSE_MIN_COUNT = 3
REPEATED_SPARSE_RELATIVE_TOLERANCE = 0.10


MIN_VALID_CAPACITY_AH = 0.05
MAX_VALID_CAPACITY_AH = 20.0


XJTU_SUMMARY_RELATIVE_TOLERANCE = 0.35


MAX_CAPACITY_CONSISTENCY_RELATIVE_DIFFERENCE = 0.05


TJU_REFERENCE_WINDOW = 5
TJU_MIN_REFERENCE_CYCLES = 3
TJU_CAPACITY_RATIO_LOWER = 0.80
TJU_CAPACITY_RATIO_UPPER = 1.20

SCHEMA_VERSION = "2.2"
SCRIPT_BUILD = "STAGE1_V2_2_FINAL_20260901"


CHEMISTRY = {
    "XJTU": "NCM",
    "HUST": "UNKNOWN",
    "TJU_NCM": "NCM",
    "TJU_NCA": "NCA",
}


CYCLE_MANIFEST_FIELDS = [
    "dataset",
    "chemistry",
    "battery_id",
    "cycle",
    "source_file",
    "condition_group",
    "description",
    "is_diagnostic_cycle",
    "ambient_temperature_c",
    "time_source",
    "capacity_ah",
    "capacity_source",
    "provided_capacity_ah",
    "raw_discharge_capacity_span_ah",
    "integrated_discharge_capacity_ah",
    "capacity_relative_difference",
    "curve_quality_eligible",
    "curve_quality_exclusion_reason",
    "soh_label_eligible",
    "soh_label_exclusion_reason",
    "capacity_reference_ah",
    "capacity_to_reference_ratio",
    "charge_raw_points",
    "discharge_raw_points",
    "charge_duration_s",
    "discharge_duration_s",
    "charge_filtered_time_gap_count",
    "discharge_filtered_time_gap_count",
    "charge_max_raw_active_gap_s",
    "discharge_max_raw_active_gap_s",
    "charge_ignored_active_time_s",
    "discharge_ignored_active_time_s",
    "charge_preserved_sparse_interval_count",
    "discharge_preserved_sparse_interval_count",
    "charge_preserved_sparse_time_s",
    "discharge_preserved_sparse_time_s",
    "temperature_available",
    "processed_file",
]

BATTERY_MANIFEST_FIELDS = [
    "dataset",
    "chemistry",
    "battery_id",
    "source_file",
    "condition_group",
    "total_cycles_in_source",
    "valid_cycles",
    "skipped_cycles",
    "soh_label_eligible_cycles",
    "soh_label_excluded_cycles",
    "curve_quality_eligible_cycles",
    "curve_quality_excluded_cycles",
    "capacity_min_ah",
    "capacity_max_ah",
    "capacity_mean_ah",
    "soh_eligible_capacity_min_ah",
    "soh_eligible_capacity_max_ah",
    "soh_eligible_capacity_mean_ah",
    "processed_file",
]

QUALITY_FIELDS = [
    "dataset",
    "battery_id",
    "cycle",
    "source_file",
    "reason",
    "detail",
]


def natural_key(value: Any) -> List[Any]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    ]


def sorted_files(folder: Path, suffix: str) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == suffix],
        key=lambda path: natural_key(path.name),
    )


def safe_file_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned.strip("._") or "battery"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return repr(value)


def to_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float64)
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        array = pd.to_numeric(pd.Series(np.asarray(value).reshape(-1)), errors="coerce").to_numpy()
    return np.asarray(array, dtype=np.float64)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return ""
    return " ".join(str(item) for item in array if str(item) != "")


def valid_capacity(value: Any) -> bool:
    try:
        number = float(value)
    except Exception:
        return False
    return bool(
        np.isfinite(number)
        and MIN_VALID_CAPACITY_AH <= number <= MAX_VALID_CAPACITY_AH
    )


def relative_difference(a: Any, b: Any) -> Optional[float]:
    if not valid_capacity(a) or not valid_capacity(b):
        return None
    a_float = float(a)
    b_float = float(b)
    return abs(a_float - b_float) / max(abs(a_float), abs(b_float), 1e-12)


def lookup_mapping(mapping: Any, key: Any) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    candidates = [key, str(key)]
    try:
        candidates.append(int(key))
    except Exception:
        pass
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def find_column(columns: Iterable[Any], exact_names: Sequence[str]) -> Optional[Any]:
    normalized = {str(column).strip().lower(): column for column in columns}
    for name in exact_names:
        if name.strip().lower() in normalized:
            return normalized[name.strip().lower()]
    return None


def write_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def matlab_field_names(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "_fieldnames"):
        return list(getattr(value, "_fieldnames", []) or [])
    if isinstance(value, np.void) and value.dtype.names:
        return list(value.dtype.names)
    if isinstance(value, Mapping):
        return [str(key) for key in value.keys()]
    return []


def matlab_get(value: Any, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if hasattr(value, field):
        return getattr(value, field)
    if isinstance(value, np.void) and value.dtype.names and field in value.dtype.names:
        return value[field]
    if isinstance(value, Mapping):
        return value.get(field, default)
    return default


def scalar_matlab_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


def parse_xjtu_system_time(raw_time: Any, expected_length: int) -> Tuple[np.ndarray, str]:
    values = np.asarray(raw_time).reshape(-1)
    if values.size == expected_length and expected_length > 1:
        text = pd.Series([str(item).strip() for item in values])
        parsed = pd.to_datetime(
            text,
            format="%Y-%m-%d,%H:%M:%S",
            errors="coerce",
        )
        valid_ratio = float(parsed.notna().mean())
        if valid_ratio >= 0.95:
            first_valid = parsed.dropna().iloc[0]
            seconds = (parsed - first_valid).dt.total_seconds()
            seconds = seconds.interpolate(limit_direction="both").to_numpy(dtype=np.float64)
            if np.all(np.isfinite(seconds)) and np.all(np.diff(seconds) >= 0):
                return seconds, "system_time"
    return np.arange(expected_length, dtype=np.float64), "sample_index_fallback_1s"


def extract_xjtu_summary_capacity_fields(
    summary: Any,
    cycle_count: int,
) -> Tuple[List[Tuple[str, np.ndarray]], Dict[str, Any]]:
    summary = scalar_matlab_object(summary)
    schema: Dict[str, Any] = {
        "type": type(summary).__name__,
        "fields": matlab_field_names(summary),
        "numeric_field_shapes": {},
        "capacity_candidates": [],
    }
    candidates: List[Tuple[int, str, np.ndarray]] = []

    for field in schema["fields"]:
        raw = matlab_get(summary, field)
        try:
            array = np.asarray(raw, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        schema["numeric_field_shapes"][field] = list(np.shape(raw))
        if array.size != cycle_count:
            continue

        name = field.lower()
        if not any(token in name for token in ("capacity", "cap", "discharge", "dq")):
            continue

        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue

        converted = array.copy()
        unit = "Ah"
        median_value = float(np.nanmedian(np.abs(finite)))
        if "mah" in name or median_value > 50:
            converted = converted / 1000.0
            unit = "mAh_to_Ah"

        if "discharge" in name and ("capacity" in name or "cap" in name):
            rank = 0
        elif "dq" in name:
            rank = 1
        elif "capacity" in name:
            rank = 2
        else:
            rank = 3

        schema["capacity_candidates"].append({
            "field": field,
            "unit_handling": unit,
            "median_ah": float(np.nanmedian(converted)),
            "min_ah": float(np.nanmin(converted)),
            "max_ah": float(np.nanmax(converted)),
        })
        candidates.append((rank, field, converted))

    candidates.sort(key=lambda item: (item[0], item[1].lower()))
    return [(field, array) for _, field, array in candidates], schema


def align_cycle_arrays(
    time_s: Any,
    voltage_v: Any,
    current_a: Any,
    raw_capacity_ah: Any = None,
    temperature_c: Any = None,
) -> Dict[str, Any]:
    time = to_float_array(time_s)
    voltage = to_float_array(voltage_v)
    current = to_float_array(current_a)

    required_length = min(len(time), len(voltage), len(current))
    if required_length < 2:
        raise ValueError("fewer than two aligned time/voltage/current points")

    time = time[:required_length]
    voltage = voltage[:required_length]
    current = current[:required_length]

    capacity = to_float_array(raw_capacity_ah)
    if len(capacity) < required_length:
        capacity = None
    else:
        capacity = capacity[:required_length]

    temperature = to_float_array(temperature_c)
    if len(temperature) < required_length:
        temperature = None
    else:
        temperature = temperature[:required_length]

    mask = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    if int(mask.sum()) < 2:
        raise ValueError("fewer than two finite aligned points")

    time = time[mask]
    voltage = voltage[mask]
    current = current[mask]
    if capacity is not None:
        capacity = capacity[mask]
    if temperature is not None:
        temperature = temperature[mask]

    order = np.argsort(time, kind="stable")
    time = time[order]
    voltage = voltage[order]
    current = current[order]
    if capacity is not None:
        capacity = capacity[order]
    if temperature is not None:
        temperature = temperature[order]

    time = time - time[0]
    return {
        "time": time,
        "voltage": voltage,
        "current": current,
        "raw_capacity": capacity,
        "temperature": temperature,
    }


def cumulative_active_curves(
    time_s: np.ndarray,
    current_a: np.ndarray,
    active_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    raw_dt = np.diff(time_s)
    active_pairs = active_mask[:-1] & active_mask[1:]
    valid_active_dt = raw_dt[
        active_pairs & np.isfinite(raw_dt) & (raw_dt > 0)
    ]
    if valid_active_dt.size:
        median_active_dt = float(np.median(valid_active_dt))
        gap_limit = max(
            MIN_ACTIVE_TIME_GAP_LIMIT_S,
            ACTIVE_TIME_GAP_FACTOR * median_active_dt,
        )
        max_raw_active_gap = float(np.max(valid_active_dt))
    else:
        median_active_dt = None
        gap_limit = MIN_ACTIVE_TIME_GAP_LIMIT_S
        max_raw_active_gap = None

    candidate_large_dt = valid_active_dt[valid_active_dt > gap_limit]
    sparse_cadence_center: Optional[float] = None
    sparse_cadence_tolerance: Optional[float] = None
    if candidate_large_dt.size >= REPEATED_SPARSE_MIN_COUNT:
        candidate_center = float(np.median(candidate_large_dt))
        candidate_tolerance = max(
            1e-9,
            REPEATED_SPARSE_RELATIVE_TOLERANCE * candidate_center,
        )
        close_to_center = (
            np.abs(candidate_large_dt - candidate_center) <= candidate_tolerance
        )
        if float(np.mean(close_to_center)) >= 0.80:
            sparse_cadence_center = candidate_center
            sparse_cadence_tolerance = candidate_tolerance

    elapsed = np.zeros(len(time_s), dtype=np.float64)
    throughput = np.zeros(len(time_s), dtype=np.float64)
    filtered_gap_count = 0
    ignored_active_time = 0.0
    preserved_sparse_count = 0
    preserved_sparse_time = 0.0

    for index in range(1, len(time_s)):
        elapsed[index] = elapsed[index - 1]
        throughput[index] = throughput[index - 1]
        dt = float(time_s[index] - time_s[index - 1])
        if dt <= 0 or not np.isfinite(dt):
            continue
        if bool(active_mask[index - 1]) and bool(active_mask[index]):
            if dt > gap_limit:
                is_repeated_sparse_interval = (
                    sparse_cadence_center is not None
                    and sparse_cadence_tolerance is not None
                    and abs(dt - sparse_cadence_center) <= sparse_cadence_tolerance
                )
                if is_repeated_sparse_interval:
                    preserved_sparse_count += 1
                    preserved_sparse_time += dt
                else:
                    filtered_gap_count += 1
                    ignored_active_time += dt
                    continue
            elapsed[index] += dt
            mean_current = 0.5 * (
                abs(float(current_a[index - 1])) + abs(float(current_a[index]))
            )
            throughput[index] += mean_current * dt / 3600.0
    gap_info = {
        "median_raw_active_gap_s": median_active_dt,
        "max_raw_active_gap_s": max_raw_active_gap,
        "active_gap_limit_s": float(gap_limit),
        "filtered_time_gap_count": int(filtered_gap_count),
        "ignored_active_time_s": float(ignored_active_time),
        "preserved_sparse_interval_count": int(preserved_sparse_count),
        "preserved_sparse_time_s": float(preserved_sparse_time),
        "preserved_sparse_cadence_s": sparse_cadence_center,
    }
    return elapsed, throughput, gap_info


def interpolate_series(
    x: np.ndarray,
    y: Optional[np.ndarray],
    grid: np.ndarray,
) -> Optional[np.ndarray]:
    if y is None:
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return None
    x_finite = x[finite]
    y_finite = y[finite]
    unique_x, unique_indices = np.unique(x_finite, return_index=True)
    y_unique = y_finite[unique_indices]
    if unique_x.size < 2 or unique_x[-1] <= unique_x[0]:
        return None
    return np.interp(grid, unique_x, y_unique).astype(np.float32)


def build_phase(
    aligned: Dict[str, Any],
    phase: str,
) -> Optional[Dict[str, Any]]:
    time = aligned["time"]
    voltage = aligned["voltage"]
    current = aligned["current"]
    temperature = aligned["temperature"]

    if phase == "charge":
        mask = current > CURRENT_ZERO_THRESHOLD_A
    elif phase == "discharge":
        mask = current < -CURRENT_ZERO_THRESHOLD_A
    else:
        raise ValueError(f"unsupported phase: {phase}")

    raw_points = int(mask.sum())
    if raw_points < MIN_PHASE_POINTS:
        return None

    active_time, throughput, gap_info = cumulative_active_curves(time, current, mask)
    phase_time = active_time[mask]
    phase_voltage = voltage[mask]
    phase_current = current[mask]
    phase_temperature = temperature[mask] if temperature is not None else None
    phase_throughput = throughput[mask]

    duration = float(phase_time[-1] - phase_time[0])
    if not np.isfinite(duration) or duration <= 0:
        return None

    phase_time = phase_time - phase_time[0]
    phase_throughput = phase_throughput - phase_throughput[0]
    grid = np.linspace(0.0, duration, RESAMPLE_POINTS, dtype=np.float64)

    voltage_resampled = interpolate_series(phase_time, phase_voltage, grid)
    current_resampled = interpolate_series(phase_time, phase_current, grid)
    throughput_resampled = interpolate_series(phase_time, phase_throughput, grid)
    temperature_resampled = interpolate_series(phase_time, phase_temperature, grid)

    if voltage_resampled is None or current_resampled is None or throughput_resampled is None:
        return None

    return {
        "time_s": grid.astype(np.float32),
        "voltage_v": voltage_resampled,
        "current_a": current_resampled,
        "capacity_throughput_ah": np.maximum.accumulate(throughput_resampled).astype(np.float32),
        "temperature_c": temperature_resampled,
        "duration_s": duration,
        "raw_point_count": raw_points,
        "integrated_capacity_ah": float(phase_throughput[-1]),
        **gap_info,
    }


def choose_capacity(
    dataset: str,
    provided_capacity_ah: Any,
    provided_capacity_source: str,
    raw_span_ah: Any,
    integrated_capacity_ah: Any,
    provided_authoritative: bool,
) -> Tuple[Optional[float], str]:
    provided = float(provided_capacity_ah) if valid_capacity(provided_capacity_ah) else None
    raw_span = float(raw_span_ah) if valid_capacity(raw_span_ah) else None
    integrated = float(integrated_capacity_ah) if valid_capacity(integrated_capacity_ah) else None

    if provided is not None and provided_authoritative:
        return provided, provided_capacity_source

    if provided is not None:
        comparisons = [
            difference
            for difference in (
                relative_difference(provided, raw_span),
                relative_difference(provided, integrated),
            )
            if difference is not None
        ]
        if comparisons and min(comparisons) <= XJTU_SUMMARY_RELATIVE_TOLERANCE:
            return provided, provided_capacity_source

    if raw_span is not None:
        return raw_span, f"{dataset}:raw_discharge_capacity_span"
    if integrated is not None:
        return integrated, f"{dataset}:integrated_discharge_current"
    return None, "unavailable"


def prepare_cycle_record(
    *,
    dataset: str,
    chemistry: str,
    battery_id: str,
    cycle_number: int,
    source_file: str,
    condition_group: str,
    description: str,
    ambient_temperature_c: Optional[float],
    time_source: str,
    time_s: Any,
    voltage_v: Any,
    current_a: Any,
    raw_capacity_ah: Any,
    temperature_c: Any,
    provided_capacity_ah: Any,
    provided_capacity_source: str,
    provided_authoritative: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        aligned = align_cycle_arrays(
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            raw_capacity_ah=raw_capacity_ah,
            temperature_c=temperature_c,
        )
    except Exception as exc:
        return None, f"alignment_failed: {exc}"

    charge = build_phase(aligned, "charge")
    discharge = build_phase(aligned, "discharge")
    if charge is None:
        return None, "insufficient_valid_charge_phase"
    if discharge is None:
        return None, "insufficient_valid_discharge_phase"

    raw_span: Optional[float] = None
    raw_capacity = aligned["raw_capacity"]
    discharge_mask = aligned["current"] < -CURRENT_ZERO_THRESHOLD_A
    if raw_capacity is not None:
        values = raw_capacity[discharge_mask]
        values = values[np.isfinite(values)]
        if values.size >= MIN_PHASE_POINTS:
            candidate = float(np.max(values) - np.min(values))
            if valid_capacity(candidate):
                raw_span = candidate

    integrated_capacity = discharge["integrated_capacity_ah"]
    capacity, capacity_source = choose_capacity(
        dataset=dataset,
        provided_capacity_ah=provided_capacity_ah,
        provided_capacity_source=provided_capacity_source,
        raw_span_ah=raw_span,
        integrated_capacity_ah=integrated_capacity,
        provided_authoritative=provided_authoritative,
    )
    if capacity is None:
        return None, "no_valid_discharge_capacity"


    if valid_capacity(integrated_capacity):
        consistency_reference = float(integrated_capacity)
    elif valid_capacity(raw_span):
        consistency_reference = float(raw_span)
    else:
        consistency_reference = None
    consistency_difference = relative_difference(capacity, consistency_reference)
    curve_quality_eligible = (
        consistency_difference is None
        or consistency_difference <= MAX_CAPACITY_CONSISTENCY_RELATIVE_DIFFERENCE
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "chemistry": chemistry,
        "battery_id": battery_id,
        "cycle": int(cycle_number),
        "source_file": source_file,
        "condition_group": condition_group,
        "description": description,
        "is_diagnostic_cycle": "test capacity" in description.lower(),
        "ambient_temperature_c": (
            float(ambient_temperature_c)
            if ambient_temperature_c is not None and np.isfinite(ambient_temperature_c)
            else None
        ),
        "time_source": time_source,
        "temperature_available": bool(aligned["temperature"] is not None),
        "raw_cycle_point_count": int(len(aligned["time"])),
        "charge": charge,
        "discharge": discharge,
        "capacity_ah": float(capacity),
        "capacity": float(capacity),
        "capacity_source": capacity_source,
        "provided_capacity_ah": (
            float(provided_capacity_ah) if valid_capacity(provided_capacity_ah) else None
        ),
        "raw_discharge_capacity_span_ah": raw_span,
        "integrated_discharge_capacity_ah": float(integrated_capacity),
        "capacity_relative_difference": consistency_difference,
        "curve_quality_eligible": bool(curve_quality_eligible),
        "curve_quality_exclusion_reason": (
            ""
            if curve_quality_eligible
            else "capacity_integration_disagreement_gt_5pct"
        ),
        "soh_label_eligible": True,
        "soh_label_exclusion_reason": "",
        "capacity_reference_ah": None,
        "capacity_to_reference_ratio": None,
    }
    return record, None


def mark_xjtu_soh_label_quality(records: List[Dict[str, Any]]) -> str:
    diagnostic_count = sum(bool(item["is_diagnostic_cycle"]) for item in records)
    non_diagnostic_count = len(records) - diagnostic_count

    if diagnostic_count >= 2:
        policy = "repeated_xjtu_diagnostic_capacity_protocol"
        for item in records:
            eligible = bool(item["is_diagnostic_cycle"])
            item["soh_label_eligible"] = eligible
            item["soh_label_exclusion_reason"] = (
                "" if eligible else "xjtu_non_reference_ageing_protocol"
            )
    elif non_diagnostic_count > 0:
        policy = "repeated_xjtu_ageing_protocol"
        for item in records:
            eligible = not bool(item["is_diagnostic_cycle"])
            item["soh_label_eligible"] = eligible
            item["soh_label_exclusion_reason"] = (
                "" if eligible else "xjtu_single_initial_diagnostic_protocol"
            )
    else:
        policy = "only_available_xjtu_protocol"
        for item in records:
            item["soh_label_eligible"] = True
            item["soh_label_exclusion_reason"] = ""
    return policy


def mark_tju_soh_label_quality(records: List[Dict[str, Any]]) -> str:
    accepted_capacities: List[float] = []
    for item in sorted(records, key=lambda value: int(value["cycle"])):
        capacity = float(item["capacity_ah"])
        if len(accepted_capacities) >= TJU_MIN_REFERENCE_CYCLES:
            reference = float(
                np.median(accepted_capacities[-TJU_REFERENCE_WINDOW:])
            )
            ratio = capacity / max(reference, 1e-12)
            item["capacity_reference_ah"] = reference
            item["capacity_to_reference_ratio"] = ratio
            eligible = (
                TJU_CAPACITY_RATIO_LOWER <= ratio <= TJU_CAPACITY_RATIO_UPPER
            )
        else:
            eligible = True

        item["soh_label_eligible"] = bool(eligible)
        item["soh_label_exclusion_reason"] = (
            "" if eligible else "abrupt_capacity_deviation_from_trailing_median"
        )
        if eligible:
            accepted_capacities.append(capacity)

    return (
        f"trailing_median_window_{TJU_REFERENCE_WINDOW}_"
        f"ratio_{TJU_CAPACITY_RATIO_LOWER:.2f}_{TJU_CAPACITY_RATIO_UPPER:.2f}"
    )


def process_xjtu_file(path: Path, batch_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset_name = "XJTU"
    battery_id = f"XJTU__{batch_name}__{path.stem}"
    quality: List[Dict[str, Any]] = []

    mat = sio.loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["data", "summary"],
    )
    if "data" not in mat:
        raise KeyError("MAT file has no 'data' variable")

    cycles = np.asarray(mat["data"], dtype=object).reshape(-1)
    summary_candidates, summary_schema = extract_xjtu_summary_capacity_fields(
        mat.get("summary"),
        len(cycles),
    )

    valid_records: List[Dict[str, Any]] = []
    for index, cycle in enumerate(cycles):
        cycle_number = index + 1
        try:
            voltage = matlab_get(cycle, "voltage_V")
            current = matlab_get(cycle, "current_A")
            raw_capacity = matlab_get(cycle, "capacity_Ah")
            temperature = matlab_get(cycle, "temperature_C")
            description = to_text(matlab_get(cycle, "description"))

            if voltage is None or current is None:
                raise KeyError(
                    f"missing named XJTU fields; available={matlab_field_names(cycle)}"
                )

            voltage_array = to_float_array(voltage)
            time, time_source = parse_xjtu_system_time(
                matlab_get(cycle, "system_time"),
                len(voltage_array),
            )

            provided_capacity = None
            provided_source = "unavailable"
            for field, values in summary_candidates:
                if index < len(values) and valid_capacity(values[index]):
                    provided_capacity = float(values[index])
                    provided_source = f"XJTU:summary.{field}"
                    break

            record, reason = prepare_cycle_record(
                dataset=dataset_name,
                chemistry=CHEMISTRY[dataset_name],
                battery_id=battery_id,
                cycle_number=cycle_number,
                source_file=str(path),
                condition_group=batch_name,
                description=description,
                ambient_temperature_c=None,
                time_source=time_source,
                time_s=time,
                voltage_v=voltage_array,
                current_a=current,
                raw_capacity_ah=raw_capacity,
                temperature_c=temperature,
                provided_capacity_ah=provided_capacity,
                provided_capacity_source=provided_source,
                provided_authoritative=False,
            )
            if record is None:
                quality.append({
                    "dataset": dataset_name,
                    "battery_id": battery_id,
                    "cycle": cycle_number,
                    "source_file": str(path),
                    "reason": reason,
                    "detail": description,
                })
            else:
                valid_records.append(record)
        except Exception as exc:
            quality.append({
                "dataset": dataset_name,
                "battery_id": battery_id,
                "cycle": cycle_number,
                "source_file": str(path),
                "reason": "cycle_exception",
                "detail": f"{type(exc).__name__}: {exc}",
            })

    soh_label_policy = mark_xjtu_soh_label_quality(valid_records)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "chemistry": CHEMISTRY[dataset_name],
        "battery_id": battery_id,
        "source_file": str(path),
        "condition_group": batch_name,
        "metadata": {
            "batch": batch_name,
            "protocol_name": path.stem.split("_battery")[0],
            "xjtu_summary_schema": summary_schema,
            "soh_label_policy": soh_label_policy,
        },
        "source_cycle_count": int(len(cycles)),
        "cycles": valid_records,
    }
    del mat, cycles
    gc.collect()
    return payload, quality


def process_hust_file(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset_name = "HUST"
    battery_id = f"HUST__{path.stem}"
    quality: List[Dict[str, Any]] = []

    with path.open("rb") as file:
        raw = pickle.load(file)
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("HUST top-level object must be a non-empty mapping")

    matching_cell = raw.get(path.stem)
    if matching_cell is None:
        matching_cell = raw[next(iter(raw.keys()))]
    if not isinstance(matching_cell, Mapping):
        raise TypeError("HUST cell object must be a mapping")

    cycle_data = matching_cell.get("data")
    dq_data = matching_cell.get("dq")
    if not isinstance(cycle_data, Mapping):
        raise TypeError("HUST cell['data'] must be a cycle mapping")

    ordered_cycle_keys = sorted(cycle_data.keys(), key=natural_key)
    valid_records: List[Dict[str, Any]] = []
    observed_statuses: Counter = Counter()

    for cycle_key in ordered_cycle_keys:
        try:
            cycle_number = int(cycle_key)
        except Exception:
            cycle_number = len(valid_records) + len(quality) + 1

        try:
            frame = cycle_data[cycle_key]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("cycle object is not a pandas DataFrame")

            voltage_col = find_column(frame.columns, ["Voltage (V)"])
            current_col = find_column(frame.columns, ["Current (mA)"])
            capacity_col = find_column(frame.columns, ["Capacity (mAh)"])
            time_col = find_column(frame.columns, ["Time (s)"])
            status_col = find_column(frame.columns, ["Status"])

            missing = [
                name
                for name, column in {
                    "Voltage (V)": voltage_col,
                    "Current (mA)": current_col,
                    "Capacity (mAh)": capacity_col,
                    "Time (s)": time_col,
                }.items()
                if column is None
            ]
            if missing:
                raise KeyError(f"missing HUST columns: {missing}")

            statuses: List[str] = []
            if status_col is not None:
                statuses = [str(item) for item in frame[status_col].dropna().unique()]
                observed_statuses.update(statuses)

            dq_value = lookup_mapping(dq_data, cycle_key)
            dq_ah = float(dq_value) / 1000.0 if dq_value is not None else None

            record, reason = prepare_cycle_record(
                dataset=dataset_name,
                chemistry=CHEMISTRY[dataset_name],
                battery_id=battery_id,
                cycle_number=cycle_number,
                source_file=str(path),
                condition_group=path.stem.split("-")[0],
                description=" | ".join(statuses),
                ambient_temperature_c=None,
                time_source="HUST:Time (s)",
                time_s=frame[time_col].to_numpy(),
                voltage_v=frame[voltage_col].to_numpy(),
                current_a=to_float_array(frame[current_col].to_numpy()) / 1000.0,
                raw_capacity_ah=to_float_array(frame[capacity_col].to_numpy()) / 1000.0,
                temperature_c=None,
                provided_capacity_ah=dq_ah,
                provided_capacity_source="HUST:dq",
                provided_authoritative=True,
            )
            if record is None:
                quality.append({
                    "dataset": dataset_name,
                    "battery_id": battery_id,
                    "cycle": cycle_number,
                    "source_file": str(path),
                    "reason": reason,
                    "detail": " | ".join(statuses),
                })
            else:
                valid_records.append(record)
        except Exception as exc:
            quality.append({
                "dataset": dataset_name,
                "battery_id": battery_id,
                "cycle": cycle_number,
                "source_file": str(path),
                "reason": "cycle_exception",
                "detail": f"{type(exc).__name__}: {exc}",
            })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "chemistry": CHEMISTRY[dataset_name],
        "battery_id": battery_id,
        "source_file": str(path),
        "condition_group": path.stem.split("-")[0],
        "metadata": {
            "hust_cell_key": str(next(iter(raw.keys()))),
            "observed_statuses": sorted(observed_statuses.keys()),
            "has_rul": isinstance(matching_cell.get("rul"), Mapping),
            "has_dq": isinstance(dq_data, Mapping),
            "soh_label_policy": "all_authoritative_hust_dq_cycles",
        },
        "source_cycle_count": int(len(ordered_cycle_keys)),
        "cycles": valid_records,
    }
    del raw, cycle_data
    gc.collect()
    return payload, quality


def parse_tju_condition(file_stem: str) -> Dict[str, Any]:
    match = re.match(r"CY(?P<temperature>\d+)-(?P<rate>\d+)_", file_stem, flags=re.I)
    if not match:
        return {
            "condition_group": "UNKNOWN",
            "temperature_c": None,
            "nominal_rate_c": None,
        }
    temperature = int(match.group("temperature"))
    rate_token = match.group("rate")
    rate_map = {"025": 0.25, "05": 0.5, "1": 1.0}
    rate = rate_map.get(rate_token)
    if rate is None:
        try:
            rate = float(rate_token)
        except Exception:
            rate = None
    return {
        "condition_group": f"{temperature}C_{rate_token}C",
        "temperature_c": temperature,
        "nominal_rate_c": rate,
    }


def process_tju_file(
    path: Path,
    dataset_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    battery_id = f"{dataset_name}__{path.stem}"
    quality: List[Dict[str, Any]] = []
    condition = parse_tju_condition(path.stem)

    frame = pd.read_csv(path, low_memory=False)
    required_columns = [
        "time/s",
        "Ecell/V",
        "<I>/mA",
        "Q discharge/mA.h",
        "cycle number",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing TJU columns: {missing}")

    numeric_cycle = pd.to_numeric(frame["cycle number"], errors="coerce")
    cycle_values = sorted(numeric_cycle.dropna().unique(), key=float)
    valid_records: List[Dict[str, Any]] = []

    for cycle_value in cycle_values:
        cycle_number = int(round(float(cycle_value)))
        try:
            cycle_frame = frame[numeric_cycle == cycle_value]
            if cycle_frame.empty:
                raise ValueError("empty cycle DataFrame")

            q_discharge_mAh = pd.to_numeric(
                cycle_frame["Q discharge/mA.h"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            finite_q = q_discharge_mAh[np.isfinite(q_discharge_mAh)]
            provided_capacity = (
                float(np.max(finite_q)) / 1000.0 if finite_q.size else None
            )

            record, reason = prepare_cycle_record(
                dataset=dataset_name,
                chemistry=CHEMISTRY[dataset_name],
                battery_id=battery_id,
                cycle_number=cycle_number,
                source_file=str(path),
                condition_group=condition["condition_group"],
                description=condition["condition_group"],
                ambient_temperature_c=condition["temperature_c"],
                time_source="TJU:time/s",
                time_s=cycle_frame["time/s"].to_numpy(),
                voltage_v=cycle_frame["Ecell/V"].to_numpy(),
                current_a=to_float_array(cycle_frame["<I>/mA"].to_numpy()) / 1000.0,
                raw_capacity_ah=q_discharge_mAh / 1000.0,
                temperature_c=None,
                provided_capacity_ah=provided_capacity,
                provided_capacity_source=f"{dataset_name}:max_Q_discharge",
                provided_authoritative=True,
            )
            if record is None:
                quality.append({
                    "dataset": dataset_name,
                    "battery_id": battery_id,
                    "cycle": cycle_number,
                    "source_file": str(path),
                    "reason": reason,
                    "detail": condition["condition_group"],
                })
            else:
                valid_records.append(record)
        except Exception as exc:
            quality.append({
                "dataset": dataset_name,
                "battery_id": battery_id,
                "cycle": cycle_number,
                "source_file": str(path),
                "reason": "cycle_exception",
                "detail": f"{type(exc).__name__}: {exc}",
            })

    soh_label_policy = mark_tju_soh_label_quality(valid_records)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "chemistry": CHEMISTRY[dataset_name],
        "battery_id": battery_id,
        "source_file": str(path),
        "condition_group": condition["condition_group"],
        "metadata": {
            **condition,
            "soh_label_policy": soh_label_policy,
        },
        "source_cycle_count": int(len(cycle_values)),
        "cycles": valid_records,
    }
    del frame
    gc.collect()
    return payload, quality


class DatasetStatistics:
    def __init__(self) -> None:
        self.battery_count = 0
        self.source_cycle_count = 0
        self.valid_cycle_count = 0
        self.skipped_cycle_count = 0
        self.soh_label_eligible_cycle_count = 0
        self.soh_label_excluded_cycle_count = 0
        self.curve_quality_eligible_cycle_count = 0
        self.curve_quality_excluded_cycle_count = 0
        self.filtered_time_gap_cycle_count = 0
        self.filtered_time_gap_count = 0
        self.preserved_sparse_interval_cycle_count = 0
        self.preserved_sparse_interval_count = 0
        self.capacity_sum = 0.0
        self.capacity_min = np.inf
        self.capacity_max = -np.inf
        self.eligible_capacity_sum = 0.0
        self.eligible_capacity_min = np.inf
        self.eligible_capacity_max = -np.inf
        self.capacity_sources: Counter = Counter()

    def update(self, payload: Dict[str, Any], skipped: int) -> None:
        self.battery_count += 1
        self.source_cycle_count += int(payload["source_cycle_count"])
        self.skipped_cycle_count += int(skipped)
        for cycle in payload["cycles"]:
            capacity = float(cycle["capacity_ah"])
            self.valid_cycle_count += 1
            if bool(cycle["soh_label_eligible"]):
                self.soh_label_eligible_cycle_count += 1
                self.eligible_capacity_sum += capacity
                self.eligible_capacity_min = min(self.eligible_capacity_min, capacity)
                self.eligible_capacity_max = max(self.eligible_capacity_max, capacity)
            else:
                self.soh_label_excluded_cycle_count += 1
            if bool(cycle["curve_quality_eligible"]):
                self.curve_quality_eligible_cycle_count += 1
            else:
                self.curve_quality_excluded_cycle_count += 1
            gap_count = int(cycle["charge"]["filtered_time_gap_count"]) + int(
                cycle["discharge"]["filtered_time_gap_count"]
            )
            if gap_count > 0:
                self.filtered_time_gap_cycle_count += 1
                self.filtered_time_gap_count += gap_count
            sparse_count = int(
                cycle["charge"]["preserved_sparse_interval_count"]
            ) + int(cycle["discharge"]["preserved_sparse_interval_count"])
            if sparse_count > 0:
                self.preserved_sparse_interval_cycle_count += 1
                self.preserved_sparse_interval_count += sparse_count
            self.capacity_sum += capacity
            self.capacity_min = min(self.capacity_min, capacity)
            self.capacity_max = max(self.capacity_max, capacity)
            self.capacity_sources[cycle["capacity_source"]] += 1

    def as_dict(self) -> Dict[str, Any]:
        mean = (
            self.capacity_sum / self.valid_cycle_count
            if self.valid_cycle_count
            else None
        )
        eligible_mean = (
            self.eligible_capacity_sum / self.soh_label_eligible_cycle_count
            if self.soh_label_eligible_cycle_count
            else None
        )
        return {
            "battery_count": self.battery_count,
            "source_cycle_count": self.source_cycle_count,
            "valid_cycle_count": self.valid_cycle_count,
            "skipped_cycle_count": self.skipped_cycle_count,
            "soh_label_eligible_cycle_count": self.soh_label_eligible_cycle_count,
            "soh_label_excluded_cycle_count": self.soh_label_excluded_cycle_count,
            "curve_quality_eligible_cycle_count": (
                self.curve_quality_eligible_cycle_count
            ),
            "curve_quality_excluded_cycle_count": (
                self.curve_quality_excluded_cycle_count
            ),
            "filtered_time_gap_cycle_count": self.filtered_time_gap_cycle_count,
            "filtered_time_gap_count": self.filtered_time_gap_count,
            "preserved_sparse_interval_cycle_count": (
                self.preserved_sparse_interval_cycle_count
            ),
            "preserved_sparse_interval_count": self.preserved_sparse_interval_count,
            "capacity_min_ah": self.capacity_min if self.valid_cycle_count else None,
            "capacity_max_ah": self.capacity_max if self.valid_cycle_count else None,
            "capacity_mean_ah": mean,
            "soh_eligible_capacity_min_ah": (
                self.eligible_capacity_min
                if self.soh_label_eligible_cycle_count
                else None
            ),
            "soh_eligible_capacity_max_ah": (
                self.eligible_capacity_max
                if self.soh_label_eligible_cycle_count
                else None
            ),
            "soh_eligible_capacity_mean_ah": eligible_mean,
            "capacity_sources": dict(self.capacity_sources),
        }


def cycle_manifest_row(cycle: Dict[str, Any], processed_file: str) -> Dict[str, Any]:
    charge = cycle["charge"]
    discharge = cycle["discharge"]
    return {
        "dataset": cycle["dataset"],
        "chemistry": cycle["chemistry"],
        "battery_id": cycle["battery_id"],
        "cycle": cycle["cycle"],
        "source_file": cycle["source_file"],
        "condition_group": cycle["condition_group"],
        "description": cycle["description"],
        "is_diagnostic_cycle": cycle["is_diagnostic_cycle"],
        "ambient_temperature_c": cycle["ambient_temperature_c"],
        "time_source": cycle["time_source"],
        "capacity_ah": cycle["capacity_ah"],
        "capacity_source": cycle["capacity_source"],
        "provided_capacity_ah": cycle["provided_capacity_ah"],
        "raw_discharge_capacity_span_ah": cycle["raw_discharge_capacity_span_ah"],
        "integrated_discharge_capacity_ah": cycle["integrated_discharge_capacity_ah"],
        "capacity_relative_difference": cycle["capacity_relative_difference"],
        "curve_quality_eligible": cycle["curve_quality_eligible"],
        "curve_quality_exclusion_reason": cycle["curve_quality_exclusion_reason"],
        "soh_label_eligible": cycle["soh_label_eligible"],
        "soh_label_exclusion_reason": cycle["soh_label_exclusion_reason"],
        "capacity_reference_ah": cycle["capacity_reference_ah"],
        "capacity_to_reference_ratio": cycle["capacity_to_reference_ratio"],
        "charge_raw_points": charge["raw_point_count"],
        "discharge_raw_points": discharge["raw_point_count"],
        "charge_duration_s": charge["duration_s"],
        "discharge_duration_s": discharge["duration_s"],
        "charge_filtered_time_gap_count": charge["filtered_time_gap_count"],
        "discharge_filtered_time_gap_count": discharge["filtered_time_gap_count"],
        "charge_max_raw_active_gap_s": charge["max_raw_active_gap_s"],
        "discharge_max_raw_active_gap_s": discharge["max_raw_active_gap_s"],
        "charge_ignored_active_time_s": charge["ignored_active_time_s"],
        "discharge_ignored_active_time_s": discharge["ignored_active_time_s"],
        "charge_preserved_sparse_interval_count": (
            charge["preserved_sparse_interval_count"]
        ),
        "discharge_preserved_sparse_interval_count": (
            discharge["preserved_sparse_interval_count"]
        ),
        "charge_preserved_sparse_time_s": charge["preserved_sparse_time_s"],
        "discharge_preserved_sparse_time_s": discharge["preserved_sparse_time_s"],
        "temperature_available": cycle["temperature_available"],
        "processed_file": processed_file,
    }


def battery_manifest_row(payload: Dict[str, Any], processed_file: str, skipped: int) -> Dict[str, Any]:
    capacities = [float(cycle["capacity_ah"]) for cycle in payload["cycles"]]
    eligible_capacities = [
        float(cycle["capacity_ah"])
        for cycle in payload["cycles"]
        if bool(cycle["soh_label_eligible"])
    ]
    eligible_count = len(eligible_capacities)
    curve_quality_eligible_count = sum(
        bool(cycle["curve_quality_eligible"]) for cycle in payload["cycles"]
    )
    return {
        "dataset": payload["dataset"],
        "chemistry": payload["chemistry"],
        "battery_id": payload["battery_id"],
        "source_file": payload["source_file"],
        "condition_group": payload["condition_group"],
        "total_cycles_in_source": payload["source_cycle_count"],
        "valid_cycles": len(payload["cycles"]),
        "skipped_cycles": skipped,
        "soh_label_eligible_cycles": eligible_count,
        "soh_label_excluded_cycles": len(payload["cycles"]) - eligible_count,
        "curve_quality_eligible_cycles": curve_quality_eligible_count,
        "curve_quality_excluded_cycles": (
            len(payload["cycles"]) - curve_quality_eligible_count
        ),
        "capacity_min_ah": min(capacities) if capacities else "",
        "capacity_max_ah": max(capacities) if capacities else "",
        "capacity_mean_ah": float(np.mean(capacities)) if capacities else "",
        "soh_eligible_capacity_min_ah": (
            min(eligible_capacities) if eligible_capacities else ""
        ),
        "soh_eligible_capacity_max_ah": (
            max(eligible_capacities) if eligible_capacities else ""
        ),
        "soh_eligible_capacity_mean_ah": (
            float(np.mean(eligible_capacities)) if eligible_capacities else ""
        ),
        "processed_file": processed_file,
    }


def process_and_save_battery(
    payload: Dict[str, Any],
    quality_rows: List[Dict[str, Any]],
    run_path: Path,
    cycle_writer: csv.DictWriter,
    battery_writer: csv.DictWriter,
    quality_writer: csv.DictWriter,
    statistics: Dict[str, DatasetStatistics],
) -> None:
    dataset_name = payload["dataset"]
    output_folder = run_path / "processed_batteries" / dataset_name
    output_file = output_folder / f"{safe_file_name(payload['battery_id'])}.pkl"
    relative_output = str(output_file.relative_to(run_path))

    write_pickle(payload, output_file)
    for cycle in payload["cycles"]:
        cycle_writer.writerow(cycle_manifest_row(cycle, relative_output))
    for row in quality_rows:
        quality_writer.writerow(row)
    skipped_cycles = max(
        0,
        int(payload["source_cycle_count"]) - len(payload["cycles"]),
    )
    battery_writer.writerow(
        battery_manifest_row(payload, relative_output, skipped_cycles)
    )
    statistics[dataset_name].update(payload, skipped_cycles)


def main() -> None:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_path = OUTPUT_ROOT / run_name
    manifest_path = run_path / "manifests"
    manifest_path.mkdir(parents=True, exist_ok=False)

    config = {
        "schema_version": SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_path": str(BASE_PATH),
        "paths": {
            "XJTU": str(XJTU_PATH),
            "HUST": str(HUST_PATH),
            "TJU_NCM": str(TJU_NCM_PATH),
            "TJU_NCA": str(TJU_NCA_PATH),
        },
        "resample_points": RESAMPLE_POINTS,
        "min_phase_points": MIN_PHASE_POINTS,
        "current_zero_threshold_a": CURRENT_ZERO_THRESHOLD_A,
        "active_time_gap_factor": ACTIVE_TIME_GAP_FACTOR,
        "min_active_time_gap_limit_s": MIN_ACTIVE_TIME_GAP_LIMIT_S,
        "repeated_sparse_min_count": REPEATED_SPARSE_MIN_COUNT,
        "repeated_sparse_relative_tolerance": (
            REPEATED_SPARSE_RELATIVE_TOLERANCE
        ),
        "valid_capacity_range_ah": [MIN_VALID_CAPACITY_AH, MAX_VALID_CAPACITY_AH],
        "max_capacity_consistency_relative_difference": (
            MAX_CAPACITY_CONSISTENCY_RELATIVE_DIFFERENCE
        ),
        "tju_soh_label_quality": {
            "reference_window": TJU_REFERENCE_WINDOW,
            "minimum_reference_cycles": TJU_MIN_REFERENCE_CYCLES,
            "capacity_ratio_range": [
                TJU_CAPACITY_RATIO_LOWER,
                TJU_CAPACITY_RATIO_UPPER,
            ],
            "uses_future_cycles": False,
        },
        "chemistry": CHEMISTRY,
    }
    (run_path / "processing_config.json").write_text(
        json.dumps(json_safe(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    statistics: Dict[str, DatasetStatistics] = defaultdict(DatasetStatistics)
    fatal_errors: List[Dict[str, Any]] = []

    cycle_manifest_file = (manifest_path / "cycle_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    )
    battery_manifest_file = (manifest_path / "battery_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    )
    quality_file = (manifest_path / "cycle_quality_issues.csv").open(
        "w", newline="", encoding="utf-8-sig"
    )

    cycle_writer = csv.DictWriter(cycle_manifest_file, fieldnames=CYCLE_MANIFEST_FIELDS)
    battery_writer = csv.DictWriter(battery_manifest_file, fieldnames=BATTERY_MANIFEST_FIELDS)
    quality_writer = csv.DictWriter(quality_file, fieldnames=QUALITY_FIELDS)
    cycle_writer.writeheader()
    battery_writer.writeheader()
    quality_writer.writeheader()

    try:
        print("\n========== Stage 1 V2.2 FINAL preprocessing started ==========")
        print("Script build:", SCRIPT_BUILD)
        print("Run directory:", run_path)


        for batch_number in range(1, 7):
            batch_name = f"Batch-{batch_number}"
            batch_path = XJTU_PATH / batch_name
            if not batch_path.exists():
                fatal_errors.append({
                    "dataset": "XJTU",
                    "source_file": str(batch_path),
                    "error": "batch_directory_missing",
                })
                continue
            for path in sorted_files(batch_path, ".mat"):
                print("[XJTU]", path.name)
                try:
                    payload, issues = process_xjtu_file(path, batch_name)
                    process_and_save_battery(
                        payload, issues, run_path,
                        cycle_writer, battery_writer, quality_writer, statistics,
                    )
                except Exception as exc:
                    fatal_errors.append({
                        "dataset": "XJTU",
                        "source_file": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-8:],
                    })
                gc.collect()


        for path in sorted_files(HUST_PATH, ".pkl"):
            print("[HUST]", path.name)
            try:
                payload, issues = process_hust_file(path)
                process_and_save_battery(
                    payload, issues, run_path,
                    cycle_writer, battery_writer, quality_writer, statistics,
                )
            except Exception as exc:
                fatal_errors.append({
                    "dataset": "HUST",
                    "source_file": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-8:],
                })
            gc.collect()


        for dataset_name, folder in (
            ("TJU_NCM", TJU_NCM_PATH),
            ("TJU_NCA", TJU_NCA_PATH),
        ):
            for path in sorted_files(folder, ".csv"):
                print(f"[{dataset_name}]", path.name)
                try:
                    payload, issues = process_tju_file(path, dataset_name)
                    process_and_save_battery(
                        payload, issues, run_path,
                        cycle_writer, battery_writer, quality_writer, statistics,
                    )
                except Exception as exc:
                    fatal_errors.append({
                        "dataset": dataset_name,
                        "source_file": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc().splitlines()[-8:],
                    })
                gc.collect()
    finally:
        cycle_manifest_file.close()
        battery_manifest_file.close()
        quality_file.close()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "script_build": SCRIPT_BUILD,
        "run_path": str(run_path),
        "datasets": {
            name: statistics[name].as_dict()
            for name in ("XJTU", "HUST", "TJU_NCM", "TJU_NCA")
        },
        "fatal_file_errors": fatal_errors,
    }
    (run_path / "processing_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "latest_run_path.txt").write_text(
        str(run_path),
        encoding="utf-8",
    )

    validation_zip = run_path / "STAGE1_VALIDATION_REPORT.zip"
    validation_files = [
        run_path / "processing_config.json",
        run_path / "processing_summary.json",
        manifest_path / "battery_manifest.csv",
        manifest_path / "cycle_manifest.csv",
        manifest_path / "cycle_quality_issues.csv",
    ]
    with zipfile.ZipFile(
        validation_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for report_file in validation_files:
            archive.write(report_file, arcname=str(report_file.relative_to(run_path)))

    print("\n========== Stage 1 V2.2 FINAL preprocessing finished ==========")
    print(json.dumps(json_safe(summary["datasets"]), ensure_ascii=False, indent=2))
    print("Fatal file errors:", len(fatal_errors))
    print("Please return this validation ZIP:")
    print("  ", validation_zip)


if __name__ == "__main__":
    main()
