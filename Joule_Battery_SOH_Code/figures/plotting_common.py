from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


DEFAULT_BASE_PATH = Path(
    os.environ.get(
        "CRPS_BASE_PATH",
        Path(__file__).resolve().parents[1] / "workspace",
    )
)


COLORS = {
    "XJTU": "#6AB4DC",
    "HUST": "#42B8B1",
    "TJU_NCM": "#E78489",
    "TJU_NCA": "#E1A8BC",
    "CNN_BiLSTM_Attention": "#6AB4DC",
    "Transformer_Attention": "#42B8B1",
    "MLP_Feature_Attention": "#E78489",
    "Average_Ensemble": "#D8CA77",
    "Adaptive_Ensemble": "#D96E7F",
    "selected": "#D96E7F",
    "rejected": "#D9BDBD",
    "PC": "#6AB4DC",
    "TC": "#42B8B1",
    "neutral": "#7A7A7A",
    "light_blue": "#B1DDF4",
    "light_teal": "#9ED6E2",
    "yellow": "#D8CA77",
    "pink": "#E1A8BC",
    "black": "#222222",
}

DATASET_LABELS = {
    "XJTU": "XJTU",
    "HUST": "HUST",
    "TJU_NCM": "TJU-NCM",
    "TJU_NCA": "TJU-NCA",
}

MODEL_LABELS = {
    "CNN_BiLSTM_Attention": "CNN-BiLSTM-attention",
    "Transformer_Attention": "Transformer-attention",
    "MLP_Feature_Attention": "MLP-feature-attention",
    "Average_Ensemble": "Average ensemble",
    "Adaptive_Ensemble": "Adaptive ensemble",
}


def set_publication_style() -> None:

    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "lines.markersize": 4,
            "patch.linewidth": 0.6,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: Optional[str] = None) -> None:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333", pad=2)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color="#D9D9D9",
            linewidth=0.55,
            alpha=0.65,
            linestyle="-",
            zorder=0,
        )
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
        clip_on=False,
    )


def add_zero_line(ax: plt.Axes, orientation: str = "horizontal") -> None:
    if orientation == "horizontal":
        ax.axhline(0, color="#555555", linewidth=0.75, linestyle="--", zorder=1)
    else:
        ax.axvline(0, color="#555555", linewidth=0.75, linestyle="--", zorder=1)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def read_csv_checked(path: Path, columns: Iterable[str] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}")


    frame = pd.read_csv(path, low_memory=False)
    require_columns(frame, columns, path)
    return frame


def read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_accepted_run(
    candidate: Path,
    expected_schema: str,
    required_files: Sequence[str],
) -> Tuple[bool, str]:
    metadata_dir = (
        candidate / "validation_report"
        if (candidate / "validation_report").is_dir()
        else candidate
    )
    summary_path = metadata_dir / "processing_summary.json"
    config_path = metadata_dir / "processing_config.json"
    if not summary_path.is_file() or not config_path.is_file():
        return False, "missing processing metadata"
    try:
        summary = read_json(summary_path)
        config = read_json(config_path)
    except Exception as exc:
        return False, f"invalid metadata: {exc}"

    schema = str(summary.get("schema_version", config.get("schema_version", "")))
    if schema != expected_schema:
        return False, f"schema {schema!r}, expected {expected_schema!r}"
    validation_status = summary.get("validation_status")
    if validation_status is not None and str(validation_status).upper() != "PASS":
        return False, f"validation_status={validation_status!r}"
    if summary.get("fatal_file_errors"):
        return False, "fatal_file_errors is not empty"
    if summary.get("fatal_checks"):
        return False, "fatal_checks is not empty"
    missing = [
        name
        for name in required_files
        if not (candidate / name).exists() and not (metadata_dir / name).exists()
    ]
    if missing:
        return False, f"missing {missing}"
    return True, "accepted"


def report_directory(run: Path) -> Path:

    nested = run / "validation_report"
    return nested if nested.is_dir() else run


def find_latest_accepted_run(
    base_path: Path,
    output_root_name: str,
    expected_schema: str,
    required_files: Sequence[str],
    explicit_run: Optional[Union[str, Path]] = None,
) -> Path:

    if explicit_run:
        candidates = [Path(explicit_run)]
    else:
        output_root = base_path / output_root_name
        candidates: List[Path] = []
        latest_file = output_root / "latest_run_path.txt"
        if latest_file.is_file():
            text = latest_file.read_text(encoding="utf-8").strip()
            if text:
                candidates.append(Path(text))
        if output_root.is_dir():
            candidates.extend(sorted(output_root.glob("run_*"), reverse=True))

    unique_candidates: List[Path] = []
    seen: Set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    diagnostics = []
    for candidate in unique_candidates:
        accepted, reason = _is_accepted_run(
            candidate, expected_schema, required_files
        )
        if accepted:
            print(f"Accepted input run: {candidate}")
            return candidate
        diagnostics.append(f"{candidate}: {reason}")

    details = "\n".join(diagnostics) if diagnostics else "no run directories found"
    raise FileNotFoundError(
        f"No accepted {output_root_name} run was found.\n{details}\n"
        "Run the corresponding validated stage first, or pass --run explicitly."
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        ".pdf": {},
        ".svg": {},
    }
    for suffix, kwargs in outputs.items():
        path = output_dir / f"{stem}{suffix}"
        temporary = output_dir / f".{stem}.writing{suffix}"
        try:
            fig.savefig(
                temporary,
                bbox_inches="tight",
                facecolor="white",
                **kwargs,
            )
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Saved: {path}")


def representative_battery(
    frame: pd.DataFrame,
    prediction_column: Optional[str] = None,
) -> str:

    require_columns(frame, ["battery_id"], Path("in-memory dataframe"))
    if prediction_column and {"y_true", prediction_column}.issubset(frame.columns):
        rows = []
        for battery_id, group in frame.groupby("battery_id", sort=True):
            error = group[prediction_column].to_numpy(float) - group["y_true"].to_numpy(float)
            rows.append((battery_id, float(np.sqrt(np.mean(error**2)))))
        rows.sort(key=lambda item: (item[1], item[0]))
    else:
        rows = [
            (battery_id, float(len(group)))
            for battery_id, group in frame.groupby("battery_id", sort=True)
        ]
        rows.sort(key=lambda item: (item[1], item[0]))
    return str(rows[len(rows) // 2][0])


def feature_label(name: str) -> str:

    exact = {
        "voltage_hysteresis_low_soc_v": "Voltage hysteresis, low SOC",
        "voltage_hysteresis_mid_soc_v": "Voltage hysteresis, mid SOC",
        "voltage_hysteresis_high_soc_v": "Voltage hysteresis, high SOC",
    }
    if name in exact:
        return exact[name]
    text = name

    text = text.replace("discharge_", "Dchg. ")
    text = text.replace("charge_", "Chg. ")
    text = text.replace("voltage_", "voltage ")
    text = text.replace("c_rate_", "C-rate ")
    text = text.replace("gradient_abs_p90", "|gradient| P90")
    text = text.replace("gradient_iqr", "gradient IQR")
    text = text.replace("early_slope", "early slope")
    text = text.replace("mid_slope", "mid slope")
    text = text.replace("late_slope", "late slope")
    text = text.replace("_v_per_q", "")
    text = text.replace("_v", "")
    text = text.replace("_", " ")
    return " ".join(text.split())


def minmax_scale(values: np.ndarray, minimum_size: float, maximum_size: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.full(values.shape, minimum_size, dtype=float)
    if not np.any(finite):
        return result
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if high - low < 1e-12:
        result[finite] = (minimum_size + maximum_size) / 2
    else:
        result[finite] = minimum_size + (values[finite] - low) * (
            maximum_size - minimum_size
        ) / (high - low)
    return result


def grouped_offsets(group_count: int, total_width: float = 0.76) -> Tuple[np.ndarray, float]:
    width = total_width / max(group_count, 1)
    offsets = (np.arange(group_count) - (group_count - 1) / 2) * width
    return offsets, width
