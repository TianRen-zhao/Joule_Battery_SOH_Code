from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plotting_common import (
    COLORS,
    DATASET_LABELS,
    DEFAULT_BASE_PATH,
    find_latest_accepted_run,
    panel_label,
    read_csv_checked,
    save_figure,
    set_publication_style,
    style_axis,
)


DATASET_ORDER = ["XJTU", "HUST", "TJU_NCM", "TJU_NCA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, default=None, help="Explicit Stage 1 run")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def stage1_run(base_path: Path, explicit: Optional[Path]) -> Path:
    required = [
        "manifests/battery_manifest.csv",
        "manifests/cycle_manifest.csv",
        *[f"processed_batteries/{name}" for name in DATASET_ORDER],
    ]
    return find_latest_accepted_run(
        base_path,
        "DATAPROCESSING_V2_2",
        "2.2",
        required,
        explicit,
    )


def load_manifests(run: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    batteries = read_csv_checked(
        run / "manifests" / "battery_manifest.csv",
        [
            "dataset",
            "battery_id",
            "valid_cycles",
            "soh_label_eligible_cycles",
            "processed_file",
        ],
    )
    cycles = read_csv_checked(
        run / "manifests" / "cycle_manifest.csv",
        [
            "dataset",
            "battery_id",
            "cycle",
            "capacity_ah",
            "soh_label_eligible",
            "curve_quality_eligible",
        ],
    )
    for column in ("soh_label_eligible", "curve_quality_eligible"):
        if cycles[column].dtype != bool:
            cycles[column] = cycles[column].astype(str).str.lower().eq("true")
    return batteries, cycles


def choose_representative_rows(batteries: pd.DataFrame) -> Dict[str, pd.Series]:
    selected: Dict[str, pd.Series] = {}
    for dataset in DATASET_ORDER:
        subset = batteries.loc[batteries["dataset"].eq(dataset)].copy()
        if subset.empty:
            raise ValueError(f"No batteries found for {dataset}")
        target = float(subset["soh_label_eligible_cycles"].median())
        subset["distance"] = (
            subset["soh_label_eligible_cycles"].astype(float) - target
        ).abs()
        subset = subset.sort_values(["distance", "battery_id"], kind="stable")
        selected[dataset] = subset.iloc[0]
    return selected


def load_representative_payloads(
    run: Path, rows: Dict[str, pd.Series]
) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for dataset, row in rows.items():
        relative = str(row["processed_file"]).replace("\\", "/")
        path = run / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Representative battery file missing: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or "cycles" not in payload:
            raise ValueError(f"Unexpected Stage 1 battery payload: {path}")
        payloads[dataset] = payload
    return payloads


def midlife_cycle(payload: dict) -> dict:
    cycles = [
        item
        for item in payload["cycles"]
        if bool(item.get("soh_label_eligible", False))
        and bool(item.get("curve_quality_eligible", False))
    ]
    if not cycles:
        raise ValueError(f"No eligible curves for {payload.get('battery_id')}")
    cycles.sort(key=lambda item: int(item["cycle"]))
    return cycles[len(cycles) // 2]


def phase_xy(cycle: dict, phase_name: str) -> Tuple[np.ndarray, np.ndarray]:
    phase = cycle.get(phase_name, {})
    throughput = np.asarray(phase.get("capacity_throughput_ah", []), dtype=float)
    voltage = np.asarray(phase.get("voltage_v", []), dtype=float)
    length = min(len(throughput), len(voltage))
    throughput, voltage = throughput[:length], voltage[:length]
    finite = np.isfinite(throughput) & np.isfinite(voltage)
    throughput, voltage = throughput[finite], voltage[finite]
    if len(throughput) < 3 or float(np.ptp(throughput)) <= 0:
        raise ValueError(f"Invalid {phase_name} curve in cycle {cycle.get('cycle')}")
    x = 100.0 * (throughput - throughput[0]) / (throughput[-1] - throughput[0])
    return x, voltage


def plot_phase_profiles(ax: plt.Axes, payloads: Dict[str, dict], phase: str) -> None:
    for dataset in DATASET_ORDER:
        cycle = midlife_cycle(payloads[dataset])
        x, voltage = phase_xy(cycle, phase)
        ax.plot(x, voltage, color=COLORS[dataset], label=DATASET_LABELS[dataset])
    ax.set_xlabel("Normalized phase throughput (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{phase.capitalize()} voltage profiles")
    style_axis(ax)


def normalized_soh_trajectories(cycles: pd.DataFrame) -> Dict[str, np.ndarray]:

    grid = np.linspace(0, 100, 101)
    result: Dict[str, np.ndarray] = {}
    eligible = cycles.loc[
        cycles["soh_label_eligible"] & cycles["curve_quality_eligible"]
    ].copy()
    eligible["capacity_ah"] = pd.to_numeric(
        eligible["capacity_ah"], errors="coerce"
    )
    for dataset in DATASET_ORDER:
        curves = []
        for _, group in eligible.loc[eligible["dataset"].eq(dataset)].groupby(
            "battery_id", sort=True
        ):
            group = group.sort_values("cycle", kind="stable")
            capacity = group["capacity_ah"].to_numpy(float)
            finite = np.isfinite(capacity) & (capacity > 0)
            capacity = capacity[finite]
            if len(capacity) < 5:
                continue
            reference_capacity = float(capacity[0])
            soh = capacity / reference_capacity


            soh = np.clip(soh, 0.0, 1.0)
            life = np.linspace(0, 100, len(soh))
            curves.append(np.interp(grid, life, soh))
        if not curves:
            raise ValueError(f"No valid SOH trajectories for {dataset}")
        result[dataset] = np.asarray(curves)
    return result


def plot_soh_summary(ax: plt.Axes, trajectories: Dict[str, np.ndarray]) -> None:
    grid = np.linspace(0, 100, 101)
    for dataset in DATASET_ORDER:
        values = trajectories[dataset]
        median = np.nanmedian(values, axis=0)
        lower, upper = np.nanquantile(values, [0.25, 0.75], axis=0)
        ax.fill_between(grid, lower, upper, color=COLORS[dataset], alpha=0.16, lw=0)
    ax.plot(grid, median, color=COLORS[dataset], label=DATASET_LABELS[dataset])
    ax.axhline(0.8, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Normalized cycle life (%)")
    ax.set_ylabel("SOH")
    ax.set_title("Battery-level SOH trajectories")
    style_axis(ax)


def plot_representative_capacity(
    ax: plt.Axes, cycles: pd.DataFrame, rows: Dict[str, pd.Series]
) -> None:
    for dataset in DATASET_ORDER:
        battery_id = rows[dataset]["battery_id"]
        group = cycles.loc[
            cycles["dataset"].eq(dataset)
            & cycles["battery_id"].eq(battery_id)
            & cycles["soh_label_eligible"]
            & cycles["curve_quality_eligible"]
        ].sort_values("cycle", kind="stable")
        ax.plot(
            group["cycle"],
            group["capacity_ah"],
            color=COLORS[dataset],
            label=DATASET_LABELS[dataset],
        )
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (Ah)")
    ax.set_title("Representative capacity trajectories")
    style_axis(ax)


def plot_dataset_scale(ax: plt.Axes, batteries: pd.DataFrame) -> None:
    cell_counts = (
        batteries.groupby("dataset", sort=False)["battery_id"].nunique().reindex(DATASET_ORDER)
    )
    cycle_counts = (
        batteries.groupby("dataset", sort=False)["soh_label_eligible_cycles"]
        .sum()
        .reindex(DATASET_ORDER)
    )
    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    bars1 = ax.bar(
        x - width / 2,
        cell_counts,
        width,
        color=COLORS["light_blue"],
        label="Batteries",
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        cycle_counts,
        width,
        color=COLORS["yellow"],
        label="Eligible cycles",
        zorder=3,
    )
    ax.set_yscale("log")


    ax.set_ylim(0.7, 1.2e7)
    ax.set_yticks([10.0 ** power for power in range(8)])
    ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASET_ORDER], rotation=18)
    ax.set_ylabel(r"Count (log$_{10}$ scale)")
    ax.set_title("Dataset scale")
    ax.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8)
    for bars in (bars1, bars2):
        for bar in bars:
            value = int(round(bar.get_height()))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.08,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=5.4,
                rotation=0,
            )
    style_axis(ax, "y")


def plot_cycle_count_distribution(ax: plt.Axes, batteries: pd.DataFrame) -> None:
    data = [
        batteries.loc[batteries["dataset"].eq(name), "soh_label_eligible_cycles"].to_numpy(float)
        for name in DATASET_ORDER
    ]
    parts = ax.violinplot(data, positions=np.arange(1, 5), showmedians=True, widths=0.75)
    for body, dataset in zip(parts["bodies"], DATASET_ORDER):
        body.set_facecolor(COLORS[dataset])
        body.set_edgecolor("none")
        body.set_alpha(0.65)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#555555")
            parts[key].set_linewidth(0.7)
    ax.set_xticks(np.arange(1, 5), [DATASET_LABELS[name] for name in DATASET_ORDER], rotation=18)
    ax.set_ylabel("Eligible cycles per battery")
    ax.set_title("Battery life distribution")
    style_axis(ax, "y")


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = stage1_run(args.base_path, args.run)
    output_dir = args.output_dir or args.base_path / "FIG_FINAL" / "Figure1"
    batteries, cycles = load_manifests(run)
    rows = choose_representative_rows(batteries)
    payloads = load_representative_payloads(run, rows)
    trajectories = normalized_soh_trajectories(cycles)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), constrained_layout=True)
    plot_phase_profiles(axes[0, 0], payloads, "charge")
    plot_phase_profiles(axes[0, 1], payloads, "discharge")
    plot_soh_summary(axes[1, 0], trajectories)
    plot_representative_capacity(axes[1, 1], cycles, rows)
    plot_dataset_scale(axes[2, 0], batteries)
    plot_cycle_count_distribution(axes[2, 1], batteries)

    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, label, x=-0.22, y=1.15)
    handles = [
        Line2D([0], [0], color=COLORS[name], lw=1.8, label=DATASET_LABELS[name])
        for name in DATASET_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=4,
        columnspacing=1.35,
    )
    save_figure(fig, output_dir, "Figure1_dataset_overview")
    plt.close(fig)


if __name__ == "__main__":
    main()
