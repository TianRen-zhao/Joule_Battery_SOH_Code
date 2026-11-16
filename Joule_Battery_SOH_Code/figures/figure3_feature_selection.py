from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting_common import (
    COLORS,
    DEFAULT_BASE_PATH,
    feature_label,
    find_latest_accepted_run,
    grouped_offsets,
    minmax_scale,
    panel_label,
    read_csv_checked,
    report_directory,
    save_figure,
    set_publication_style,
    style_axis,
)


GROUP_LABELS = {
    "charge_voltage_shape": "Charge-voltage shape",
    "discharge_voltage_shape": "Discharge-voltage shape",
    "current_stress": "Current stress",
    "charge_discharge_relation": "Charge-discharge relation",
}
GROUP_COLORS = {
    "charge_voltage_shape": "#6AB4DC",
    "discharge_voltage_shape": "#42B8B1",
    "current_stress": "#E78489",
    "charge_discharge_relation": "#D8CA77",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, default=None, help="Explicit Stage 3 run or extracted report")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def locate_run(base_path: Path, explicit: Optional[Path]) -> Path:
    return find_latest_accepted_run(
        base_path,
        "FEATURESELECTION_V3_0_1",
        "3.0.1",
        [
            "feature_ranking.csv",
            "bootstrap_stability.csv",
            "redundancy_matrix.csv",
            "selected_feature_manifest.csv",
        ],
        explicit,
    )


def load_data(run: Path):
    report = report_directory(run)
    ranking = read_csv_checked(
        report / "feature_ranking.csv",
        [
            "feature",
            "pc_balanced_score",
            "tc_balanced_score",
            "raw_pc_tc_score",
            "score_ci025",
            "score_ci975",
            "redundancy_constrained_top_k_frequency",
        ],
    )
    selected = read_csv_checked(
        report / "selected_feature_manifest.csv",
        [
            "selection_rank",
            "feature",
            "raw_pc_tc_score",
            "maximum_redundancy_with_earlier_selection",
            "redundancy_adjusted_selection_score",
            "group",
            "pc_balanced_score",
            "tc_balanced_score",
            "raw_top_k_frequency",
            "redundancy_constrained_top_k_frequency",
        ],
    ).sort_values("selection_rank")
    bootstrap = read_csv_checked(
        report / "bootstrap_stability.csv",
        ["feature", "score_ci025", "score_ci975", "redundancy_constrained_top_k_frequency"],
    )
    redundancy = read_csv_checked(report / "redundancy_matrix.csv", ["feature"])
    if len(selected) != 20 or selected["feature"].nunique() != 20:
        raise ValueError("Stage 3 must contain exactly 20 unique selected features")
    if ranking["feature"].nunique() != 39:
        raise ValueError("Stage 3 ranking must contain 39 unique candidate features")
    if set(ranking["feature"]) != set(bootstrap["feature"]):
        raise ValueError("Ranking and bootstrap feature sets do not match")
    return ranking, selected, redundancy


def plot_capability_space(ax: plt.Axes, ranking: pd.DataFrame, selected: pd.DataFrame) -> None:
    selected_names = set(selected["feature"])
    mask = ranking["feature"].isin(selected_names)
    sizes = minmax_scale(ranking["raw_pc_tc_score"].to_numpy(float), 18, 90)
    ax.scatter(
        ranking.loc[~mask, "tc_balanced_score"],
        ranking.loc[~mask, "pc_balanced_score"],
        s=sizes[~mask.to_numpy()],
        color=COLORS["rejected"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.75,
        label="Not selected",
        zorder=2,
    )
    ax.scatter(
        ranking.loc[mask, "tc_balanced_score"],
        ranking.loc[mask, "pc_balanced_score"],
        s=sizes[mask.to_numpy()],
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        label="Selected",
        zorder=3,
    )
    for _, row in selected.nsmallest(3, "selection_rank").iterrows():
        ax.annotate(
            feature_label(row["feature"]),
            (row["tc_balanced_score"], row["pc_balanced_score"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.8,
        )
    x_median = float(ranking["tc_balanced_score"].median())
    y_median = float(ranking["pc_balanced_score"].median())
    ax.axvline(x_median, color="#999999", lw=0.65, ls="--")
    ax.axhline(y_median, color="#999999", lw=0.65, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Trend consistency (TC)")
    ax.set_ylabel("Predictive capability (PC)")
    ax.set_title("Source-domain capability space")
    ax.text(
        0.98,
        0.04,
        "red: selected\ngray: not selected",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    style_axis(ax)


def plot_selection_scores(ax: plt.Axes, selected: pd.DataFrame) -> None:
    data = selected.sort_values("selection_rank", ascending=False)
    y = np.arange(len(data))
    ax.barh(
        y,
        data["raw_pc_tc_score"],
        color="#E8E8E8",
        edgecolor="none",
        label="Raw PC-TC score",
        zorder=2,
    )
    ax.barh(
        y,
        data["redundancy_adjusted_selection_score"],
        color=COLORS["selected"],
        edgecolor="none",
        label="Redundancy-adjusted score",
        zorder=3,
    )
    ax.set_yticks(y, [f"{int(rank)}  {feature_label(name)}" for rank, name in zip(data["selection_rank"], data["feature"])])
    ax.set_xlabel("Score (gray: raw; red: adjusted)")
    ax.set_title("Final redundancy-aware ranking")
    style_axis(ax, "x")


def plot_pc_tc_components(ax: plt.Axes, selected: pd.DataFrame) -> None:
    data = selected.sort_values("selection_rank", ascending=False)
    y = np.arange(len(data))
    offsets, height = grouped_offsets(2, 0.72)
    ax.barh(y + offsets[0], data["pc_balanced_score"], height, color=COLORS["PC"], label="PC")
    ax.barh(y + offsets[1], data["tc_balanced_score"], height, color=COLORS["TC"], label="TC")
    ax.set_yticks(y, [feature_label(name) for name in data["feature"]])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Score (blue: PC; teal: TC)")
    ax.set_title("PC and TC composition")
    style_axis(ax, "x")


def plot_bootstrap_stability(ax: plt.Axes, selected: pd.DataFrame) -> None:
    data = selected.sort_values("selection_rank", ascending=False)
    y = np.arange(len(data))
    ax.hlines(
        y,
        data["raw_top_k_frequency"],
        data["redundancy_constrained_top_k_frequency"],
        color="#BDBDBD",
        linewidth=0.8,
        zorder=1,
    )
    ax.scatter(data["raw_top_k_frequency"], y, color=COLORS["PC"], s=18, label="Raw Top-20")
    ax.scatter(
        data["redundancy_constrained_top_k_frequency"],
        y,
        color=COLORS["selected"],
        s=18,
        label="Redundancy-aware Top-20",
    )
    ax.set_yticks(y, [feature_label(name) for name in data["feature"]])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Frequency (blue: raw; red: adjusted)")
    ax.set_title("Bootstrap selection stability")
    style_axis(ax, "x")


def plot_selected_redundancy(
    ax: plt.Axes, redundancy: pd.DataFrame, selected: pd.DataFrame
) -> None:
    names = selected.sort_values("selection_rank")["feature"].tolist()
    indexed = redundancy.set_index("feature")
    missing = [name for name in names if name not in indexed.index or name not in indexed.columns]
    if missing:
        raise ValueError(f"Selected features missing from redundancy matrix: {missing}")
    matrix = indexed.loc[names, names].to_numpy(float)
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    tick_positions = [0, 4, 9, 14, 19]
    tick_labels = ["1", "5", "10", "15", "20"]
    ax.set_xticks(tick_positions, tick_labels, fontsize=5.8)
    ax.set_yticks(tick_positions, tick_labels, fontsize=5.8)
    ax.set_xlabel("Selection rank")
    ax.set_ylabel("Selection rank")
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    maximum = float(np.max(upper))
    ax.text(
        0.02,
        0.98,
        f"max pairwise redundancy = {maximum:.3f}\nhard threshold < 0.990",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    ax.set_title("Redundancy among selected features")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Absolute rank correlation")
    colorbar.outline.set_linewidth(0.6)


def plot_group_composition(ax: plt.Axes, selected: pd.DataFrame) -> None:
    counts = selected.groupby("group")["feature"].count().reindex(GROUP_LABELS, fill_value=0)
    labels = [GROUP_LABELS[group] for group in counts.index]
    bars = ax.barh(
        np.arange(len(labels)),
        counts,
        height=0.58,
        color=[GROUP_COLORS[group] for group in counts.index],
        zorder=3,
    )
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Selected features")
    ax.set_title("Composition of the 20-feature set")
    ax.set_xlim(0, max(counts) * 1.2)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_width() + 0.18, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    style_axis(ax, "x")


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = locate_run(args.base_path, args.run)
    output_dir = args.output_dir or args.base_path / "FIG_FINAL" / "Figure3"
    ranking, selected, redundancy = load_data(run)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 10.0), constrained_layout=True)
    plot_capability_space(axes[0, 0], ranking, selected)
    plot_selection_scores(axes[0, 1], selected)
    plot_pc_tc_components(axes[1, 0], selected)
    plot_bootstrap_stability(axes[1, 1], selected)
    plot_selected_redundancy(axes[2, 0], redundancy, selected)
    plot_group_composition(axes[2, 1], selected)
    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, label, x=-0.25, y=1.16)
    fig.text(
        0.5,
        -0.008,
        "Feature scoring used XJTU and HUST only; TJU-NCM and TJU-NCA labels were excluded.",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#555555",
    )
    save_figure(fig, output_dir, "Figure3_feature_selection")
    plt.close(fig)


if __name__ == "__main__":
    main()
