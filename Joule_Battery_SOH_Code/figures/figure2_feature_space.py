from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plotting_common import (
    COLORS,
    DATASET_LABELS,
    DEFAULT_BASE_PATH,
    feature_label,
    find_latest_accepted_run,
    panel_label,
    read_csv_checked,
    report_directory,
    save_figure,
    set_publication_style,
    style_axis,
)


DATASET_ORDER = ["XJTU", "HUST", "TJU_NCM", "TJU_NCA"]
GROUP_ORDER = [
    "charge_voltage_shape",
    "discharge_voltage_shape",
    "current_stress",
    "charge_discharge_relation",
]
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
HEATMAP_FEATURES = [
    "charge_voltage_mean_v",
    "charge_voltage_curvature_v_per_q",
    "discharge_voltage_mean_v",
    "discharge_voltage_gradient_abs_p90_v_per_q",
    "charge_c_rate_mean",
    "discharge_c_rate_mean",
    "voltage_hysteresis_mid_soc_v",
    "voltage_hysteresis_high_soc_v",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, default=None, help="Explicit Stage 2 run")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def locate_run(base_path: Path, explicit: Optional[Path]) -> Path:
    required = [
        "feature_definition.csv",
        *[f"datasets/{name}_features.pkl" for name in DATASET_ORDER],
    ]
    return find_latest_accepted_run(
        base_path,
        "FEATUREPROCESSING_V2_0",
        "2.0.2",
        required,
        explicit,
    )


def load_payloads(run: Path) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for dataset in DATASET_ORDER:
        path = run / "datasets" / f"{dataset}_features.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 2 dataset missing: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        required = {"X", "y_soh", "feature_names", "battery_id", "cycle"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"{path} is missing payload keys: {missing}")
        X = np.asarray(payload["X"])
        if X.ndim != 2 or X.shape[1] != 39 or not np.all(np.isfinite(X)):
            raise ValueError(f"Invalid 39-feature matrix in {path}: {X.shape}")
        payloads[dataset] = payload
    reference = list(payloads[DATASET_ORDER[0]]["feature_names"])
    for dataset in DATASET_ORDER[1:]:
        if list(payloads[dataset]["feature_names"]) != reference:
            raise ValueError(f"Feature order mismatch in {dataset}")
    return payloads


def balanced_indices(battery_ids: np.ndarray, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.asarray(battery_ids, dtype=object)
    unique = np.asarray(sorted(set(ids.tolist())), dtype=object)
    per_battery = max(1, int(np.ceil(target / len(unique))))
    chosen = []
    for battery_id in unique:
        candidates = np.flatnonzero(ids == battery_id)
        take = min(per_battery, len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > target:
        chosen = rng.choice(chosen, size=target, replace=False)
    return np.sort(chosen)


def balanced_samples(
    payloads: Dict[str, dict], per_domain: int = 2500
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    samples: Dict[str, np.ndarray] = {}
    indices: Dict[str, np.ndarray] = {}
    for position, dataset in enumerate(DATASET_ORDER):
        payload = payloads[dataset]
        index = balanced_indices(
            np.asarray(payload["battery_id"], dtype=object),
            min(per_domain, len(payload["X"])),
            2026 + position,
        )
        samples[dataset] = np.asarray(payload["X"], dtype=float)[index]
        indices[dataset] = index
    return samples, indices


def plot_feature_taxonomy(ax: plt.Axes, definitions: pd.DataFrame) -> None:
    counts = definitions.groupby("group")["feature"].count().reindex(GROUP_ORDER)
    y = np.arange(len(GROUP_ORDER))
    bars = ax.barh(
        y,
        counts,
        color=[GROUP_COLORS[group] for group in GROUP_ORDER],
        height=0.6,
        zorder=3,
    )
    ax.set_yticks(y, [GROUP_LABELS[group] for group in GROUP_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Candidate features")
    ax.set_title("Feature taxonomy")

    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_ylim(4.75, -0.6)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    ax.legend(
        handles=[Patch(facecolor=GROUP_COLORS[group], label=GROUP_LABELS[group])
                 for group in GROUP_ORDER],
        loc="lower center", ncol=2, fontsize=5.1, labelspacing=0.5,
        columnspacing=0.6, handlelength=0.8, handletextpad=0.25,
        frameon=False,
    )
    style_axis(ax, "x")


def plot_soh_distributions(ax: plt.Axes, payloads: Dict[str, dict]) -> None:
    values = []
    capped_counts = []
    for position, dataset in enumerate(DATASET_ORDER):
        y = np.asarray(payloads[dataset]["y_soh"], dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError(f"{dataset} has invalid SOH labels")


        above_one = int(np.count_nonzero(y > 1.0))
        capped_counts.append((dataset, above_one, len(y)))
        y = np.minimum(y, 1.0)
        rng = np.random.default_rng(4100 + position)
        if len(y) > 6000:
            y = rng.choice(y, size=6000, replace=False)
        values.append(y)
    parts = ax.violinplot(values, positions=np.arange(1, 5), showmedians=True, widths=0.76)
    for body, dataset in zip(parts["bodies"], DATASET_ORDER):
        body.set_facecolor(COLORS[dataset])
        body.set_edgecolor("none")
        body.set_alpha(0.65)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#555555")
            parts[key].set_linewidth(0.7)
    ax.set_xticks(np.arange(1, 5), [DATASET_LABELS[name] for name in DATASET_ORDER], rotation=18)
    ax.set_ylabel("Displayed SOH (capped at 1)")
    ax.set_title("SOH distributions (display capped at 1)")
    ax.set_ylim(0, 1)
    ax.text(
        0.02, 0.48,
        "Raw $Q/Q_0$ > 1 shown at 1:\n"
        + "\n".join(f"{DATASET_LABELS[name]}: {100 * count / total:.1f}%"
                    for name, count, total in capped_counts),
        transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    ax.legend(
        handles=[Patch(facecolor=COLORS[name], alpha=0.65,
                       label=DATASET_LABELS[name]) for name in DATASET_ORDER],
        loc="lower left", ncol=2, fontsize=5.8, frameon=False,
        handlelength=1.0, columnspacing=0.8,
    )
    style_axis(ax, "y")


def plot_pca(ax: plt.Axes, samples: Dict[str, np.ndarray]) -> StandardScaler:
    stacked = np.vstack([samples[name] for name in DATASET_ORDER])
    scaler = StandardScaler().fit(stacked)
    transformed = scaler.transform(stacked)
    projection = PCA(n_components=2, random_state=2026).fit(transformed)
    Z = projection.transform(transformed)
    start = 0
    for dataset in DATASET_ORDER:
        count = len(samples[dataset])
        points = Z[start : start + count]
        start += count
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=5,
            alpha=0.35,
            linewidths=0,
            color=COLORS[dataset],
            rasterized=True,
            label=DATASET_LABELS[dataset],
        )
    variance = 100 * projection.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)")
    ax.set_title("Domain structure of candidate features")
    ax.legend(loc="upper right", ncol=2, fontsize=5.6, frameon=False,
              markerscale=1.8, columnspacing=0.8)
    style_axis(ax)
    return scaler


def plot_domain_heatmap(
    ax: plt.Axes,
    payloads: Dict[str, dict],
    samples: Dict[str, np.ndarray],
    scaler: StandardScaler,
) -> None:
    names = list(payloads[DATASET_ORDER[0]]["feature_names"])
    index = [names.index(feature) for feature in HEATMAP_FEATURES]
    matrix = []
    for dataset in DATASET_ORDER:
        standardized = scaler.transform(samples[dataset])
        matrix.append(np.median(standardized[:, index], axis=0))
    matrix = np.asarray(matrix).T
    limit = max(1.0, float(np.nanquantile(np.abs(matrix), 0.95)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(4), [DATASET_LABELS[name] for name in DATASET_ORDER], rotation=18)
    ax.set_yticks(np.arange(len(index)), [feature_label(name) for name in HEATMAP_FEATURES])
    ax.set_title("Cross-domain feature shifts")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Median standardized value")
    colorbar.outline.set_linewidth(0.6)
    ax.legend(
        handles=[Patch(facecolor="#B2182B", label="Positive z"),
                 Patch(facecolor="#2166AC", label="Negative z")],
        loc="upper right", fontsize=5.5, frameon=True,
        facecolor="white", edgecolor="none", framealpha=0.9,
        handlelength=0.9, borderpad=0.3,
    )
    style_axis(ax)


def normalized_feature_trajectories(payload: dict, feature: str) -> np.ndarray:
    names = list(payload["feature_names"])
    feature_index = names.index(feature)
    X = np.asarray(payload["X"], dtype=float)
    battery_ids = np.asarray(payload["battery_id"], dtype=object)
    cycles = np.asarray(payload["cycle"], dtype=int)
    grid = np.linspace(0, 100, 101)
    curves = []
    for battery_id in sorted(set(battery_ids.tolist())):
        index = np.flatnonzero(battery_ids == battery_id)
        order = np.argsort(cycles[index], kind="stable")
        values = X[index[order], feature_index]
        if len(values) < 5 or not np.all(np.isfinite(values)):
            continue
        life = np.linspace(0, 100, len(values))
        curves.append(np.interp(grid, life, values))
    return np.asarray(curves)


def plot_feature_trajectory(
    ax: plt.Axes, payloads: Dict[str, dict], feature: str
) -> None:
    grid = np.linspace(0, 100, 101)
    for dataset in DATASET_ORDER:
        curves = normalized_feature_trajectories(payloads[dataset], feature)
        median = np.median(curves, axis=0)
        lower, upper = np.quantile(curves, [0.25, 0.75], axis=0)
        ax.fill_between(grid, lower, upper, color=COLORS[dataset], alpha=0.14, linewidth=0)
        ax.plot(grid, median, color=COLORS[dataset], label=DATASET_LABELS[dataset])
    ax.set_xlabel("Normalized cycle life (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(feature_label(feature))
    ax.legend(loc="best", ncol=2, frameon=False, fontsize=5.8,
              columnspacing=0.8, handlelength=1.4)
    style_axis(ax)


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = locate_run(args.base_path, args.run)
    output_dir = args.output_dir or args.base_path / "FIG_FINAL" / "Figure2"
    report = report_directory(run)
    definitions = read_csv_checked(report / "feature_definition.csv", ["feature", "group", "unit"])
    payloads = load_payloads(run)
    samples, _ = balanced_samples(payloads)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.5), constrained_layout=True)
    plot_feature_taxonomy(axes[0, 0], definitions)
    plot_soh_distributions(axes[0, 1], payloads)
    scaler = plot_pca(axes[1, 0], samples)
    plot_domain_heatmap(axes[1, 1], payloads, samples, scaler)
    plot_feature_trajectory(axes[2, 0], payloads, "charge_voltage_q50_v")
    plot_feature_trajectory(axes[2, 1], payloads, "discharge_voltage_mean_v")

    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, label, x=-0.22, y=1.15)
    save_figure(fig, output_dir, "Figure2_feature_space")
    plt.close(fig)


if __name__ == "__main__":
    main()
