from __future__ import annotations

import argparse
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_common import (
    COLORS, DATASET_LABELS, DEFAULT_BASE_PATH, feature_label,
    find_latest_accepted_run, panel_label, report_directory, save_figure,
    set_publication_style, style_axis,
)

ROOT = "INTERPRETABILITY_V5_0_2_CONDITIONAL"
BUILD = "STAGE5_V5_0_2_CONDITIONAL_REPLAY_20261006"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
TARGETS = ("TJU_NCM", "TJU_NCA")
FILES = ("processing_summary.json", "processing_config.json",
         "feature_sensitivity.csv", "temporal_sensitivity.csv",
         "branch_removal_test_description.csv")
BRANCH_LABELS = {
    "CNN_BiLSTM_Attention": "CNN-BiLSTM",
    "MLP_Feature_Attention": "MLP attention",
    "HGB_Window_SOH": "Window HGB",
    "Ridge_Window_SOH": "Window Ridge",
}
SCOPES = ("pooled_cycles", "battery_macro")
SCOPE_LABELS = {"pooled_cycles": "Pooled cycles", "battery_macro": "Battery macro"}
SCOPE_COLORS = {"pooled_cycles": COLORS["light_blue"], "battery_macro": COLORS["pink"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, help="Stage 5 run, extracted report, or ZIP")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def locate_run(base: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None and explicit.is_file() and explicit.suffix.lower() == ".zip":
        return explicit
    return find_latest_accepted_run(base, ROOT, "5.0.2", list(FILES), explicit)


def report_bytes(run: Path, filename: str) -> bytes:
    if run.is_file():
        with zipfile.ZipFile(run) as archive:
            matches = [n for n in archive.namelist() if Path(n).name == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected one {filename} in {run}; found {len(matches)}")
            return archive.read(matches[0])
    return (report_directory(run) / filename).read_bytes()


def read_csv(run: Path, name: str, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(report_bytes(run, name)))
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    return frame


def load_data(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads(report_bytes(run, "processing_summary.json").decode("utf-8-sig"))
    config = json.loads(report_bytes(run, "processing_config.json").decode("utf-8-sig"))
    if summary.get("validation_status") != "PASS":
        raise ValueError("Stage 5 validation did not PASS")
    if config.get("schema_version") != "5.0.2" or config.get("script_build") != BUILD:
        raise ValueError("Expected the accepted Stage 5 V5.0.2 conditional build")
    if config.get("stage4_script_build") != STAGE4_BUILD:
        raise ValueError("The Stage 5 report has a different Stage 4 lineage")
    if config.get("explanation_measure") != "non-additive validation-reference replacement sensitivity":
        raise ValueError("Unsupported explanation measure")
    if config.get("test_labels_used_for_explanation_or_model_selection") is not False:
        raise ValueError("Test labels affected explanation or model selection")
    names = list(config.get("feature_names", []))
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("Expected 20 distinct selected features")

    fields = ("target", "component", "normalized_sensitivity",
              "normalized_ci95_lower", "normalized_ci95_upper",
              "battery_count", "explained_window_count")
    feature = read_csv(run, "feature_sensitivity.csv", fields)
    temporal = read_csv(run, "temporal_sensitivity.csv", fields)
    removal = read_csv(run, "branch_removal_test_description.csv", (
        "target", "dropped_branch", "evaluation_scope", "frozen_ensemble_RMSE",
        "without_branch_RMSE", "RMSE_increase_when_branch_removed",
        "status", "test_labels_used_for_model_selection"))

    for kind, frame in (("feature", feature), ("temporal", temporal)):
        if frame.duplicated(["target", "component"]).any():
            raise ValueError(f"Duplicate {kind} keys")
        if set(frame["target"]) != set(TARGETS):
            raise ValueError(f"Incomplete {kind} target domains")
        values = frame[["normalized_sensitivity", "normalized_ci95_lower",
                        "normalized_ci95_upper"]].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {kind} sensitivity values")
        if ((values[:, 1] > values[:, 0] + 1e-7)
                | (values[:, 0] > values[:, 2] + 1e-7)).any():
            raise ValueError(f"Invalid {kind} confidence intervals")
        for target in TARGETS:
            rows = frame.loc[frame["target"].eq(target)]
            if len(rows) != 20 or not np.isclose(rows["normalized_sensitivity"].sum(), 1.0, atol=2e-5):
                raise ValueError(f"{target}: invalid normalized {kind} rows")

    for target in TARGETS:
        if set(feature.loc[feature["target"].eq(target), "component"]) != set(names):
            raise ValueError(f"{target}: feature names differ from Stage 5 configuration")
        offsets = pd.to_numeric(temporal.loc[temporal["target"].eq(target), "component"]).astype(int)
        if set(offsets) != set(range(-19, 1)):
            raise ValueError(f"{target}: invalid temporal positions")
        subset = removal.loc[removal["target"].eq(target)]
        if len(subset) != 4 or len(set(subset["dropped_branch"])) != 2:
            raise ValueError(f"{target}: incomplete branch-removal comparisons")
    if removal.duplicated(["target", "dropped_branch", "evaluation_scope"]).any():
        raise ValueError("Duplicate branch-removal comparisons")
    if set(removal["evaluation_scope"]) != set(SCOPES):
        raise ValueError("Unexpected branch-removal scopes")
    if not removal["status"].eq("descriptive_test_evaluation_only").all():
        raise ValueError("Branch removal was undefined in at least one condition")
    if not removal["test_labels_used_for_model_selection"].eq(False).all():
        raise ValueError("Test labels affected branch selection")
    delta = removal["without_branch_RMSE"] - removal["frozen_ensemble_RMSE"]
    if not np.isfinite(delta.to_numpy(float)).all() or not np.allclose(
        delta, removal["RMSE_increase_when_branch_removed"], atol=1e-9):
        raise ValueError("Branch-removal RMSE differences are inconsistent")
    return feature, temporal, removal


def plot_feature(ax: plt.Axes, frame: pd.DataFrame, target: str) -> None:
    data = (frame.loc[frame["target"].eq(target)]
            .nlargest(10, "normalized_sensitivity")
            .sort_values("normalized_sensitivity", kind="stable"))
    y = np.arange(len(data))
    center = data["normalized_sensitivity"].to_numpy(float)
    lo = np.maximum(0, center - data["normalized_ci95_lower"].to_numpy(float))
    hi = np.maximum(0, data["normalized_ci95_upper"].to_numpy(float) - center)
    ax.barh(y, 100 * center, color=COLORS[target], height=0.64, alpha=0.82, zorder=2)
    ax.errorbar(100 * center, y, xerr=100 * np.vstack((lo, hi)),
                fmt="none", ecolor=COLORS["black"], elinewidth=0.7, capsize=1.6, zorder=3)
    ax.set_yticks(y, [feature_label(x) for x in data["component"]])
    ax.set_xlabel("Normalized replacement sensitivity (%)")
    ax.set_title(f"{DATASET_LABELS[target]} | leading feature sensitivities")
    style_axis(ax, "x")


def plot_cross_domain(ax: plt.Axes, feature: pd.DataFrame) -> None:
    data = feature.pivot(index="component", columns="target", values="normalized_sensitivity")
    if data.isna().any().any():
        raise ValueError("Incomplete cross-domain feature data")
    data = (data.assign(mean=data[list(TARGETS)].mean(axis=1))
            .nlargest(10, "mean").sort_values("mean"))
    y = np.arange(len(data))
    ax.hlines(y, 100 * data[TARGETS[0]], 100 * data[TARGETS[1]],
              color="#BDBDBD", linewidth=0.8)
    for target in TARGETS:
        ax.scatter(100 * data[target], y, s=16, color=COLORS[target],
                   label=DATASET_LABELS[target], zorder=3)
    ax.set_yticks(y, [feature_label(x) for x in data.index])
    ax.set_xlabel("Normalized replacement sensitivity (%)")
    ax.set_title("Feature sensitivities across targets")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax, "x")


def plot_temporal(ax: plt.Axes, temporal: pd.DataFrame) -> None:
    for target in TARGETS:
        data = temporal.loc[temporal["target"].eq(target)].copy()
        data["relative_cycle"] = data["component"].astype(int)
        data = data.sort_values("relative_cycle")
        ax.plot(data["relative_cycle"], 100 * data["normalized_sensitivity"],
                marker="o", markersize=2.4, linewidth=1.3,
                color=COLORS[target], label=DATASET_LABELS[target])
    ax.set_xticks([-19, -15, -10, -5, 0])
    ax.set_xlabel("Cycle relative to current cycle")
    ax.set_ylabel("Normalized replacement sensitivity (%)")
    ax.set_title("Temporal sensitivity of the final ensemble")
    ax.legend(loc="upper left", frameon=False)
    style_axis(ax, "y")


def plot_removal(ax: plt.Axes, removal: pd.DataFrame, target: str) -> None:
    data = removal.loc[removal["target"].eq(target)]
    branches = data["dropped_branch"].drop_duplicates().tolist()
    if len(branches) != 2 or not set(branches).issubset(BRANCH_LABELS):
        raise ValueError(f"{target}: unknown branches")
    y = np.arange(len(branches))
    for scope, shift in (("pooled_cycles", -0.16), ("battery_macro", 0.16)):
        rows = data.loc[data["evaluation_scope"].eq(scope)].set_index("dropped_branch")
        values = 1000 * rows.loc[branches, "RMSE_increase_when_branch_removed"].to_numpy(float)
        ax.barh(y + shift, values, height=0.30, color=SCOPE_COLORS[scope],
                label=SCOPE_LABELS[scope], edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_yticks(y, [BRANCH_LABELS[x] for x in branches])
    ax.invert_yaxis()


    ax.set_ylim(len(branches) - 0.55, -1.25)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel(r"RMSE increase after removal ($\times 10^{-3}$ SOH)")
    ax.set_title(f"{DATASET_LABELS[target]} | frozen branch removal")
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98),
              ncol=1, frameon=False)
    style_axis(ax, "x")


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = locate_run(args.base_path, args.run)
    feature, temporal, removal = load_data(run)
    output_dir = args.output_dir or args.base_path / "FIG_FINAL" / "Figure5"
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.6), constrained_layout=True)
    plot_feature(axes[0, 0], feature, "TJU_NCM")
    plot_feature(axes[0, 1], feature, "TJU_NCA")
    plot_cross_domain(axes[1, 0], feature)
    plot_temporal(axes[1, 1], temporal)
    plot_removal(axes[2, 0], removal, "TJU_NCM")
    plot_removal(axes[2, 1], removal, "TJU_NCA")
    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, label, x=-0.20, y=1.09)
    save_figure(fig, output_dir, "Figure5_interpretability")
    plt.close(fig)


if __name__ == "__main__":
    main()
