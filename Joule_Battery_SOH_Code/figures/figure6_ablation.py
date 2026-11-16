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
from matplotlib.lines import Line2D

from plotting_common import (
    COLORS, DATASET_LABELS, DEFAULT_BASE_PATH, find_latest_accepted_run,
    grouped_offsets, panel_label, report_directory, save_figure,
    set_publication_style, style_axis,
)


ROOT = "ABLATION_RESULTS_V6_0_CONDITIONAL"
BUILD = "STAGE6_V6_0_CONDITIONAL_MATCHED_20261006"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
REFERENCE_CASE = "Selected20_With_Source_Pretraining_Stage4"
TARGETS = ("TJU_NCM", "TJU_NCA")
BRANCHES = ("CNN_BiLSTM_Attention", "MLP_Feature_Attention",
            "HGB_Window_SOH", "Ridge_Window_SOH")
MODELS = (*BRANCHES, "Equal_Weight_Selected_Branches",
          "Equal_Weight_All_Four", "Adaptive_Ensemble")
MODEL_LABELS = ("CNN-BiLSTM", "MLP", "Window HGB", "Window Ridge",
                "Equal selected", "Equal all", "Adaptive")
BRANCH_COLORS = {
    "CNN_BiLSTM_Attention": COLORS["CNN_BiLSTM_Attention"],
    "MLP_Feature_Attention": COLORS["MLP_Feature_Attention"],
    "HGB_Window_SOH": COLORS["TC"],
    "Ridge_Window_SOH": COLORS["yellow"],
}
REQUIRED_FILES = (
    "processing_config.json", "processing_summary.json",
    "feature_selection_ablation_metrics.csv",
    "source_pretraining_ablation_metrics.csv",
    "ensemble_fusion_ablation_metrics.csv",
    "paired_battery_bootstrap_effects.csv", "ensemble_weights.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, help="Stage 6 run, extracted report, or ZIP")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def locate_run(base: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None and explicit.is_file() and explicit.suffix.lower() == ".zip":
        return explicit
    return find_latest_accepted_run(base, ROOT, "6.0", list(REQUIRED_FILES), explicit)


def report_bytes(run: Path, filename: str) -> bytes:
    if run.is_file():
        with zipfile.ZipFile(run) as archive:
            matches = [name for name in archive.namelist()
                       if Path(name).name == filename and not name.endswith("/")]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {filename} in {run}")
            return archive.read(matches[0])
    return (report_directory(run) / filename).read_bytes()


def report_csv(run: Path, filename: str, required: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(report_bytes(run, filename)))
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} lacks {sorted(missing)}")
    return frame


def load_data(run: Path) -> dict[str, pd.DataFrame]:
    summary = json.loads(report_bytes(run, "processing_summary.json").decode("utf-8-sig"))
    config = json.loads(report_bytes(run, "processing_config.json").decode("utf-8-sig"))
    if summary.get("validation_status") != "PASS":
        raise ValueError("Stage 6 validation did not PASS")
    if config.get("schema_version") != "6.0" or config.get("script_build") != BUILD:
        raise ValueError("Expected the conditional Stage 6 build")
    if config.get("stage4_build") != STAGE4_BUILD:
        raise ValueError("Stage 6 does not match the accepted conditional Stage 4 build")
    if config.get("target_test_used_for_training_or_selection") is not False:
        raise ValueError("Final-test labels influenced training or selection")
    if config.get("selected_feature_count") != 20 or config.get("full_feature_count") != 39:
        raise ValueError("Unexpected feature-control dimensions")
    if summary.get("integrity_checks") and not all(summary["integrity_checks"].values()):
        raise ValueError("Stage 6 reports a failed integrity check")

    metrics = ("target", "method", "evaluation_scope", "RMSE", "sample_count", "battery_count")
    data = {
        "feature": report_csv(run, "feature_selection_ablation_metrics.csv", metrics),
        "pretrain": report_csv(run, "source_pretraining_ablation_metrics.csv", metrics),
        "fusion": report_csv(run, "ensemble_fusion_ablation_metrics.csv", metrics),
        "effects": report_csv(run, "paired_battery_bootstrap_effects.csv", (
            "experiment", "target", "reference_method", "comparator_method",
            "evaluation_scope", "observed_rmse_difference",
            "rmse_difference_ci95_lower", "rmse_difference_ci95_upper",
            "positive_means_reference_better")),
        "weights": report_csv(run, "ensemble_weights.csv", (
            "case", "target", "model", "weight", "weight_scope",
            "condition_group", "selected", "fusion_mode", "test_labels_used")),
    }
    expected_methods = {
        "feature": ("Selected20_Transfer_Adaptive", "All39_Transfer_Adaptive"),
        "pretrain": ("With_Source_Pretraining", "Without_Source_Pretraining"),
        "fusion": MODELS,
    }
    for kind in ("feature", "pretrain", "fusion"):
        frame = data[kind]
        if frame.duplicated(["target", "evaluation_scope", "method"]).any():
            raise ValueError(f"Duplicate {kind} metric rows")
        for target in TARGETS:
            for scope in ("pooled_cycles", "battery_macro"):
                subset = frame.loc[frame["target"].eq(target)
                                   & frame["evaluation_scope"].eq(scope)]
                if set(subset["method"]) != set(expected_methods[kind]):
                    raise ValueError(f"Incomplete {kind} metrics for {target}/{scope}")
                if not np.isfinite(subset["RMSE"].to_numpy(float)).all():
                    raise ValueError(f"Non-finite {kind} RMSE")
    weights = data["weights"].loc[data["weights"]["case"].eq(REFERENCE_CASE)].copy()
    if weights.empty or weights["test_labels_used"].astype(str).str.lower().eq("true").any():
        raise ValueError("Stage 4 fusion weights are missing or test-label dependent")
    data["weights"] = weights
    effects = data["effects"]
    if not effects["positive_means_reference_better"].astype(str).str.lower().eq("true").all():
        raise ValueError("Unexpected paired effect sign convention")
    if effects.duplicated(["experiment", "target", "evaluation_scope", "comparator_method"]).any():
        raise ValueError("Duplicate bootstrap comparisons")
    if len(effects) != 16:
        raise ValueError("Expected eight comparisons for each evaluation scope")
    for field in ("observed_rmse_difference", "rmse_difference_ci95_lower",
                  "rmse_difference_ci95_upper"):
        if not np.isfinite(effects[field].to_numpy(float)).all():
            raise ValueError(f"Non-finite bootstrap value: {field}")
    if (effects["rmse_difference_ci95_lower"] > effects["rmse_difference_ci95_upper"]).any():
        raise ValueError("Reversed bootstrap confidence interval")
    return data


def plot_control(ax: plt.Axes, frame: pd.DataFrame, methods: tuple[str, str],
                 labels: tuple[str, str], colors: tuple[str, str], title: str) -> None:
    data = frame.loc[frame["evaluation_scope"].eq("pooled_cycles")]
    x = np.arange(2)
    offsets, width = grouped_offsets(2, 0.68)
    for method, label, color, offset in zip(methods, labels, colors, offsets):
        values = data.loc[data["method"].eq(method)].set_index("target").loc[list(TARGETS), "RMSE"]
        bars = ax.bar(x + offset, values, width, color=color, alpha=0.86,
                      label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.4f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, [DATASET_LABELS[t] for t in TARGETS])
    ax.set_ylim(0, 1.46 * data["RMSE"].max())
    ax.set_ylabel("Pooled-cycle RMSE")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax, "y")


def plot_fusion(ax: plt.Axes, frame: pd.DataFrame, scope: str) -> None:
    data = frame.loc[frame["evaluation_scope"].eq(scope)]
    x = np.arange(len(MODELS))
    offsets, width = grouped_offsets(2, 0.74)
    for target, offset in zip(TARGETS, offsets):
        rows = data.loc[data["target"].eq(target)].set_index("method").loc[list(MODELS)]
        ax.bar(x + offset, rows["RMSE"], width, color=COLORS[target],
               alpha=0.82, label=DATASET_LABELS[target], zorder=3)
    ax.set_xticks(x, MODEL_LABELS, rotation=32, ha="right", fontsize=6.1)
    ax.set_ylim(0, max(0.014, 1.32 * data["RMSE"].max()))
    ax.set_ylabel("RMSE")
    ax.set_title("Fusion comparison: " +
                 ("pooled cycles" if scope == "pooled_cycles" else "battery macro"))
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax, "y")


def effect_label(row: pd.Series) -> str:
    target = DATASET_LABELS[row["target"]]
    experiment = row["experiment"]
    if experiment == "feature_selection":
        name = "20 vs 39 features"
    elif experiment == "source_pretraining":
        name = "Pretrain vs scratch"
    elif row["comparator_method"] == "Equal_Weight_Selected_Branches":
        name = "Adaptive vs equal selected"
    elif row["comparator_method"] in BRANCHES:
        name = "Adaptive vs " + BRANCHES_SHORT[row["comparator_method"]]
    else:
        raise ValueError("Unrecognized bootstrap comparator")
    return f"{target}: {name}"


BRANCHES_SHORT = {
    "CNN_BiLSTM_Attention": "CNN",
    "MLP_Feature_Attention": "MLP",
    "HGB_Window_SOH": "HGB",
    "Ridge_Window_SOH": "Ridge",
}


def plot_effects(ax: plt.Axes, effects: pd.DataFrame) -> None:

    data = effects.loc[effects["evaluation_scope"].eq("pooled_cycles")].copy()
    order = ["feature_selection", "source_pretraining", "ensemble_fusion"]
    data["order"] = data["experiment"].map({name: i for i, name in enumerate(order)})
    data["target_order"] = data["target"].map({name: i for i, name in enumerate(TARGETS)})
    data = data.sort_values(["target_order", "order", "comparator_method"],
                            kind="stable").iloc[::-1].reset_index(drop=True)
    if len(data) != 8:
        raise ValueError("Expected eight pooled-cycle bootstrap comparisons")
    y = np.arange(len(data))
    scale = 1000
    low = scale * data["rmse_difference_ci95_lower"].to_numpy(float)
    high = scale * data["rmse_difference_ci95_upper"].to_numpy(float)
    center = scale * data["observed_rmse_difference"].to_numpy(float)
    for i, row in data.iterrows():
        color = COLORS[row["target"]]
        ax.plot([low[i], high[i]], [y[i], y[i]], lw=1.2, color=color)
        ax.scatter(center[i], y[i], s=19, color=color,
                   edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(0, lw=0.8, ls="--", color="#555555")
    ax.set_yticks(y, [effect_label(row) for _, row in data.iterrows()], fontsize=5.7)
    ax.set_xlabel(r"RMSE difference, comparator $-$ reference ($\times 10^{-3}$ SOH)")
    ax.set_title("Paired battery effects, pooled cycles (95% CI)")
    ax.legend(handles=[Line2D([], [], color=COLORS[t], marker="o", lw=1,
                               label=DATASET_LABELS[t]) for t in TARGETS],
              loc="upper right", frameon=False)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.12 * (right - left))
    style_axis(ax, "x")


def plot_weights(ax: plt.Axes, weights: pd.DataFrame) -> None:


    ncm = weights.loc[weights["target"].eq("TJU_NCM")
                      & weights["weight_scope"].eq("condition")]
    nca = weights.loc[weights["target"].eq("TJU_NCA")
                      & weights["weight_scope"].eq("global")]
    if set(ncm["condition_group"]) != {"25C_05C", "45C_05C"} or set(nca["condition_group"]) != {"ALL"}:
        raise ValueError("Stage 6 conditional weights do not match Stage 4 modes")
    groups = [
        ("NCM 25C/0.5C", ncm.loc[ncm["condition_group"].eq("25C_05C")]),
        ("NCM 45C/0.5C", ncm.loc[ncm["condition_group"].eq("45C_05C")]),
        ("NCA global", nca),
    ]
    for _, group in groups:
        if set(group["model"]) != set(BRANCHES) or not np.isclose(group["weight"].sum(), 1, atol=1e-6):
            raise ValueError("Incomplete or unnormalized fusion weights")
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for model in BRANCHES:
        values = np.array([g.set_index("model").loc[model, "weight"] for _, g in groups], float)
        ax.bar(x, values, bottom=bottom, width=0.56,
               color=BRANCH_COLORS[model], zorder=3)
        for i, value in enumerate(values):
            if value >= 0.09:
                ax.text(i, bottom[i] + value / 2, f"{value:.2f}",
                        ha="center", va="center", fontsize=6)
        bottom += values
    ax.set_xticks(x, [label for label, _ in groups], rotation=20, ha="right", fontsize=6.2)
    ax.set_ylim(0, 1.75)
    ax.set_ylabel("Validation-learned fusion weight")
    ax.set_title("Weights by target and operating condition")
    active_models = ("CNN_BiLSTM_Attention", "HGB_Window_SOH", "Ridge_Window_SOH")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=BRANCH_COLORS[model],
                                     label=BRANCHES_SHORT[model]) for model in active_models],
              loc="upper right", ncol=1, frameon=False)
    style_axis(ax, "y")


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = locate_run(args.base_path, args.run)
    data = load_data(run)
    output = args.output_dir or args.base_path / "FIG_FINAL" / "Figure6"
    fig, axes = plt.subplots(3, 2, figsize=(9.3, 9.7), constrained_layout=True)
    plot_control(axes[0, 0], data["feature"],
                 ("Selected20_Transfer_Adaptive", "All39_Transfer_Adaptive"),
                 ("Selected 20", "All 39"),
                 (COLORS["selected"], COLORS["light_blue"]), "Feature selection")
    plot_control(axes[0, 1], data["pretrain"],
                 ("With_Source_Pretraining", "Without_Source_Pretraining"),
                 ("With source pretraining", "Target-only scratch"),
                 (COLORS["TC"], COLORS["pink"]), "Source pretraining")
    plot_fusion(axes[1, 0], data["fusion"], "pooled_cycles")
    plot_fusion(axes[1, 1], data["fusion"], "battery_macro")
    plot_effects(axes[2, 0], data["effects"])
    plot_weights(axes[2, 1], data["weights"])
    for label, ax in zip("ABCDEF", axes.flat):
        panel_label(ax, label, x=-0.18, y=1.09)
    save_figure(fig, output, "Figure6_ablation")
    plt.close(fig)


if __name__ == "__main__":
    main()
