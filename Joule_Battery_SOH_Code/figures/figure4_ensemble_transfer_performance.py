from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting_common import (
    COLORS,
    DATASET_LABELS,
    DEFAULT_BASE_PATH,
    find_latest_accepted_run,
    grouped_offsets,
    panel_label,
    report_directory,
    save_figure,
    set_publication_style,
    style_axis,
)


STAGE4_ROOT_NAME = "SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
TARGETS = ["TJU_NCM", "TJU_NCA"]
NEURAL_MODELS = [
    "CNN_BiLSTM_Attention",
    "MLP_Feature_Attention",
]
BASE_MODELS = [
    *NEURAL_MODELS,
    "HGB_Window_SOH",
    "Ridge_Window_SOH",
]
PERFORMANCE_MODELS = BASE_MODELS + ["Adaptive_Ensemble"]
MODEL_LABELS = {
    "CNN_BiLSTM_Attention": "CNN-BiLSTM-attention",
    "MLP_Feature_Attention": "MLP-feature-attention",
    "HGB_Window_SOH": "Window HGB",
    "Ridge_Window_SOH": "Window Ridge",
    "Adaptive_Ensemble": "Adaptive ensemble",
}
MODEL_COLORS = {
    **COLORS,
    "HGB_Window_SOH": COLORS["TC"],
    "Ridge_Window_SOH": COLORS["yellow"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, default=None, help="Explicit Stage 4 run or extracted report")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def locate_run(base_path: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None and explicit.is_file() and explicit.suffix.lower() == ".zip":
        print("Accepted input report:", explicit)
        return explicit
    return find_latest_accepted_run(
        base_path,
        STAGE4_ROOT_NAME,
        "4.0.2",
        [
            "training_history.csv",
            "target_test_metrics.csv",
            "ensemble_weights.csv",
            "TJU_NCM_test_predictions.csv",
            "TJU_NCA_test_predictions.csv",
        ],
        explicit,
    )


def report_json(run: Path, filename: str) -> dict:
    if run.is_file():
        with zipfile.ZipFile(run) as archive:
            return json.loads(archive.read(filename).decode("utf-8-sig"))
    return json.loads((report_directory(run) / filename).read_text(encoding="utf-8-sig"))


def report_csv(run: Path, filename: str, required: list[str]) -> pd.DataFrame:
    if run.is_file():
        with zipfile.ZipFile(run) as archive:
            with archive.open(filename) as content:
                frame = pd.read_csv(content, dtype={"battery_id": str})
    else:
        frame = pd.read_csv(report_directory(run) / filename, dtype={"battery_id": str})
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")
    return frame


def load_data(run: Path):
    summary = report_json(run, "processing_summary.json")
    config = report_json(run, "processing_config.json")
    if summary.get("validation_status") != "PASS":
        raise ValueError("Stage 4 validation did not PASS")
    if (summary.get("script_build") != STAGE4_BUILD
            or config.get("script_build") != STAGE4_BUILD):
        raise ValueError("Figure 4 requires Stage 4 conditional-contribution R2")
    if config.get("test_labels_used_for_ensemble_weights") is not False:
        raise ValueError("Stage 4 does not establish validation-only fusion weights")

    history = report_csv(run, "training_history.csv",
        ["phase", "model", "epoch", "train_mse", "validation_battery_domain_balanced_mse"],
    )
    metrics = report_csv(run, "target_test_metrics.csv",
        ["target", "model", "evaluation_scope", "MAE", "RMSE", "MAPE_percent", "R2"],
    )
    predictions = {}
    required_prediction_columns = [
        "battery_id",
        "cycle",
        "y_true",
        *PERFORMANCE_MODELS,
    ]
    for target in TARGETS:
        frame = report_csv(run, f"{target}_test_predictions.csv", required_prediction_columns)
        if frame.duplicated(["battery_id", "cycle"]).any():
            raise ValueError(f"{target} contains duplicate test prediction keys")
        if not np.isfinite(frame[["y_true", *PERFORMANCE_MODELS]].to_numpy(float)).all():
            raise ValueError(f"{target} contains non-finite values")
        frame = frame.sort_values(["battery_id", "cycle"], kind="stable")
        pooled = metrics.loc[metrics["target"].eq(target)
                             & metrics["evaluation_scope"].eq("pooled_cycles")]
        if set(PERFORMANCE_MODELS) - set(pooled["model"]):
            raise ValueError(f"{target} final-test metrics are incomplete")
        for model in PERFORMANCE_MODELS:
            row = pooled.loc[pooled["model"].eq(model)]
            if len(row) != 1:
                raise ValueError(f"Expected one pooled metric for {target}/{model}")
            actual = np.sqrt(np.mean((frame[model].to_numpy(float)
                                      - frame["y_true"].to_numpy(float)) ** 2))
            if not np.isclose(actual, float(row.iloc[0]["RMSE"]), rtol=0, atol=2e-7):
                raise ValueError(f"{target}/{model} predictions and RMSE report disagree")
        predictions[target] = frame
    return history, metrics, predictions


def plot_convergence(ax: plt.Axes, history: pd.DataFrame, phase: str, title: str) -> None:
    subset = history.loc[history["phase"].eq(phase)]
    if subset.empty:
        raise ValueError(f"No training history found for phase {phase}")
    for model in NEURAL_MODELS:
        data = subset.loc[subset["model"].eq(model)].sort_values("epoch")
        if data.empty:
            continue
        train_rmse = np.sqrt(np.clip(data["train_mse"].to_numpy(float), 0, None))
        valid_rmse = np.sqrt(
            np.clip(data["validation_battery_domain_balanced_mse"].to_numpy(float), 0, None)
        )
        ax.plot(data["epoch"], train_rmse, color=MODEL_COLORS[model], alpha=0.35, ls="--", lw=0.85)
        ax.plot(data["epoch"], valid_rmse, color=MODEL_COLORS[model], lw=1.25)
        best = int(np.nanargmin(valid_rmse))
        ax.scatter(data["epoch"].iloc[best], valid_rmse[best], s=15, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.4, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], lw=1.4, label=MODEL_LABELS[model])
        for model in NEURAL_MODELS
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#555555", lw=1.1, ls="-", label="Validation"),
            Line2D([0], [0], color="#555555", lw=1.1, ls="--", label="Training"),
        ]
    )
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 2.2)
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False)
    style_axis(ax, "y")


def plot_representative_prediction(ax: plt.Axes, frame: pd.DataFrame, target: str) -> None:


    lengths = frame.groupby("battery_id", sort=True).size()
    typical_length = float(lengths.median())
    battery_id = min(lengths.index, key=lambda key: (
        abs(int(lengths.loc[key]) - typical_length), str(key)))
    data = frame.loc[frame["battery_id"].eq(battery_id)].sort_values("cycle")
    ax.plot(data["cycle"], data["y_true"], color=COLORS["black"], lw=1.7, label="Measured", zorder=5)
    for model in BASE_MODELS:
        ax.plot(data["cycle"], data[model], color=MODEL_COLORS[model], lw=0.8, alpha=0.72, label=MODEL_LABELS[model])
    ax.plot(
        data["cycle"],
        data["Adaptive_Ensemble"],
        color=MODEL_COLORS["Adaptive_Ensemble"],
        lw=1.55,
        label=MODEL_LABELS["Adaptive_Ensemble"],
        zorder=4,
    )
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SOH")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{DATASET_LABELS[target]} representative test battery")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    style_axis(ax)


def plot_parity(
    ax: plt.Axes,
    predictions: Dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
) -> None:
    lower, upper = np.inf, -np.inf
    for target in TARGETS:
        frame = predictions[target]
        x = frame["y_true"].to_numpy(float)
        y = frame["Adaptive_Ensemble"].to_numpy(float)
        lower = min(lower, float(x.min()), float(y.min()))
        upper = max(upper, float(x.max()), float(y.max()))
        ax.scatter(
            x,
            y,
            s=5,
            alpha=0.25,
            linewidths=0,
            color=COLORS[target],
            rasterized=True,
            label=DATASET_LABELS[target],
        )
    ax.plot([lower, upper], [lower, upper], color="#333333", ls="--", lw=0.8)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    text = []
    pooled = metrics.loc[
        metrics["model"].eq("Adaptive_Ensemble")
        & metrics["evaluation_scope"].eq("pooled_cycles")
    ]
    for target in TARGETS:
        row = pooled.loc[pooled["target"].eq(target)].iloc[0]
        text.append(f"{DATASET_LABELS[target]}: $R^2$={row['R2']:.3f}")
    ax.text(0.04, 0.96, "\n".join(text), transform=ax.transAxes, va="top", fontsize=6.3)
    ax.set_xlabel("Measured SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title("Adaptive-ensemble parity")
    ax.legend(loc="lower right")
    style_axis(ax)


def plot_model_rmse(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    pooled = metrics.loc[metrics["evaluation_scope"].eq("pooled_cycles")]
    x = np.arange(len(PERFORMANCE_MODELS))
    offsets, width = grouped_offsets(len(TARGETS), 0.72)
    for offset, target in zip(offsets, TARGETS):
        rows = pooled.loc[pooled["target"].eq(target)].set_index("model").loc[PERFORMANCE_MODELS]
        bars = ax.bar(
            x + offset,
            rows["RMSE"],
            width,
            color=COLORS[target],
            alpha=0.82,
            label=DATASET_LABELS[target],
            zorder=3,
        )
        for bar, r2 in zip(bars, rows["R2"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{r2:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.1,
                rotation=0,
            )
    ax.set_xticks(
        x,
        [MODEL_LABELS[name].replace("-attention", "") for name in PERFORMANCE_MODELS],
        rotation=32,
        ha="right",
        fontsize=5.6,
    )
    maximum = float(
        pooled.loc[pooled["model"].isin(PERFORMANCE_MODELS), "RMSE"].max()
    )
    ax.set_ylim(0, max(0.014, maximum * 1.38))
    ax.set_ylabel("RMSE")
    ax.set_title("Same-split final-test model comparison")
    legend_handles = [
        Patch(facecolor=COLORS[target], alpha=0.82, label=DATASET_LABELS[target])
        for target in TARGETS
    ]
    legend_handles.append(
        Line2D([], [], color="none", label="Bar labels: $R^2$")
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    style_axis(ax, "y")


def per_battery_rmse(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = []
    for _, group in frame.groupby("battery_id", sort=True):
        error = group[column].to_numpy(float) - group["y_true"].to_numpy(float)
        values.append(float(np.sqrt(np.mean(error**2))))
    return np.asarray(values)


def plot_battery_errors(
    ax: plt.Axes, predictions: Dict[str, pd.DataFrame]
) -> None:
    values = [per_battery_rmse(predictions[target], "Adaptive_Ensemble") for target in TARGETS]
    box = ax.boxplot(
        values,
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.0},
    )
    rng = np.random.default_rng(2026)
    for position, target, data, patch in zip([1, 2], TARGETS, values, box["boxes"]):
        patch.set_facecolor(COLORS[target])
        patch.set_alpha(0.45)
        patch.set_edgecolor(COLORS[target])
        jitter = rng.normal(position, 0.035, size=len(data))
        ax.scatter(jitter, data, s=12, color=COLORS[target], alpha=0.75, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks([1, 2], [DATASET_LABELS[target] for target in TARGETS])
    ax.set_ylabel("Battery-level RMSE")
    ax.set_title("Error variation across held-out batteries")
    style_axis(ax, "y")


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = locate_run(args.base_path, args.run)
    output_dir = args.output_dir or args.base_path / "FIG_FINAL" / "Figure4"
    history, metrics, predictions = load_data(run)

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 10.2), constrained_layout=True)
    plot_convergence(axes[0, 0], history, "source_pretraining", "Source-domain pretraining")
    plot_convergence(axes[0, 1], history, "target_finetuning_TJU_NCM", "TJU-NCM target adaptation")
    plot_convergence(axes[1, 0], history, "target_finetuning_TJU_NCA", "TJU-NCA target adaptation")
    plot_representative_prediction(axes[1, 1], predictions["TJU_NCM"], "TJU_NCM")
    plot_representative_prediction(axes[2, 0], predictions["TJU_NCA"], "TJU_NCA")
    plot_parity(axes[2, 1], predictions, metrics)
    plot_model_rmse(axes[3, 0], metrics)
    plot_battery_errors(axes[3, 1], predictions)
    for label, ax in zip("ABCDEFGH", axes.flat):
        panel_label(ax, label)

    save_figure(fig, output_dir, "Figure4_ensemble_transfer_performance")
    plt.close(fig)


if __name__ == "__main__":
    main()
