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
    COLORS, DATASET_LABELS, DEFAULT_BASE_PATH, feature_label,
    find_latest_accepted_run, panel_label, report_directory, save_figure,
    set_publication_style, style_axis,
)

ROOT = "CROSS_DOMAIN_RESULTS_V7_2_0_CONDITIONAL"
BUILD = "STAGE7_V7_2_0_CONDITIONAL_MATCHED_20261006"
STAGE4_BUILD = "STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006"
CASES = ("XJTU_HUST_to_TJU_NCM", "TJU_NCM_to_TJU_NCA")
TARGET = {CASES[0]: "TJU_NCM", CASES[1]: "TJU_NCA"}
CASE_TITLES = {CASES[0]: "XJTU + HUST → TJU-NCM", CASES[1]: "TJU-NCM → TJU-NCA"}
BUDGETS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
MODELS = (
    "CNN_BiLSTM_Attention", "MLP_Feature_Attention", "HGB_Window_SOH",
    "Ridge_Window_SOH", "Average_Ensemble", "Adaptive_Ensemble",
)
MODEL_LABELS = {
    "CNN_BiLSTM_Attention": "CNN-BiLSTM",
    "MLP_Feature_Attention": "MLP attention",
    "HGB_Window_SOH": "Window HGB",
    "Ridge_Window_SOH": "Window Ridge",
    "Average_Ensemble": "Equal average",
    "Adaptive_Ensemble": "Adaptive ensemble",
}
MODEL_COLORS = {
    "CNN_BiLSTM_Attention": COLORS["CNN_BiLSTM_Attention"],
    "MLP_Feature_Attention": COLORS["MLP_Feature_Attention"],
    "HGB_Window_SOH": COLORS["TC"],
    "Ridge_Window_SOH": COLORS["yellow"],
    "Average_Ensemble": COLORS["neutral"],
    "Adaptive_Ensemble": COLORS["Adaptive_Ensemble"],
}
TRAJECTORY_BUDGETS = (0.0, 0.1, 0.2, 0.6)
TRAJECTORY_COLORS = {0.0: COLORS["neutral"], 0.1: COLORS["light_blue"],
                     0.2: COLORS["TC"], 0.6: COLORS["Adaptive_Ensemble"]}
FILES = (
    "processing_config.json", "processing_summary.json",
    "learning_curve_summary.csv", "from_scratch_learning_curve_summary.csv",
    "transfer_vs_scratch_paired_effects.csv", "feature_domain_shift.csv",
    "all_final_test_predictions.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--run", type=Path, help="Stage 7 run, extracted report, or validation ZIP")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def locate_run(base: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None and explicit.is_file() and explicit.suffix.lower() == ".zip":
        return explicit
    return find_latest_accepted_run(base, ROOT, "7.2.0", list(FILES), explicit)


def report_bytes(run: Path, name: str) -> bytes:
    if run.is_file():
        with zipfile.ZipFile(run) as archive:
            matches = [n for n in archive.namelist()
                       if not n.endswith("/") and Path(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} in {run}; found {len(matches)}")
            return archive.read(matches[0])
    return (report_directory(run) / name).read_bytes()


def report_csv(run: Path, name: str, columns: tuple[str, ...],
               usecols: Optional[list[str]] = None) -> pd.DataFrame:

    if run.is_file():
        content = BytesIO(report_bytes(run, name))
        frame = pd.read_csv(content, usecols=usecols, low_memory=False,
                            dtype={"battery_id": str})
    else:
        frame = pd.read_csv(report_directory(run) / name, usecols=usecols,
                            low_memory=False, dtype={"battery_id": str})
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks columns: {sorted(missing)}")
    return frame


def check_learning_rows(frame: pd.DataFrame, budgets: tuple[float, ...],
                        models: tuple[str, ...], label: str) -> None:
    if frame.duplicated(["case", "model", "evaluation_scope",
                         "requested_target_label_fraction"]).any():
        raise ValueError(f"Duplicate {label} summary keys")
    for case in CASES:
        for model in models:
            for scope in ("pooled_cycles", "battery_macro"):
                rows = frame.loc[frame["case"].eq(case)
                                 & frame["model"].eq(model)
                                 & frame["evaluation_scope"].eq(scope)]
                if len(rows) != len(budgets) or not np.allclose(
                    rows["requested_target_label_fraction"].sort_values(), budgets
                ):
                    raise ValueError(f"{case}/{model}/{scope}: missing {label} budgets")
                if not np.isfinite(rows["RMSE_mean"].to_numpy(float)).all() or (rows["RMSE_mean"] <= 0).any():
                    raise ValueError(f"{case}/{model}/{scope}: invalid RMSE")


def load_data(run: Path) -> dict[str, pd.DataFrame]:
    config = json.loads(report_bytes(run, "processing_config.json").decode("utf-8-sig"))
    summary = json.loads(report_bytes(run, "processing_summary.json").decode("utf-8-sig"))
    if summary.get("validation_status") != "PASS" or summary.get("fatal_checks"):
        raise ValueError("Stage 7 validation did not PASS")
    if config.get("schema_version") != "7.2.0" or config.get("script_build") != BUILD:
        raise ValueError("Expected the Stage 7 V7.2.0 conditional build")
    if config.get("stage4_build") != STAGE4_BUILD:
        raise ValueError("Stage 7 lineage does not match conditional Stage 4")
    if config.get("target_test_truth_hidden_until_prediction_and_policy_frozen") is not True:
        raise ValueError("Test truth was not protected during prediction and policy selection")
    if config.get("paired_target_only_scratch_control") is not True:
        raise ValueError("Expected matched target-only scratch control")
    if (summary.get("zero_shot_points_are_independent_repetitions") is not False
            or config.get("zero_shot_source_fit_count_per_case") != 1):
        raise ValueError("Unexpected zero-shot repetition scheme")
    if not np.allclose(config.get("budgets", []), BUDGETS):
        raise ValueError("Unexpected Stage 7 target-label budgets")

    fields = ("case", "model", "arm", "evaluation_scope", "requested_target_label_fraction",
              "achieved_target_label_fraction", "RMSE_mean", "RMSE_std_across_seeds",
              "replicate_count")
    transfer = report_csv(run, "learning_curve_summary.csv", fields)
    scratch = report_csv(run, "from_scratch_learning_curve_summary.csv", fields)
    if not transfer["arm"].eq("transfer").all() or not scratch["arm"].eq("scratch").all():
        raise ValueError("Transfer and scratch arms are mixed")
    check_learning_rows(transfer, BUDGETS, MODELS, "transfer")
    check_learning_rows(scratch, BUDGETS[1:], MODELS, "scratch")
    zero = transfer.loc[transfer["requested_target_label_fraction"].eq(0)]
    if not zero["replicate_count"].eq(1).all() or not transfer.loc[
        transfer["requested_target_label_fraction"].gt(0), "replicate_count"].eq(3).all():
        raise ValueError("Zero-shot and nonzero repetition counts are inconsistent")

    shift = report_csv(run, "feature_domain_shift.csv",
                       ("case", "feature", "absolute_standardized_mean_shift"))
    if shift.duplicated(["case", "feature"]).any() or set(shift["case"]) != set(CASES):
        raise ValueError("Incomplete cross-domain feature shift")
    effects = report_csv(run, "transfer_vs_scratch_paired_effects.csv", (
        "case", "model", "requested_target_label_fraction",
        "observed_RMSE_improvement_mean_across_seeds",
        "RMSE_improvement_ci95_lower", "RMSE_improvement_ci95_upper", "comparison"))
    effects = effects.loc[effects["model"].eq("Adaptive_Ensemble")].copy()
    if len(effects) != 10 or effects.duplicated(["case", "requested_target_label_fraction"]).any():
        raise ValueError("Expected ten adaptive transfer-versus-scratch effects")
    if not effects["comparison"].eq("scratch_minus_transfer").all():
        raise ValueError("Bootstrap comparison has the wrong sign")
    if not np.isfinite(effects[["observed_RMSE_improvement_mean_across_seeds",
                                 "RMSE_improvement_ci95_lower",
                                 "RMSE_improvement_ci95_upper"]].to_numpy(float)).all():
        raise ValueError("Non-finite paired-bootstrap effects")

    prediction_cols = ["case", "battery_id", "cycle", "replicate_seed",
                       "requested_target_label_fraction", "y_true", "Adaptive_Ensemble"]
    predictions = report_csv(run, "all_final_test_predictions.csv",
                             tuple(prediction_cols), prediction_cols)
    if predictions.duplicated(["case", "battery_id", "cycle", "replicate_seed",
                              "requested_target_label_fraction"]).any():
        raise ValueError("Duplicate held-out prediction keys")
    if not np.isfinite(predictions[["y_true", "Adaptive_Ensemble"]].to_numpy(float)).all():
        raise ValueError("Non-finite held-out predictions")
    if set(predictions["case"]) != set(CASES):
        raise ValueError("Held-out trajectories are missing a migration case")
    return {"transfer": transfer, "scratch": scratch, "shift": shift,
            "effects": effects, "predictions": predictions}


def subset_curve(frame: pd.DataFrame, case: str, model: str, scope: str) -> pd.DataFrame:
    return (frame.loc[frame["case"].eq(case) & frame["model"].eq(model)
                      & frame["evaluation_scope"].eq(scope)]
            .sort_values("requested_target_label_fraction", kind="stable"))


def plot_adaptive(ax: plt.Axes, transfer: pd.DataFrame, scratch: pd.DataFrame,
                  case: str) -> None:
    target = TARGET[case]
    for arm, frame, color in (("Transferred", transfer, COLORS[target]),
                              ("Target-only scratch", scratch, COLORS["neutral"])):
        for scope, marker, linestyle, scope_label in (
            ("pooled_cycles", "o", "-", "pooled"),
            ("battery_macro", "s", "--", "battery macro"),
        ):
            rows = subset_curve(frame, case, "Adaptive_Ensemble", scope)
            x = 100 * rows["achieved_target_label_fraction"].to_numpy(float)
            y = rows["RMSE_mean"].to_numpy(float)
            ax.plot(x, y, color=color, marker=marker, ls=linestyle,
                    markersize=3.0, lw=1.2, label=f"{arm}, {scope_label}")
            repeat = rows["replicate_count"].to_numpy(int) > 1
            std = rows["RMSE_std_across_seeds"].to_numpy(float)
            if repeat.any():
                if not np.isfinite(std[repeat]).all():
                    raise ValueError("A repeated target-adaptation point lacks seed variation")
                ax.fill_between(x[repeat], np.maximum(y[repeat] - std[repeat], 1e-5),
                                y[repeat] + std[repeat], color=color, alpha=0.10, lw=0)
    ax.set_yscale("log")
    ax.set_xticks([0, 10, 20, 40, 60, 80])
    ax.set_xlabel("Acquired target batteries (% of adaptation pool)")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title(CASE_TITLES[case] + " | matched adaptation")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 1.45)
    ax.legend(loc="upper right", frameon=False, fontsize=6.1)
    style_axis(ax, "y")


def plot_models(ax: plt.Axes, transfer: pd.DataFrame, case: str) -> None:
    for model in MODELS:
        rows = subset_curve(transfer, case, model, "pooled_cycles")
        ax.plot(100 * rows["achieved_target_label_fraction"], rows["RMSE_mean"],
                color=MODEL_COLORS[model], marker="o", markersize=2.8,
                lw=1.65 if model == "Adaptive_Ensemble" else 1.0,
                alpha=1 if model == "Adaptive_Ensemble" else 0.83,
                label=MODEL_LABELS[model])
    ax.set_yscale("log")
    ax.set_xticks([0, 10, 20, 40, 60, 80])
    ax.set_xlabel("Acquired target batteries (%)")
    ax.set_ylabel("Pooled-cycle RMSE (log scale)")
    ax.set_title("Transferred model comparison: " + CASE_TITLES[case])
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 2.0)
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=6.0)
    style_axis(ax, "y")


def plot_shift(ax: plt.Axes, shift: pd.DataFrame) -> None:
    data = shift.pivot(index="feature", columns="case",
                       values="absolute_standardized_mean_shift").reindex(columns=list(CASES))
    if data.isna().any().any():
        raise ValueError("Feature shift is not available for both cases")
    data = data.assign(mean=data[list(CASES)].mean(axis=1)).nlargest(10, "mean").sort_values("mean")
    y = np.arange(len(data))
    for index, case in enumerate(CASES):
        ax.barh(y + (index - 0.5) * 0.34, data[case], 0.34,
                color=COLORS[TARGET[case]], label=CASE_TITLES[case],
                alpha=0.82, zorder=3)
    ax.set_yticks(y, [feature_label(name) for name in data.index])
    ax.set_xlabel("Absolute standardized mean shift")
    ax.set_title("Source-to-target feature mean shifts")
    ax.legend(loc="lower right", frameon=False, fontsize=6.2)
    style_axis(ax, "x")


def plot_transfer_effect(ax: plt.Axes, effects: pd.DataFrame) -> None:

    rows = []
    for case in CASES:
        rows.extend(effects.loc[effects["case"].eq(case)]
                    .sort_values("requested_target_label_fraction", ascending=False)
                    .to_dict("records"))
    data = pd.DataFrame(rows)
    if len(data) != 10:
        raise ValueError("Missing adaptive transfer-versus-scratch paired effects")
    y = np.arange(len(data))
    for i, row in data.iterrows():
        color = COLORS[TARGET[row["case"]]]
        ax.plot([row["RMSE_improvement_ci95_lower"],
                 row["RMSE_improvement_ci95_upper"]], [y[i], y[i]],
                color=color, lw=1.1, zorder=2)
        ax.scatter(row["observed_RMSE_improvement_mean_across_seeds"], y[i],
                   color=color, s=19, edgecolor="white", lw=0.4, zorder=3)
    labels = [f"{DATASET_LABELS[TARGET[row['case']]]}, "
              f"{100 * row['requested_target_label_fraction']:.0f}%"
              for _, row in data.iterrows()]
    ax.set_yticks(y, labels)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Scratch RMSE − transfer RMSE")
    ax.set_title("Pretraining benefit at matched budgets (95% CI)")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.30 * (hi - lo))
    ax.legend(handles=[Line2D([0], [0], color=COLORS[TARGET[case]], marker="o",
                              lw=1, label=DATASET_LABELS[TARGET[case]]) for case in CASES],
              loc="upper right", ncol=1, frameon=False)
    style_axis(ax, "x")


def representative_battery(predictions: pd.DataFrame, case: str) -> str:
    data = predictions.loc[predictions["case"].eq(case)
                           & np.isclose(predictions["requested_target_label_fraction"], 0.2)]
    scores = []
    for battery_id, group in data.groupby("battery_id", sort=True):
        rmse = [float(np.sqrt(np.mean((seed["Adaptive_Ensemble"] - seed["y_true"]) ** 2)))
                for _, seed in group.groupby("replicate_seed", sort=True)]
        if len(rmse) != 3:
            raise ValueError("Representative battery lacks three 20% replicates")
        scores.append((battery_id, float(np.mean(rmse))))
    if not scores:
        raise ValueError(f"No 20% final-test predictions for {case}")
    scores.sort(key=lambda pair: (pair[1], str(pair[0])))
    return str(scores[len(scores) // 2][0])


def plot_trajectory(ax: plt.Axes, predictions: pd.DataFrame, case: str) -> None:
    battery_id = representative_battery(predictions, case)
    data = predictions.loc[predictions["case"].eq(case)
                           & predictions["battery_id"].eq(battery_id)].copy()
    true_range = data.groupby("cycle")["y_true"].agg(lambda values: values.max() - values.min())
    if (true_range > 1e-9).any():
        raise ValueError("Reference SOH changes across budgets or seeds")
    measured = data.groupby("cycle", as_index=False)["y_true"].first().sort_values("cycle")
    ax.plot(measured["cycle"], measured["y_true"], color=COLORS["black"],
            lw=1.65, label="Measured", zorder=5)
    extrema = [measured["y_true"].min(), measured["y_true"].max()]
    for budget in TRAJECTORY_BUDGETS:
        selected = data.loc[np.isclose(data["requested_target_label_fraction"], budget)]
        if selected.empty:
            raise ValueError(f"Missing trajectory for {case} at {budget}")
        curve = (selected.groupby("cycle", as_index=False)["Adaptive_Ensemble"]
                 .mean().sort_values("cycle"))
        extrema.extend([curve["Adaptive_Ensemble"].min(), curve["Adaptive_Ensemble"].max()])
        ax.plot(curve["cycle"], curve["Adaptive_Ensemble"],
                color=TRAJECTORY_COLORS[budget], ls="--" if budget == 0 else "-",
                lw=1.05 if budget == 0 else 1.25,
                label=f"Transfer, {100 * budget:.0f}%")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SOH")
    ax.set_ylim(min(0, min(extrema) - 0.04), max(1.02, max(extrema) + 0.04))
    ax.set_title(f"Representative test battery: {DATASET_LABELS[TARGET[case]]}")
    ax.legend(loc="lower right", ncol=1, frameon=False, fontsize=6.0)
    style_axis(ax)


def main() -> None:
    args = parse_args()
    set_publication_style()
    run = locate_run(args.base_path, args.run)
    data = load_data(run)
    output_dir = args.output_dir or args.base_path / "FIG_FINAL" / "Figure7"
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 12.0), constrained_layout=True)
    for j, case in enumerate(CASES):
        plot_adaptive(axes[0, j], data["transfer"], data["scratch"], case)
        plot_models(axes[1, j], data["transfer"], case)
        plot_trajectory(axes[3, j], data["predictions"], case)
    plot_shift(axes[2, 0], data["shift"])
    plot_transfer_effect(axes[2, 1], data["effects"])
    for label, ax in zip("ABCDEFGH", axes.flat):
        panel_label(ax, label, x=-0.20, y=1.12)
    save_figure(fig, output_dir, "Figure7_few_shot_generalization")
    plt.close(fig)


if __name__ == "__main__":
    main()
