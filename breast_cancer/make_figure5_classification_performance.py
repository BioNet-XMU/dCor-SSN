#!/usr/bin/env python3
"""Rebuild manuscript Figure 5 from frozen classification CSV outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


DEFAULT_INPUT = BASE / "outputs"
DEFAULT_OUTPUT = DEFAULT_INPUT / "figures"
METHOD_ORDER = ["dCor", "Pearson", "Spearman"]
METHOD_LABELS = {
    "dCor": "dCor-SSN",
    "Pearson": "Pearson-SSN",
    "Spearman": "Spearman-SSN",
}
COLORS = {"dCor": "#D55E00", "Pearson": "#0072B2", "Spearman": "#009E73"}
MARKERS = {"dCor": "o", "Pearson": "s", "Spearman": "^"}
LINESTYLES = {"dCor": "-", "Pearson": "--", "Spearman": (0, (1.5, 1.8))}
SHORT_LABELS = {"dCor": "dCor", "Pearson": "Pearson", "Spearman": "Spearman"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(summary: pd.DataFrame, raw: pd.DataFrame) -> None:
    summary_required = {"analysis", "method", "q", "auroc_mean", "auroc_sd"}
    raw_required = {"analysis", "split", "method", "q", "auroc"}
    for name, frame, required in (
        ("summary", summary, summary_required),
        ("raw", raw, raw_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
    missing_methods = set(METHOD_ORDER).difference(summary["method"].unique())
    if missing_methods:
        raise ValueError(f"Summary table is missing methods: {sorted(missing_methods)}")


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.grid(axis="y", color="#EAEAEA", linewidth=0.6)
    axis.tick_params(axis="both", labelsize=7.5, width=0.8, length=3, color="#333333")
    axis.set_axisbelow(True)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16, 1.08, label, transform=axis.transAxes, ha="left", va="top",
        fontsize=9, fontweight="bold",
    )


def plot_metric(axis: plt.Axes, frame: pd.DataFrame, metric: str) -> None:
    for method in METHOD_ORDER:
        subset = frame.loc[frame["method"].eq(method)].sort_values("q")
        x = subset["q"].to_numpy(dtype=float)
        mean = subset[f"{metric}_mean"].to_numpy(dtype=float)
        sd = subset[f"{metric}_sd"].to_numpy(dtype=float)
        axis.plot(
            x, mean, color=COLORS[method], marker=MARKERS[method], markersize=3.4,
            markeredgewidth=0, linewidth=1.35, linestyle=LINESTYLES[method],
            label=METHOD_LABELS[method],
        )
        axis.fill_between(
            x, mean - sd, mean + sd, color=COLORS[method], alpha=0.12, linewidth=0
        )


def permutation_across_q(raw: pd.DataFrame) -> pd.DataFrame:
    permutation = raw.loc[raw["analysis"].eq("permuted_labels")].copy()
    if permutation.empty:
        raise ValueError("Raw result table contains no permuted_labels rows")
    return (
        permutation.groupby(["split", "method"], as_index=False)
        .agg(auroc_across_q=("auroc", "mean"), n_q=("q", "nunique"))
    )


def plot_permutation_box(axis: plt.Axes, frame: pd.DataFrame) -> None:
    method_frames = [
        frame.loc[frame["method"].eq(method)].sort_values("split")
        for method in METHOD_ORDER
    ]
    values = [subset["auroc_across_q"].to_numpy(dtype=float) for subset in method_frames]
    box = axis.boxplot(
        values, positions=range(1, 4), widths=0.55, patch_artist=True,
        showfliers=False, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 3.0, "markeredgewidth": 0.7},
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"linewidth": 0.8, "color": "#444444"},
        whiskerprops={"linewidth": 0.8, "color": "#444444"},
        capprops={"linewidth": 0.8, "color": "#444444"},
    )
    for patch, method in zip(box["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.36)
    for position, method, method_values, subset in zip(
        range(1, 4), METHOD_ORDER, values, method_frames
    ):
        split_ids = subset["split"].to_numpy(dtype=int)
        jitter = (((split_ids * 37) % 101) / 100.0 - 0.5) * 0.28
        axis.scatter(
            position + jitter,
            method_values,
            s=7,
            color=COLORS[method],
            alpha=0.24,
            edgecolors="none",
            rasterized=False,
            zorder=1,
        )
    axis.axhline(0.5, color="#AFAFAF", linewidth=0.8, linestyle=(0, (4, 4)))
    axis.set_xticks(range(1, 4))
    axis.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER])
    axis.set_xlabel("SSN representation")
    axis.set_ylabel("Mean AUROC across q")
    axis.set_ylim(0.30, 0.67)
    axis.set_yticks([0.30, 0.40, 0.50, 0.60])


def generate_figure(
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    include_baselines: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    summary_path = input_dir / "new_plslr_summary.csv"
    raw_path = input_dir / "new_plslr_results.csv"
    if not summary_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Figure 5 requires {summary_path} and {raw_path}")

    summary = pd.read_csv(summary_path)
    raw = pd.read_csv(raw_path)
    validate_inputs(summary, raw)
    real = summary.loc[summary["analysis"].eq("real_labels")].copy()
    permuted = permutation_across_q(raw)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.linewidth": 0.8,
    })
    figure, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.16, 0.84]}
    )
    axis_a, axis_b = axes

    plot_metric(axis_a, real, "auroc")
    handles, labels = axis_a.get_legend_handles_labels()
    if include_baselines:
        baselines = [("Age only", 0.949), ("Abundance only", 0.965)]
        baseline_handles = []
        for (name, value), style in zip(baselines, ((0, (6, 3)), (0, (2, 2)))):
            baseline_handles.append(axis_a.axhline(
                value, color="#4C4C4C", linewidth=0.95, linestyle=style,
                label=f"{name} ({value:.3f})",
            ))
        axis_a.legend(
            handles=baseline_handles, loc="lower right", ncol=2, frameon=False,
            handlelength=2.5, columnspacing=1.25,
        )
    axis_a.set(
        xlabel="q",
        ylabel="Outer-test AUROC",
        xlim=(0.01, 0.21),
        ylim=(0.75, 1.00),
    )
    q_ticks = np.arange(0.02, 0.201, 0.02)
    axis_a.set_xticks(q_ticks)
    axis_a.set_xticklabels([f"{value:.2f}" for value in q_ticks])
    axis_a.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])

    plot_permutation_box(axis_b, permuted)
    for axis, label in zip(axes, "AB"):
        add_panel_label(axis, label)
        style_axis(axis)

    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.015), handlelength=2.5, columnspacing=1.7,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.28, wspace=0.34)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Figure5_classification_performance"
    base = output_dir / (f"{stem}_with_baselines" if include_baselines else stem)
    figure.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)

    real_columns = [
        column
        for column in real.columns
        if column in {"analysis", "method", "q", "n"} or column.startswith("auroc_")
    ]
    real.loc[:, real_columns].to_csv(
        output_dir / "Figure5_A_real_label_plot_data.csv", index=False
    )
    permuted.to_csv(output_dir / "Figure5_B_permuted_label_plot_data.csv", index=False)
    manifest = {
        "figure": "Figure 5",
        "panel_a_source": summary_path.name,
        "panel_a_source_sha256": sha256(summary_path),
        "panel_b_source": raw_path.name,
        "panel_b_source_sha256": sha256(raw_path),
        "panel_b_aggregation": "arithmetic mean of the ten prespecified q-specific AUROCs within each permutation and method",
        "n_permutations_by_method": permuted.groupby("method")["split"].nunique().to_dict(),
        "panel_a_summary": "mean plus or minus one standard deviation across 100 outer splits",
        "layout": "two panels: real-label AUROC sensitivity and permutation negative control",
        "exports": ["png", "pdf", "svg", "tiff"],
    }
    if include_baselines:
        pd.DataFrame(baselines, columns=["model", "auroc"]).to_csv(
            output_dir / "Figure5_A_baselines.csv", index=False
        )
        manifest["panel_a_baselines"] = {
            "source": "Original manuscript and supporting information baseline values",
            "age_only_auroc": 0.949,
            "abundance_only_auroc": 0.965,
        }
    (output_dir / "Figure5_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Figure 5 written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    generate_figure(args.input_dir, args.output_dir, include_baselines=args.include_baselines)


if __name__ == "__main__":
    main()
