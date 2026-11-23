"""Final five-class sample classification with one shared linear backbone.

All samples share five linear M1--M2 edges.  Each class additionally activates
five matched edges in exactly one of the Linear, Quadratic, Absolute, Sine or
Logarithmic modules.  Five aggregated heterogeneous-edge ranks are classified
with stratified five-fold cross-validation.

This file is the single, self-contained entry point of the heterogeneous
simulation. It contains, in order:

1. the five relationship transformations and the three matched SWEET network
   estimators (dCor, Pearson, Spearman);
2. data generation, feature construction and cross-validated classification;
3. the supplementary diagnostic figures;
4. the manuscript Figure 3;
5. result tables and the command-line entry point.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "outputs" / "section_b_original"
MAIN_FIGURE_DIR = OUTPUT_DIR / "figures"
SUPPLEMENTARY_FIGURE_DIR = OUTPUT_DIR / "supplementary_figures"
DATA_TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE3_STEM = "Figure3_heterogeneous_sample_classification"

METHODS = ("dCor-SWEET", "Pearson-SWEET", "Spearman-SWEET")
CLASS_NAMES = ("Linear", "Quadratic", "Absolute", "Sine", "Logarithmic")
COLORS = {
    "dCor-SWEET": "#D55E00",
    "Pearson-SWEET": "#0072B2",
    "Spearman-SWEET": "#009E73",
}
MARKERS = {"dCor-SWEET": "o", "Pearson-SWEET": "s", "Spearman-SWEET": "^"}
LINESTYLES = {
    "dCor-SWEET": "-",
    "Pearson-SWEET": (0, (5, 2.5)),
    "Spearman-SWEET": (0, (1.5, 1.8)),
}
DISPLAY_NAMES = {
    "dCor-SWEET": "dCor-SSN",
    "Pearson-SWEET": "Pearson-SSN",
    "Spearman-SWEET": "Spearman-SSN",
}
FIGURE3_SIGMAS = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), dtype=float)
RANDOM_BASELINE = 0.20
RANDOM_COLOR = "#7A7A7A"


@dataclass(frozen=True)
class Config:
    module_size: int = int(os.environ.get("SLCV_MODULE_SIZE", 10))
    n_shared: int = int(os.environ.get("SLCV_SHARED", 5))
    n_per_class: int = int(os.environ.get("SLCV_N_PER_CLASS", 50))
    n_repeats: int = int(os.environ.get("SLCV_R", 100))
    sigma_grid: tuple[float, ...] = tuple(
        float(value.strip())
        for value in os.environ.get("SLCV_SIGMA_GRID", "0,0.2,0.4,0.6,0.8,1.0").split(",")
        if value.strip()
    )
    linear_coef: float = float(os.environ.get("SLCV_LINEAR_COEF", 0.7))
    linear_intercept: float = float(os.environ.get("SLCV_LINEAR_INTERCEPT", 1.0))
    sine_frequency: float = float(os.environ.get("SLCV_SINE_FREQ", 2.0))
    log_shift: float = float(os.environ.get("SLCV_LOG_SHIFT", 1.0))
    sweet_k: float = float(os.environ.get("SLCV_K", 0.1))
    sweet_eps: float = float(os.environ.get("SLCV_EPS", 0.01))
    base_seed: int = int(os.environ.get("SLCV_BASE_SEED", 20260928))

    @property
    def n_samples(self) -> int:
        return self.n_per_class * len(CLASS_NAMES)

    @property
    def n_variables(self) -> int:
        return self.module_size * 7


CFG = Config()


# ---------------------------------------------------------------------------
# 1. Relationship transformations and matched SWEET network estimators
# ---------------------------------------------------------------------------

DependencyFunction = Callable[[np.ndarray], np.ndarray]


def transform(source: np.ndarray, relation: int, config: Config) -> np.ndarray:
    """Apply one of the five relationships in the 70-variable design."""
    if relation == 0:
        return config.linear_coef * source + config.linear_intercept
    if relation == 1:
        return source**2
    if relation == 2:
        return np.abs(source)
    if relation == 3:
        return np.sin(config.sine_frequency * source)
    if relation == 4:
        return np.log1p(np.maximum(0.0, source + config.log_shift))
    raise ValueError(f"unknown relation index {relation}")


def _pearson_stats(data: np.ndarray):
    return data.shape[0], data.sum(0), (data * data).sum(0), data.T @ data


def _pearson_from_stats(n, sx, sxx, sxy, triu) -> np.ndarray:
    mean = sx / n
    covariance = sxy - n * np.outer(mean, mean)
    variance = sxx - n * mean * mean
    denominator = np.sqrt(np.outer(variance, variance))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0,
    )
    return np.nan_to_num(correlation)[triu]


def _pearson_full(data: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(np.corrcoef(data, rowvar=False), dtype=float))


def _rank_columns(data: np.ndarray) -> np.ndarray:
    return pd.DataFrame(data).rank(axis=0, method="average").to_numpy(float)


def _spearman_full(data: np.ndarray) -> np.ndarray:
    return _pearson_full(_rank_columns(data))


def _centered_inner(raw, rows, total, n):
    n_float = float(n)
    centered = raw - (2.0 / n_float) * rows + np.outer(total, total) / (n_float**2)
    return 0.5 * (centered + centered.T)


def _dcor_from_inner(inner: np.ndarray) -> np.ndarray:
    diagonal = np.clip(np.diag(inner), 0.0, None)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    squared = np.divide(
        inner,
        denominator,
        out=np.zeros_like(inner),
        where=denominator > 0,
    )
    result = np.sqrt(np.clip(squared, 0.0, 1.0))
    return 0.5 * (result + result.T)


def _dcor_full(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, float).T
    n_samples = values.shape[1]
    distances = np.abs(values[:, :, None] - values[:, None, :])
    centered = (
        distances
        - distances.mean(2, keepdims=True)
        - distances.mean(1, keepdims=True)
        + distances.mean((1, 2))[:, None, None]
    )
    flattened = centered.reshape(centered.shape[0], n_samples * n_samples)
    return _dcor_from_inner(flattened @ flattened.T)


def _sample_weights(
    data: np.ndarray,
    dependency: DependencyFunction,
    eps: float,
) -> np.ndarray:
    n_samples = len(data)
    matrix = np.asarray(dependency(data.T), float)
    mean_dependency = (matrix.sum(1) - np.diag(matrix)) / (n_samples - 1)
    return (
        mean_dependency - mean_dependency.min() + eps
    ) / (mean_dependency.max() - mean_dependency.min() + eps)


def pearson_sweet(data, triu, config: Config):
    """Return Pearson-based SWEET edge contributions and sample weights."""
    n_samples, sums, squared_sums, cross_products = _pearson_stats(data)
    full = _pearson_from_stats(
        n_samples, sums, squared_sums, cross_products, triu
    )
    weights = _sample_weights(data, _pearson_full, config.sweet_eps)
    networks = np.empty((n_samples, len(full)))
    for sample_index, row in enumerate(data):
        leave_one_out = _pearson_from_stats(
            n_samples - 1,
            sums - row,
            squared_sums - row * row,
            cross_products - np.outer(row, row),
            triu,
        )
        networks[sample_index] = leave_one_out + (
            config.sweet_k
            * weights[sample_index]
            * (n_samples - 1)
            * (full - leave_one_out)
        )
    return networks, weights


def spearman_sweet(data, triu, config: Config):
    """Return Spearman-based SWEET edge contributions and sample weights."""
    n_samples = len(data)
    full = _pearson_from_stats(*_pearson_stats(_rank_columns(data)), triu)
    weights = _sample_weights(data, _spearman_full, config.sweet_eps)
    networks = np.empty((n_samples, len(full)))
    keep = np.ones(n_samples, bool)
    for sample_index in range(n_samples):
        keep[sample_index] = False
        leave_one_out = _pearson_from_stats(
            *_pearson_stats(_rank_columns(data[keep])), triu
        )
        networks[sample_index] = leave_one_out + (
            config.sweet_k
            * weights[sample_index]
            * (n_samples - 1)
            * (full - leave_one_out)
        )
        keep[sample_index] = True
    return networks, weights


def dcor_sweet(data, triu, config: Config):
    """Return distance-correlation SWEET edge contributions and weights."""
    values = np.asarray(data, float).T
    n_variables, n_samples = values.shape
    distances = np.abs(values[:, :, None] - values[:, None, :])
    row_sums = distances.sum(2)
    totals = row_sums.sum(1)
    flattened = distances.reshape(n_variables, n_samples * n_samples)
    raw_inner = flattened @ flattened.T
    row_inner = row_sums @ row_sums.T
    full = _dcor_from_inner(
        _centered_inner(raw_inner, row_inner, totals, n_samples)
    )[triu]
    weights = _sample_weights(data, _dcor_full, config.sweet_eps)
    networks = np.empty((n_samples, len(full)))
    for sample_index in range(n_samples):
        removed = distances[:, sample_index, :]
        raw_leave_one_out = raw_inner - 2.0 * (removed @ removed.T)
        total_leave_one_out = totals - 2.0 * row_sums[:, sample_index]
        reduced_row_sums = row_sums - distances[:, :, sample_index]
        reduced_row_sums[:, sample_index] = 0.0
        leave_one_out = _dcor_from_inner(
            _centered_inner(
                raw_leave_one_out,
                reduced_row_sums @ reduced_row_sums.T,
                total_leave_one_out,
                n_samples - 1,
            )
        )[triu]
        networks[sample_index] = leave_one_out + (
            config.sweet_k
            * weights[sample_index]
            * (n_samples - 1)
            * (full - leave_one_out)
        )
    return networks, weights


SWEET_ESTIMATORS = {
    "dCor-SWEET": dcor_sweet,
    "Pearson-SWEET": pearson_sweet,
    "Spearman-SWEET": spearman_sweet,
}


# ---------------------------------------------------------------------------
# 2. Data generation, features and cross-validated classification
# ---------------------------------------------------------------------------

def generate_dataset(seed: int, sigma: float, config: Config = CFG):
    if config.module_size != 10 or config.n_shared != 5:
        raise ValueError("this trial is defined for 10 variables per module and 5 shared positions")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(CLASS_NAMES)), config.n_per_class)
    base = rng.normal(size=(len(labels), config.module_size))
    modules = [base]
    for relation in range(len(CLASS_NAMES)):
        independent = rng.normal(size=base.shape)
        source = independent.copy()
        # The only common backbone is five linear M1--M2 edges.
        if relation == 0:
            source[:, : config.n_shared] = base[:, : config.n_shared]
        heterogeneous = slice(config.n_shared, config.module_size)
        source[:, heterogeneous] = np.where(
            (labels == relation)[:, None],
            base[:, heterogeneous],
            independent[:, heterogeneous],
        )
        modules.append(transform(source, relation, config))
    modules.append(rng.normal(size=base.shape))
    clean = np.column_stack(modules)
    column_sd = clean.std(axis=0, ddof=1)
    observed = clean + rng.normal(size=clean.shape) * sigma * column_sd[None, :]
    observed = np.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
    order = rng.permutation(len(labels))
    return observed[order], labels[order]


def edge_design(config: Config = CFG):
    triu = np.triu_indices(config.n_variables, k=1)
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(*triu))}
    shared = np.asarray(
        [lookup[(j, config.module_size + j)] for j in range(config.n_shared)], dtype=int
    )
    heterogeneous = []
    for relation in range(len(CLASS_NAMES)):
        target_start = (relation + 1) * config.module_size
        heterogeneous.append(
            np.asarray(
                [
                    lookup[(j, target_start + j)]
                    for j in range(config.n_shared, config.module_size)
                ],
                dtype=int,
            )
        )
    return triu, shared, heterogeneous


def aggregate_rank_features(networks: np.ndarray, edge_groups: list[np.ndarray]):
    ranks = rankdata(-np.abs(networks), axis=1, method="average")
    return np.column_stack([ranks[:, group].mean(axis=1) for group in edge_groups])


def classifier():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def cross_validated_predictions(features, labels, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(classifier(), features, labels, cv=folds, method="predict")


def run(config: Config = CFG):
    triu, shared_edges, heterogeneous_edges = edge_design(config)
    metric_rows, recall_rows, confusion_rows = [], [], []
    for sigma_index, sigma in enumerate(config.sigma_grid):
        for repeat in range(config.n_repeats):
            seed = config.base_seed + sigma_index * 100_000 + repeat
            print(f"sigma={sigma:g} repeat {repeat + 1}/{config.n_repeats} seed={seed}", flush=True)
            data, labels = generate_dataset(seed, sigma, config)
            for method in METHODS:
                networks, _ = SWEET_ESTIMATORS[method](data, triu, config)
                features = aggregate_rank_features(networks, heterogeneous_edges)
                shared_feature = aggregate_rank_features(networks, [shared_edges])
                predictions = cross_validated_predictions(features, labels, seed)
                shared_predictions = cross_validated_predictions(shared_feature, labels, seed)
                rng = np.random.default_rng(seed + 50_000_000)
                permuted_labels = labels[rng.permutation(len(labels))]
                permuted_predictions = cross_validated_predictions(
                    features, permuted_labels, seed + 1
                )
                metric_rows.append(
                    {
                        "sigma": sigma,
                        "repeat": repeat,
                        "seed": seed,
                        "method": method,
                        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
                        "macro_f1": f1_score(labels, predictions, average="macro"),
                        "shared_only_balanced_accuracy": balanced_accuracy_score(
                            labels, shared_predictions
                        ),
                        "permuted_label_balanced_accuracy": balanced_accuracy_score(
                            permuted_labels, permuted_predictions
                        ),
                    }
                )
                for class_id, class_name in enumerate(CLASS_NAMES):
                    mask = labels == class_id
                    recall_rows.append(
                        {
                            "sigma": sigma,
                            "repeat": repeat,
                            "method": method,
                            "class": class_name,
                            "recall": float((predictions[mask] == class_id).mean()),
                        }
                    )
                matrix = confusion_matrix(labels, predictions, labels=np.arange(5))
                for true_id, true_name in enumerate(CLASS_NAMES):
                    for predicted_id, predicted_name in enumerate(CLASS_NAMES):
                        confusion_rows.append(
                            {
                                "sigma": sigma,
                                "repeat": repeat,
                                "method": method,
                                "true_class": true_name,
                                "predicted_class": predicted_name,
                                "n_samples": int(matrix[true_id, predicted_id]),
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(recall_rows), pd.DataFrame(confusion_rows)


def summarize(metrics: pd.DataFrame, recalls: pd.DataFrame):
    summary = (
        metrics.groupby(["sigma", "method"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            shared_only_mean=("shared_only_balanced_accuracy", "mean"),
            permuted_label_mean=("permuted_label_balanced_accuracy", "mean"),
            shared_only_sd=("shared_only_balanced_accuracy", "std"),
            permuted_label_sd=("permuted_label_balanced_accuracy", "std"),
            n_repeats=("repeat", "size"),
        )
    )
    summary["balanced_accuracy_sem"] = summary["balanced_accuracy_sd"] / np.sqrt(
        summary["n_repeats"]
    )
    summary["balanced_accuracy_ci95_low"] = (
        summary["balanced_accuracy_mean"] - 1.96 * summary["balanced_accuracy_sem"]
    ).clip(0, 1)
    summary["balanced_accuracy_ci95_high"] = (
        summary["balanced_accuracy_mean"] + 1.96 * summary["balanced_accuracy_sem"]
    ).clip(0, 1)
    summary["macro_f1_sem"] = summary["macro_f1_sd"] / np.sqrt(summary["n_repeats"])
    summary["macro_f1_ci95_low"] = (
        summary["macro_f1_mean"] - 1.96 * summary["macro_f1_sem"]
    ).clip(0, 1)
    summary["macro_f1_ci95_high"] = (
        summary["macro_f1_mean"] + 1.96 * summary["macro_f1_sem"]
    ).clip(0, 1)
    recall_summary = (
        recalls.groupby(["sigma", "method", "class"], as_index=False)
        .agg(
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            n_repeats=("repeat", "size"),
        )
    )
    recall_summary["recall_sem"] = (
        recall_summary["recall_sd"] / np.sqrt(recall_summary["n_repeats"])
    )
    return summary, recall_summary


# ---------------------------------------------------------------------------
# 3. Supplementary diagnostic figures
# ---------------------------------------------------------------------------

def set_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, stem, destination):
    for extension in ("png", "pdf"):
        fig.savefig(
            destination / f"{stem}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_controls(summary: pd.DataFrame):
    """Supplementary panel: heterogeneous-edge signal versus negative controls."""
    set_style()
    subset = summary[summary["method"] == "dCor-SWEET"].sort_values("sigma")
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    ax.plot(
        subset["sigma"], subset["balanced_accuracy_mean"],
        color=COLORS["dCor-SWEET"], marker="o", label="Heterogeneous-edge features",
    )
    ax.plot(
        subset["sigma"], subset["shared_only_mean"],
        color="#6A51A3", marker="s", label="Shared-edge-only control",
    )
    ax.plot(
        subset["sigma"], subset["permuted_label_mean"],
        color="#777777", marker="^", label="Permuted-label control",
    )
    ax.axhline(0.2, color="#7F7F7F", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_xlabel("Noise level (σ)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.15, 0.9)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Signal and negative controls (dCor-SWEET)", fontweight="bold")
    save_figure(fig, "FigureS1_negative_controls", SUPPLEMENTARY_FIGURE_DIR)


def plot_confusion(confusions: pd.DataFrame, sigma: float = 0.4):
    """Supplementary panel: pooled row-normalized confusion matrices."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    image = None
    for ax, method in zip(axes, METHODS):
        subset = confusions[(confusions["sigma"] == sigma) & (confusions["method"] == method)]
        table = subset.groupby(["true_class", "predicted_class"])["n_samples"].sum().unstack(fill_value=0)
        table = table.reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
        matrix = table.to_numpy(dtype=float, copy=True)
        matrix /= matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for row in range(5):
            for column in range(5):
                value = matrix[row, column]
                ax.text(
                    column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.55 else "#222222", fontsize=6.5,
                )
        ax.set_title(method)
        ax.set_xticks(range(5), CLASS_NAMES, rotation=42, ha="right")
        ax.set_yticks(range(5), CLASS_NAMES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.025)
    colorbar.set_label("Row proportion")
    fig.suptitle(f"Five-fold cross-validated confusion at σ = {sigma:g}", fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.885, bottom=0.30, top=0.82, wspace=0.42)
    save_figure(fig, "FigureS2_confusion_noise_0p4", SUPPLEMENTARY_FIGURE_DIR)


def plot_recall_heatmaps(recall_summary: pd.DataFrame):
    """Supplementary diagnostic: class-specific recall heatmaps."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25), sharey=True)
    image = None
    for panel, (ax, method) in enumerate(zip(axes, METHODS)):
        subset = recall_summary[recall_summary["method"] == method]
        table = subset.pivot(index="class", columns="sigma", values="recall_mean")
        table = table.reindex(index=CLASS_NAMES, columns=sorted(subset["sigma"].unique()))
        values = table.to_numpy(float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.15, vmax=0.85, aspect="auto")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.62 else "#222222", fontsize=6.5,
                )
        ax.set_title(method)
        ax.set_xticks(range(values.shape[1]), [f"{value:g}" for value in table.columns])
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Noise level (σ)")
        if panel == 0:
            ax.set_ylabel("True sample class")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.025)
    colorbar.set_label("Mean recall")
    fig.suptitle("Class-specific cross-validated recall", fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.885, bottom=0.19, top=0.82, wspace=0.38)
    save_figure(fig, "FigureS3_class_recall_heatmaps", SUPPLEMENTARY_FIGURE_DIR)


# ---------------------------------------------------------------------------
# 4. Manuscript Figure 3
# ---------------------------------------------------------------------------

FIGURE3_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.6,
    "axes.titlesize": 7.9,
    "axes.titleweight": "semibold",
    "axes.labelsize": 8.1,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.72,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _fig3_style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out")


def _fig3_panel_label(ax: plt.Axes, text: str, x_offset: float = -23.0) -> None:
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(x_offset, 5.5),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        annotation_clip=False,
    )


def _fig3_overall_panel(ax, summary, mean_col, sem_col, ylabel) -> None:
    """Panels A/B: overall metric versus noise, mean with 95% CI band."""
    for method in METHODS:
        subset = summary.loc[summary["method"].eq(method)].sort_values("sigma")
        x = subset["sigma"].to_numpy(float)
        y = subset[mean_col].to_numpy(float)
        ci = 1.96 * subset[sem_col].to_numpy(float)
        ax.fill_between(
            x,
            np.clip(y - ci, 0.0, 1.0),
            np.clip(y + ci, 0.0, 1.0),
            color=COLORS[method],
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.50,
            markersize=4.4,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )
    ax.axhline(
        RANDOM_BASELINE,
        color=RANDOM_COLOR,
        linewidth=0.85,
        linestyle=(0, (5, 5)),
        zorder=2,
    )
    ax.set(
        xlim=(-0.035, 1.035),
        ylim=(0.15, 0.74),
        xticks=FIGURE3_SIGMAS,
        yticks=np.arange(0.2, 0.71, 0.1),
        xlabel="Noise level, σ",
        ylabel=ylabel,
    )
    _fig3_style_axis(ax)


def _fig3_recall_facet(ax, recall_summary, class_name, show_ylabel) -> None:
    """Panel C: one relation class per facet, recall versus noise."""
    class_data = recall_summary.loc[recall_summary["class"].eq(class_name)]
    for method in METHODS:
        subset = class_data.loc[class_data["method"].eq(method)].sort_values("sigma")
        x = subset["sigma"].to_numpy(float)
        y = subset["recall_mean"].to_numpy(float)
        ci = 1.96 * subset["recall_sem"].to_numpy(float)
        ax.fill_between(
            x,
            np.clip(y - ci, 0.0, 1.0),
            np.clip(y + ci, 0.0, 1.0),
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.25,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=3,
        )
    ax.axhline(
        RANDOM_BASELINE,
        color=RANDOM_COLOR,
        linewidth=0.8,
        linestyle=(0, (5, 5)),
        zorder=2,
    )
    ax.set(
        xlim=(-0.055, 1.055),
        ylim=(0.10, 0.82),
        xticks=(0.0, 0.5, 1.0),
        yticks=(0.2, 0.4, 0.6, 0.8),
        title=class_name,
    )
    ax.tick_params(axis="x", pad=1.5)
    if show_ylabel:
        ax.set_ylabel("Class-specific recall")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    _fig3_style_axis(ax)


def plot_manuscript_figure3(summary: pd.DataFrame, recall_summary: pd.DataFrame):
    """Manuscript Figure 3 (ACA double-column width).

    A, balanced accuracy; B, macro-F1 across the full noise grid. C, class-
    specific recall with relation classes as facets and all six noise levels
    forming each trajectory. Lines are means over repeats; bands are 95% CIs.
    """
    observed_sigmas = np.sort(summary["sigma"].unique().astype(float))
    recall_sigmas = np.sort(recall_summary["sigma"].unique().astype(float))
    if not (
        np.array_equal(observed_sigmas, FIGURE3_SIGMAS)
        and np.array_equal(recall_sigmas, FIGURE3_SIGMAS)
    ):
        raise ValueError("Figure 3 requires all six prespecified noise levels (0 to 1.0).")
    if set(summary["method"]) != set(METHODS):
        raise ValueError("Unexpected method set in the metric summary")
    if set(recall_summary["class"]) != set(CLASS_NAMES):
        raise ValueError("Unexpected relation-class set in the recall summary")

    stem = MAIN_FIGURE_DIR / FIGURE3_STEM
    # Isolate Figure 3 styling from the supplementary figures.
    with mpl.rc_context():
        mpl.rcdefaults()
        mpl.rcParams.update(FIGURE3_RC)

        fig = plt.figure(figsize=(7.35, 4.90))
        outer = fig.add_gridspec(2, 1, height_ratios=(0.92, 1.05), hspace=0.36)
        top = outer[0].subgridspec(1, 2, wspace=0.25)
        bottom = outer[1].subgridspec(1, 5, wspace=0.12)

        ax_a = fig.add_subplot(top[0, 0])
        ax_b = fig.add_subplot(top[0, 1])
        recall_axes = [fig.add_subplot(bottom[0, index]) for index in range(5)]

        _fig3_overall_panel(
            ax_a, summary,
            mean_col="balanced_accuracy_mean",
            sem_col="balanced_accuracy_sem",
            ylabel="Balanced accuracy",
        )
        _fig3_overall_panel(
            ax_b, summary,
            mean_col="macro_f1_mean",
            sem_col="macro_f1_sem",
            ylabel="Macro-F1",
        )
        for index, (ax, class_name) in enumerate(zip(recall_axes, CLASS_NAMES)):
            _fig3_recall_facet(ax, recall_summary, class_name, show_ylabel=index == 0)

        _fig3_panel_label(ax_a, "A")
        _fig3_panel_label(ax_b, "B")
        _fig3_panel_label(recall_axes[0], "C", x_offset=-25.0)

        fig.text(0.535, 0.119, "Noise level, σ", ha="center", va="center", fontsize=8.1)
        handles = [
            Line2D(
                [0], [0],
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.45,
                markersize=4.3,
                label=DISPLAY_NAMES[method],
            )
            for method in METHODS
        ]
        handles.append(
            Line2D(
                [0], [0],
                color=RANDOM_COLOR,
                linewidth=0.85,
                linestyle=(0, (5, 5)),
                label="Random baseline (0.20)",
            )
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.010),
            ncol=4,
            frameon=False,
            handlelength=2.1,
            columnspacing=1.20,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.082, right=0.992, top=0.972, bottom=0.215)

        MAIN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()

        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        fig.savefig(stem.with_suffix(".png"), dpi=360, **save_kwargs)
        fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
        fig.savefig(stem.with_suffix(".svg"), **save_kwargs)
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=1000,
            pil_kwargs={"compression": "tiff_lzw"},
            **save_kwargs,
        )
        plt.close(fig)
    print(stem.with_suffix(".png"))


# ---------------------------------------------------------------------------
# 5. Tables, validation and command-line entry point
# ---------------------------------------------------------------------------

def parameter_table(config: Config):
    return pd.DataFrame(
        [
            ("classes", ", ".join(CLASS_NAMES)),
            ("samples_per_class", config.n_per_class),
            ("samples_total", config.n_samples),
            ("n_variables", config.n_variables),
            ("common_backbone", "5 linear M1-M2 edges"),
            ("class_specific_edges", "5 per relation class"),
            ("classifier_features", "5 aggregated heterogeneous-edge ranks"),
            ("cross_validation", "stratified 5-fold"),
            ("sigma_grid", ",".join(map(str, config.sigma_grid))),
            ("n_repeats", config.n_repeats),
        ],
        columns=["parameter", "value"],
    )


def validate(metrics, recalls, confusions, config):
    expected = len(config.sigma_grid) * config.n_repeats * len(METHODS)
    assert len(metrics) == expected
    assert len(recalls) == expected * len(CLASS_NAMES)
    assert len(confusions) == expected * len(CLASS_NAMES) ** 2
    assert metrics["balanced_accuracy"].between(0, 1).all()
    assert confusions.groupby(["sigma", "repeat", "method"])["n_samples"].sum().eq(config.n_samples).all()


def write_tables(metrics, recalls, confusions, summary, recall_summary, config):
    metrics.to_csv(DATA_TABLE_DIR / "metrics_by_repeat.csv", index=False)
    recalls.to_csv(DATA_TABLE_DIR / "recall_by_repeat.csv", index=False)
    confusions.to_csv(DATA_TABLE_DIR / "confusion_counts_by_repeat.csv", index=False)
    summary.to_csv(DATA_TABLE_DIR / "metrics_summary.csv", index=False)
    recall_summary.to_csv(DATA_TABLE_DIR / "recall_summary.csv", index=False)
    parameter_table(config).to_csv(DATA_TABLE_DIR / "simulation_parameters.csv", index=False)


def generate_all_figures(summary, recall_summary, confusions, config: Config = CFG):
    """Generate manuscript Figure 3 and the supplementary diagnostic figures."""
    plot_manuscript_figure3(summary, recall_summary)
    plot_controls(summary)
    display_sigma = min(config.sigma_grid, key=lambda value: abs(value - 0.4))
    plot_confusion(confusions, display_sigma)
    plot_recall_heatmaps(recall_summary)


def load_saved_results(input_dir: Path = DATA_TABLE_DIR):
    required = {
        "metrics_summary": input_dir / "metrics_summary.csv",
        "recall_summary": input_dir / "recall_summary.csv",
        "confusions": input_dir / "confusion_counts_by_repeat.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing saved result tables: " + ", ".join(missing))
    return (
        pd.read_csv(required["metrics_summary"]),
        pd.read_csv(required["recall_summary"]),
        pd.read_csv(required["confusions"]),
    )


def regenerate_manuscript_figure3_from_tables(input_dir: Path = DATA_TABLE_DIR):
    """Rebuild only the manuscript Figure 3 without rerunning simulation."""
    summary, recall_summary, _ = load_saved_results(input_dir)
    plot_manuscript_figure3(summary, recall_summary)
    print(f"Regenerated manuscript Figure 3 in: {MAIN_FIGURE_DIR}")


def main(config: Config = CFG, figures_only: bool = False, skip_figures: bool = False,
         input_dir: Path = DATA_TABLE_DIR):
    for directory in (MAIN_FIGURE_DIR, SUPPLEMENTARY_FIGURE_DIR, DATA_TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if figures_only:
        summary, recall_summary, confusions = load_saved_results(input_dir)
        generate_all_figures(summary, recall_summary, confusions, config)
        print(f"Regenerated all figures from saved tables in: {OUTPUT_DIR}")
        return

    metrics, recalls, confusions = run(config)
    validate(metrics, recalls, confusions, config)
    summary, recall_summary = summarize(metrics, recalls)
    write_tables(metrics, recalls, confusions, summary, recall_summary, config)
    if not skip_figures:
        generate_all_figures(summary, recall_summary, confusions, config)
    print("\nCompleted successfully")
    print(f"Results: {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run or replot the shared-backbone heterogeneous simulation."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate Figure 3 and the supplementary figures from the saved CSV tables",
    )
    mode.add_argument(
        "--manuscript-figure-only",
        action="store_true",
        help="regenerate only the manuscript Figure 3 from saved tables",
    )
    mode.add_argument(
        "--skip-figures",
        action="store_true",
        help="run the simulation and write tables without generating figures",
    )
    parser.add_argument("--input-dir", type=Path,
                        help="Read precomputed CSVs from this directory in either replot mode.")
    arguments = parser.parse_args()
    if arguments.input_dir is not None and not (
        arguments.figures_only or arguments.manuscript_figure_only
    ):
        parser.error("--input-dir requires a replot mode")
    input_dir = arguments.input_dir or DATA_TABLE_DIR
    if arguments.manuscript_figure_only:
        regenerate_manuscript_figure3_from_tables(input_dir)
    else:
        main(figures_only=arguments.figures_only, skip_figures=arguments.skip_figures,
             input_dir=input_dir)
