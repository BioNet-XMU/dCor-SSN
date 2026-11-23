"""Run the frozen five-class simulation and archive the matched SWEET/LIONESS comparison."""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import dcor_ssn as ssn
import five_class as five

SETTINGS = dict(
    module_size=10, n_shared=5, n_per_class=50, n_repeats=100,
    sigma_grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), linear_coef=0.7, linear_intercept=1.0,
    sine_frequency=2.0, log_shift=1.0, sweet_k=0.1, sweet_eps=0.01, base_seed=20260928,
)
SIGMAS = SETTINGS["sigma_grid"]
N_REPEATS = SETTINGS["n_repeats"]
METHODS = (*five.METHODS, "dCor-LIONESS", "Pearson-LIONESS")
CLASSES = five.CLASS_NAMES
FEATURE_NAMES = (*CLASSES, "Shared backbone")
TRIU, SHARED, HETEROGENEOUS = five.edge_design()
SELECTED = np.stack([*HETEROGENEOUS, SHARED])
KEYS = ["sigma", "repeat", "seed", "method"]
SCORES = ["balanced_accuracy", "macro_f1", "shared_only_balanced_accuracy",
          "permuted_label_balanced_accuracy"]


def trial_seed(sigma_index, repeat):
    return SETTINGS["base_seed"] + sigma_index * 100_000 + repeat


def folds(labels, features, seed):
    assignment = np.empty(len(labels), dtype=np.uint8)
    cv = five.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for number, (_, test) in enumerate(cv.split(features, labels)):
        assignment[test] = number
    return assignment


def trial(task):
    sigma_index, repeat = task
    seed = trial_seed(sigma_index, repeat)
    x, y = five.generate_dataset(seed, SIGMAS[sigma_index])
    assert x.shape == (250, 70) and (np.bincount(y) == 50).all()
    y_perm = y[np.random.default_rng(seed + 50_000_000).permutation(len(y))]
    fold = folds(y, x, seed)
    fold_perm = folds(y_perm, x, seed + 1)
    n = len(y)
    full_pearson = ssn.pearson_matrix(x)[TRIU]
    pearson = np.vstack([
        n * full_pearson - (n - 1) * ssn.pearson_matrix(np.delete(x, i, axis=0))[TRIU]
        for i in range(n)
    ])
    dcor, edges = ssn.dcor_ssn_lioness(x)
    np.testing.assert_array_equal(edges, np.column_stack(TRIU))
    error = None
    if task == (0, 0):
        reference = (n * ssn.distance_correlation_matrix(x)
                     - (n - 1) * ssn.distance_correlation_matrix(x[1:]))[TRIU]
        error = float(np.abs(dcor[0] - reference).max())
        np.testing.assert_allclose(dcor[0], reference, atol=1e-10, rtol=0)
    selected_scores, selected_ranks, features, predictions = [], [], [], []
    for method in METHODS:
        if method in five.SWEET_ESTIMATORS:
            scores, _ = five.SWEET_ESTIMATORS[method](x, TRIU, five.CFG)
        else:
            scores = dcor if method == "dCor-LIONESS" else pearson
        assert scores.shape == (n, len(TRIU[0])) and np.isfinite(scores).all()
        ranks = rankdata(-np.abs(scores), axis=1, method="average")[:, SELECTED]
        ranks_times_two = (ranks * 2).astype(np.uint16)
        grouped = ranks.mean(axis=-1)
        if repeat == 0:
            np.testing.assert_array_equal(grouped, five.aggregate_rank_features(scores, [*HETEROGENEOUS, SHARED]))
            np.testing.assert_array_equal(ranks_times_two / 2, ranks)
        selected_scores.append(scores[:, SELECTED])
        selected_ranks.append(ranks_times_two)
        features.append(grouped)
        predictions.append(np.stack([
            five.cross_validated_predictions(grouped[:, :-1], y, seed),
            five.cross_validated_predictions(grouped[:, -1:], y, seed),
            five.cross_validated_predictions(grouped[:, :-1], y_perm, seed + 1),
        ]).astype(np.uint8))
    return (x, y.astype(np.uint8), y_perm.astype(np.uint8), fold, fold_perm, selected_scores,
            selected_ranks, features, predictions), error


def archive_noise_level(directory, sigma_index, trials, metrics, recalls, confusions):
    sigma = SIGMAS[sigma_index]
    seeds = np.array([trial_seed(sigma_index, repeat) for repeat in range(N_REPEATS)])
    names = ("X", "labels", "permuted_labels", "fold", "permuted_fold", "selected_edge_scores",
             "selected_edge_rank_times_two", "rank_features", "out_of_fold_predictions")
    arrays = {name: np.stack(xs) for name, xs in zip(names, zip(*trials))}
    np.savez_compressed(
        directory / f"sigma_{int(round(10 * sigma)):02d}.npz", sigma=sigma, seeds=seeds, methods=METHODS,
        classes=CLASSES, feature_names=FEATURE_NAMES, edge_indices=SELECTED,
        prediction_types=("five_features", "shared_only", "permuted_labels"), **arrays,
    )
    for repeat, seed in enumerate(seeds):
        y = arrays["labels"][repeat]
        y_perm = arrays["permuted_labels"][repeat]
        for m, method in enumerate(METHODS):
            pred, shared, perm = arrays["out_of_fold_predictions"][repeat, m]
            key = (sigma, repeat, seed, method)
            metrics.append((*key, balanced_accuracy_score(y, pred), f1_score(y, pred, average="macro"),
                            balanced_accuracy_score(y, shared), balanced_accuracy_score(y_perm, perm)))
            counts = confusion_matrix(y, pred, labels=np.arange(len(CLASSES)))
            recalls.extend((*key, label, counts[c, c] / counts[c].sum()) for c, label in enumerate(CLASSES))
            confusions.extend((*key, actual, predicted, counts[i, j])
                              for i, actual in enumerate(CLASSES) for j, predicted in enumerate(CLASSES))


def tables(directory, metrics, recalls, confusions, figures):
    metrics = pd.DataFrame(metrics, columns=KEYS + SCORES)
    recalls = pd.DataFrame(recalls, columns=KEYS + ["class", "recall"])
    ccols = ["true_class", "predicted_class", "n_samples"]
    confusion = pd.DataFrame(confusions, columns=KEYS + ccols)
    assert len(metrics) == 3000 and len(recalls) == 15000 and len(confusion) == 75000
    scores_long = metrics.melt(id_vars=KEYS, value_vars=SCORES, var_name="metric")
    long = pd.concat([scores_long, recalls.rename(columns={"class": "metric", "recall": "value"})])
    summary = long.groupby(["sigma", "method", "metric"], as_index=False)["value"].agg(
        mean="mean", sd="std", sem="sem", n_repeats="count")
    assert len(summary) == 270 and summary["n_repeats"].eq(100).all()

    paired = long.pivot(index=["sigma", "repeat", "seed", "metric"], columns="method", values="value")
    delta = pd.DataFrame({
        frame: paired[f"dCor-{frame}"] - paired[f"Pearson-{frame}"] for frame in ("SWEET", "LIONESS")
    })
    delta["LIONESS_minus_SWEET"] = delta["LIONESS"] - delta["SWEET"]
    delta_long = delta.reset_index().melt(
        id_vars=["sigma", "repeat", "seed", "metric"], var_name="contrast", value_name="difference")
    delta_summary = delta_long.groupby(["sigma", "metric", "contrast"], as_index=False)["difference"].agg(
        mean="mean", sd="std", sem="sem", n_repeats="count")
    for file, frame in (
        ("metrics_by_repeat.csv", metrics), ("recalls_by_repeat.csv", recalls),
        ("confusion_counts_by_repeat.csv", confusion), ("summary.csv", summary),
        ("paired_by_repeat.csv", delta_long), ("paired_summary.csv", delta_summary),
    ):
        frame.to_csv(directory / file, index=False)
    lioness = summary.loc[summary["method"].str.endswith("-LIONESS")]
    lioness.pivot(index=["sigma", "method"], columns="metric", values="mean").reindex(
        columns=SCORES + list(CLASSES)).to_csv(directory / "LIONESS_SI_table.csv")
    sweet_metrics = metrics[metrics["method"].isin(five.METHODS)]
    sweet_recalls = recalls[recalls["method"].isin(five.METHODS)]
    sweet_summary, recall_summary = five.summarize(sweet_metrics, sweet_recalls)
    five.MAIN_FIGURE_DIR = figures
    five.SUPPLEMENTARY_FIGURE_DIR = figures
    five.generate_all_figures(sweet_summary, recall_summary, confusion, five.CFG)
    return summary


def supplementary_figure(data, directory):
    panels = (
        ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"),
        ("Quadratic", "Quadratic recall"), ("Absolute", "Absolute-value recall"),
        ("shared_only_balanced_accuracy", "Shared-backbone control"),
        ("permuted_label_balanced_accuracy", "Permuted-label control"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.8), sharex=True, sharey=True)
    for i, ((metric, title), ax) in enumerate(zip(panels, axes.flat)):
        for association in ("dCor", "Pearson"):
            for frame, linestyle in (("SWEET", "-"), ("LIONESS", "--")):
                a = data.loc[data["metric"].eq(metric) & data["method"].eq(f"{association}-{frame}")]
                ax.errorbar(a["sigma"], a["mean"], yerr=a["sem"], linewidth=1.2, capsize=2,
                            color=five.COLORS[f"{association}-SWEET"], marker=five.MARKERS[f"{association}-SWEET"],
                            markersize=3, linestyle=linestyle)
        if metric == "balanced_accuracy" or i >= 4:
            ax.axhline(0.20, color=".7", linewidth=.8, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.annotate("ABCDEF"[i], (0, 1), xycoords="axes fraction", xytext=(-9, 2),
                    textcoords="offset points", va="bottom", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set(ylim=(0, .78), xticks=SIGMAS)
    for ax in axes[1]:
        ax.set_xlabel("Noise level σ")
    handles = [Line2D([], [], color=five.COLORS[f"{m}-SWEET"], marker=five.MARKERS[f"{m}-SWEET"], label=m)
               for m in ("dCor", "Pearson")]
    handles.extend([Line2D([], [], color=".25", linestyle=ls, label=f) for f, ls in
                    (("SWEET", "-"), ("LIONESS", "--"))])
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"SectionB_SWEET_vs_LIONESS.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "section_b")
    args = parser.parse_args()
    start = perf_counter()
    if asdict(five.CFG) != SETTINGS:
        raise ValueError(f"Original simulation settings differ from the frozen values: {asdict(five.CFG)}")
    assert len(np.unique(SELECTED)) == 30
    intermediate, tab, figures = (args.output / name for name in ("intermediate", "tables", "figures"))
    for directory in (intermediate, tab, figures):
        directory.mkdir(parents=True, exist_ok=True)
    group_nodes = np.column_stack(TRIU)[SELECTED.reshape(-1)]
    pd.DataFrame({
        "group": np.repeat(FEATURE_NAMES, 5), "edge_index": SELECTED.reshape(-1),
        "node_1": group_nodes[:, 0], "node_2": group_nodes[:, 1],
    }).to_csv(intermediate / "prespecified_edge_groups.csv", index=False)
    tasks = [(s, r) for s in range(len(SIGMAS)) for r in range(N_REPEATS)]
    metrics, recalls, confusions = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(trial, tasks, chunksize=2)
        for i, sigma in enumerate(SIGMAS):
            batch = list(islice(results, N_REPEATS))
            if i == 0:
                dc_error = batch[0][1]
            archive_noise_level(intermediate, i, [record for record, _ in batch], metrics, recalls, confusions)
            print(f"sigma={sigma:.1f}: {len(batch)} datasets and predictions archived", flush=True)
    summary = tables(tab, metrics, recalls, confusions, figures)
    supplementary_figure(summary, figures)
    sources = [Path(__file__), Path(five.__file__), Path(ssn.__file__)]
    manifest = {
        "configuration": SETTINGS, "methods": METHODS, "class_names": CLASSES,
        "trial_seed": "20260928 + 100000 * zero-based sigma index + zero-based repeat",
        "permutation_seed_offset": 50_000_000, "permuted_cv_seed_offset": 1,
        "CV": "Five stratified shuffled folds with original fold-wise StandardScaler and L2 logistic regression",
        "source_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in sources},
        "lioness_dcor_sample_0_max_abs_difference_vs_full_recalculation": dc_error,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "workers": args.workers, "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "wall_seconds": perf_counter() - start,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
