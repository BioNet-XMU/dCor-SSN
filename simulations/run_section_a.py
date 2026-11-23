"""Reproduce Section A and archive the inputs and intermediate ranking statistics."""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import homogeneous as original

N, P, N_EDGES, N_REPEATS, K_TOP = 200, 70, 2415, 100, 121
SIGMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SEEDS = np.arange(42, 42 + N_REPEATS)
TRIU = np.triu_indices(P, k=1)
LOOKUP = original.build_edge_lookup(TRIU)
GROUPS = original.classify_edges(LOOKUP) | original.build_b_edges(LOOKUP)
GROUP_NAMES = tuple(GROUPS)
GROUP_SIZES = np.array([len(indices) for indices in GROUPS.values()])
METHODS = tuple(original.METHOD_ORDER)
FRAMEWORKS = ("population", "SWEET")
POPULATION = {
    "dcor": original.dcor_edges,
    "pearson": original.pearson_edges,
    "spearman": original.spearman_edges,
}


def check_original_config():
    config = {
        "N_SAMPLES": N, "M": P, "MODULE_SIZE": 10, "N_MODULES": 7,
        "R": N_REPEATS, "SIGMA_GRID": SIGMAS, "BASE_SEED": 42,
        "LINEAR_COEF": 0.7, "LINEAR_INTERCEPT": 1.0, "SINE_FREQ": 2.0,
        "LOG_SHIFT": 1.0, "K": 0.1, "EPS": 0.01,
    }
    actual = {key: getattr(original.cfg, key) for key in config}
    if actual != config:
        raise ValueError(f"Original simulation settings differ from the frozen configuration: {actual}")
    if GROUP_SIZES.tolist() != [150, 645, 1620, 10, 10, 10, 10, 10]:
        raise ValueError(f"Unexpected ground-truth groups: {GROUP_SIZES}")
    return config


def rank_statistics(weights, check_rank=False):
    scores = np.abs(weights)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite edge score")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.uint16)
    np.put_along_axis(ranks, order, np.arange(1, N_EDGES + 1, dtype=np.uint16)[None, :], axis=1)
    if check_rank:
        np.testing.assert_array_equal(ranks[0], original.rank_edges_desc(scores[0]))
    rank_sums = np.column_stack([ranks[:, idx].sum(axis=1, dtype=np.uint32) for idx in GROUPS.values()])
    top5_counts = np.column_stack([
        (ranks[:, idx] <= K_TOP).sum(axis=1, dtype=np.uint16) for idx in GROUPS.values()
    ])
    np.testing.assert_array_equal(top5_counts[:, :3].sum(axis=1), K_TOP)
    return ranks, rank_sums, top5_counts


def run_trial(task):
    sigma, repeat = task
    seed = int(SEEDS[repeat])
    X = original.generate_dataset(seed, sigma)
    assert X.shape == (N, P)
    population_weights, population_ranks = [], []
    population_sums, population_counts = [], []
    sweet_sums, sweet_counts = [], []
    for method in METHODS:
        weights = POPULATION[method](X, TRIU)
        ranks, sums, counts = rank_statistics(weights[None, :], check_rank=repeat == 0)
        population_weights.append(weights)
        population_ranks.append(ranks[0])
        population_sums.append(sums[0])
        population_counts.append(counts[0])

        weights = original.SWEET_ESTIMATORS[method](X, TRIU)
        assert weights.shape == (N, N_EDGES)
        _, sums, counts = rank_statistics(weights, check_rank=repeat == 0)
        sweet_sums.append(sums)
        sweet_counts.append(counts)
    return (X, population_weights, population_ranks, population_sums, population_counts,
            sweet_sums, sweet_counts)


def save_noise_level(sigma, trials, output):
    columns = (
        "X", "population_scores", "population_ranks", "population_rank_sums", "population_top5_counts",
        "sweet_sample_rank_sums", "sweet_sample_top5_counts",
    )
    arrays = {name: np.stack(values) for name, values in zip(columns, zip(*trials))}
    filename = f"sigma_{int(round(10 * sigma)):02d}.npz"
    np.savez_compressed(output / filename, sigma=sigma, seeds=SEEDS, methods=METHODS,
                        groups=GROUP_NAMES, group_sizes=GROUP_SIZES, **arrays)
    rows = []
    for framework, rank_sums, counts in (
        ("population", arrays["population_rank_sums"], arrays["population_top5_counts"]),
        ("SWEET", arrays["sweet_sample_rank_sums"].mean(axis=2), arrays["sweet_sample_top5_counts"].mean(axis=2)),
    ):
        average_ranks, rates = rank_sums / GROUP_SIZES, counts / GROUP_SIZES
        for r, seed in enumerate(SEEDS):
            for m, method in enumerate(METHODS):
                for g, name in enumerate(GROUP_NAMES):
                    rows.append((sigma, r, seed, framework, method, name, GROUP_SIZES[g],
                                 average_ranks[r, m, g], rates[r, m, g]))
    return rows


def save_tables(rows, out):
    frame = pd.DataFrame(rows, columns=[
        "sigma", "repeat", "seed", "framework", "method", "name", "n_edges", "mean_rank", "top5_rate",
    ])
    frame["mean_rank_percent"] = frame["mean_rank"] * 100 / N_EDGES
    frame.to_csv(out / "by_repeat.csv", index=False)
    summary = frame.groupby(["sigma", "framework", "method", "name", "n_edges"], as_index=False).agg(
        mean_rank_percent=("mean_rank_percent", "mean"), rank_sd_percent=("mean_rank_percent", "std"),
        rank_sem_percent=("mean_rank_percent", "sem"), top5_rate=("top5_rate", "mean"),
        top5_rate_sd=("top5_rate", "std"), top5_rate_sem=("top5_rate", "sem"), n_repeats=("repeat", "size"),
    )
    assert len(summary) == 288 and summary["n_repeats"].eq(N_REPEATS).all()
    summary.to_csv(out / "summary.csv", index=False)

    paired = frame.pivot(index=["sigma", "repeat", "seed", "name"], columns=["framework", "method"],
                         values=["mean_rank_percent", "top5_rate"])
    contrasts = []
    for framework in FRAMEWORKS:
        for comparator in ("pearson", "spearman"):
            contrasts.append(paired_contrast(
                paired, f"{framework}:dcor_vs_{comparator}", (framework, "dcor"), (framework, comparator)
            ))
    for method in METHODS:
        contrasts.append(paired_contrast(
            paired, f"{method}:SWEET_vs_population", ("SWEET", method), ("population", method)
        ))
    deltas = pd.concat(contrasts, ignore_index=True)
    deltas.to_csv(out / "paired_by_repeat.csv", index=False)
    deltas.groupby(["sigma", "name", "contrast"], as_index=False).agg(
        mean_rank_improvement_pp=("rank_improvement_pp", "mean"), rank_improvement_sem_pp=("rank_improvement_pp", "sem"),
        mean_top5_difference_pp=("top5_difference_pp", "mean"), top5_difference_sem_pp=("top5_difference_pp", "sem"),
        n_repeats=("seed", "size"),
    ).to_csv(out / "paired_summary.csv", index=False)
    return summary


def paired_contrast(paired, label, left, right):
    rank, top5 = paired["mean_rank_percent"], paired["top5_rate"]
    return pd.DataFrame({
        "contrast": label,
        "rank_improvement_pp": rank[right] - rank[left],
        "top5_difference_pp": (top5[left] - top5[right]) * 100,
    }).reset_index()


def save_figure(summary, out):
    groups = {"related_all": "All dependent edges", "quadratic": "Quadratic", "absolute": "Absolute value"}
    display = {"dcor": "dCor", "pearson": "Pearson", "spearman": "Spearman"}
    styles = {"population": "--", "SWEET": "-"}
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 4.8), sharex=True, sharey="row")
    for col, (name, title) in enumerate(groups.items()):
        axs[0, col].set_title(title)
        for framework in FRAMEWORKS:
            for method in METHODS:
                subset = summary.loc[(summary["name"] == name) & (summary["method"] == method)
                                     & (summary["framework"] == framework)].sort_values("sigma")
                color = original.COLORS[method]
                style = styles[framework]
                for row, metric, error, scale in (
                    (0, "mean_rank_percent", "rank_sem_percent", 1),
                    (1, "top5_rate", "top5_rate_sem", 100),
                ):
                    axs[row, col].errorbar(
                        subset["sigma"], subset[metric] * scale, yerr=subset[error] * scale,
                        color=color, linestyle=style, marker=original.MARKERS[method],
                        markersize=3, linewidth=1.15, capsize=2,
                    )
        axs[0, col].axhline((N_EDGES + 1) / (2 * N_EDGES) * 100, color=".7", lw=.75, zorder=0)
        axs[1, col].axhline(K_TOP / N_EDGES * 100, color=".7", lw=.75, zorder=0)
        axs[1, col].set(xlabel="Noise level σ", xticks=SIGMAS)
    axs[0, 0].set_ylabel("Mean rank (% of all edges)")
    axs[0, 0].set(ylim=(0, 53), yticks=np.arange(0, 51, 10))
    axs[1, 0].set_ylabel("Top 5% inclusion (%)")
    axs[1, 0].set_ylim(-2, 105)
    for label, ax in zip("ABCDEF", axs.flat):
        ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(-10, 3),
                    textcoords="offset points", ha="left", va="bottom", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=original.COLORS[m], marker=original.MARKERS[m], label=display[m])
               for m in METHODS]
    for framework, label in (("population", "Population"), ("SWEET", "SWEET")):
        handles.append(Line2D([], [], color=".25", linestyle=styles[framework], label=label))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(out / "SectionA_population_vs_SWEET.pdf", bbox_inches="tight")
    fig.savefig(out / "SectionA_population_vs_SWEET.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "section_a")
    args = parser.parse_args()
    start = perf_counter()
    config = check_original_config()
    data, tables, figures = (args.output / name for name in ("intermediate", "tables", "figures"))
    for directory in (data, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    edges = pd.DataFrame({"edge_id": np.arange(N_EDGES), "node_1": TRIU[0], "node_2": TRIU[1]})
    for name, indices in GROUPS.items():
        edges[name] = np.isin(edges["edge_id"], indices)
    edges.to_csv(data / "edge_groups.csv", index=False)
    tasks = [(sigma, repeat) for sigma in SIGMAS for repeat in range(N_REPEATS)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(run_trial, tasks, chunksize=2)
        for sigma in SIGMAS:
            trials = list(islice(results, N_REPEATS))
            assert len(trials) == N_REPEATS
            rows.extend(save_noise_level(sigma, trials, data))
            print(f"sigma={sigma:.1f}: {N_REPEATS} datasets archived", flush=True)
    summary = save_tables(rows, tables)
    save_figure(summary, figures)
    sweet = summary.loc[summary["framework"].eq("SWEET")].copy()
    sweet["kind"] = np.where(sweet["name"].isin(original.B_RELATION_ORDER), "b_relation", "group")
    sweet["rank_mean"] = sweet["mean_rank_percent"] * N_EDGES / 100
    sweet["rank_sem"] = sweet["rank_sem_percent"] * N_EDGES / 100
    sweet["total_edges"] = N_EDGES
    original.FIG_DIR = figures
    original.plot_figure2(sweet, top5=summary)
    sources = [Path(original.__file__), Path(__file__)]
    manifest = {
        "configuration": config, "frameworks": FRAMEWORKS, "methods": METHODS,
        "edge_groups": dict(zip(GROUP_NAMES, GROUP_SIZES.tolist())), "top5_k": K_TOP,
        "weights_ranked_by": "absolute value, ties in original upper-triangle order",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "workers": args.workers, "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "wall_seconds": perf_counter() - start,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
