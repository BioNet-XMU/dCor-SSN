"""Reproduce the three-cohort pooled edge-concordance analysis."""
import argparse
import os
import platform
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from cohort_data import COHORTS, ROOT, load_cohort, write_json
from resources import peak_rss_mib

sys.path.insert(0, str(ROOT))
from dcor_ssn import dcor_ssn_sweet, pearson_ssn_sweet, spearman_ssn_sweet

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEET_METHODS = {
    "dcor": dcor_ssn_sweet, "pearson": pearson_ssn_sweet, "spearman": spearman_ssn_sweet,
}
QS = (.05, .10, .15, .20, .30, .40, .50)
SUMMARY_FILES = (
    "between_sample_summary.csv", "cross_method_summary.csv",
    "paired_structure_summary.csv", "high_recurrence_partition.csv",
)
COHORT_LABELS = {"ST000355": "Breast cancer", "ST003390": "Type 2 diabetes", "GC": "Gastric cancer"}
MAIN_Q = 0.10


def top_sets(scores, k):
    """Pack exact Top-q masks in feature-index order using the original rule."""
    n, m = scores.shape
    counts = np.zeros(m, dtype=np.int32)
    packed = np.empty((n, (m + 63) // 64), dtype=np.uint64)
    mask = np.zeros(64 * packed.shape[1], dtype=np.bool_)
    for i, row in enumerate(scores):
        selected = np.argpartition(row, m - k)[m - k:]
        counts[selected] += 1
        mask[selected] = True
        packed[i] = np.packbits(mask).view(np.uint64)
        mask[selected] = False
    assert counts.sum() == n * k
    return packed, counts


def jaccard(intersection, k):
    return intersection / (2 * k - intersection)


def pair_jaccards(packed, k):
    """Return all unordered sample-pair overlaps in upper-triangle order."""
    return np.concatenate([
        jaccard(np.bitwise_count(packed[i] & packed[i + 1:]).sum(axis=1, dtype=np.int32), k)
        for i in range(len(packed) - 1)
    ])


def analyse(cohort, x, registration, output, *, intermediate=None):
    n, p = x.shape
    m = registration["n_edges"]
    expected_edges = np.column_stack(np.triu_indices(p, k=1))
    sets, high, pair_values = {}, {}, {}
    summaries, cross_method, paired, partitions, runtime = [], [], [], [], []
    for method, estimator in SWEET_METHODS.items():
        recurrence_counts = []
        start = perf_counter()
        scores, edges = estimator(x)
        elapsed = perf_counter() - start
        runtime.append({"method": method, "network_seconds": elapsed, "peak_rss_mib_so_far": peak_rss_mib()})
        np.testing.assert_array_equal(edges, expected_edges)
        assert np.isfinite(scores).all()
        np.abs(scores, out=scores)
        for q in QS:
            k = int(np.ceil(q * m))
            packed, counts = top_sets(scores, k)
            if q == .10:
                ref = np.argpartition(scores[:2], m - k, axis=1)[:, m - k:]
                expected = len(np.intersect1d(*ref))
                assert np.bitwise_count(packed[0] & packed[1]).sum() == expected
            values = pair_jaccards(packed, k)
            sets[q, method] = packed
            high[q, method] = counts * 2 >= n
            pair_values[q, method] = values
            if intermediate is not None:
                recurrence_counts.append(counts.astype(np.uint16))
            n_present, n_high = int((counts > 0).sum()), int(high[q, method].sum())
            summaries.append({
                "cohort": cohort, "q": q, "method": method, "k": k, "n_sample_pairs": len(values),
                "mean_jaccard": values.mean(), "sd_jaccard": values.std(ddof=1),
                "random_set_reference": q / (2 - q),
                "n_edges_ever_top": n_present, "n_edges_recur_ge_0_5": n_high,
                "fraction_high_among_ever_top": n_high / n_present,
                "mean_recurrence_among_ever_top": k / n_present,
            })
        print(f"{cohort}: {method} built in {elapsed:.2f} s, all seven proportions complete", flush=True)
        del scores
        if intermediate is not None:
            np.savez_compressed(
                intermediate / f"{method}_edge_selection.npz",
                q=QS, k=np.ceil(np.asarray(QS) * m).astype(int), n_edges=m,
                top_q_packbits=np.stack([sets[q, method].view(np.uint8) for q in QS]),
                recurrence_counts=np.stack(recurrence_counts),
                between_sample_jaccard=np.stack([pair_values[q, method] for q in QS]),
            )
    cross_values = []
    for q in QS:
        k = int(np.ceil(q * m))
        d, p, s = (high[q, method] for method in SWEET_METHODS)
        categories = {
            "All three": d & p & s, "dCor + Spearman only": d & s & ~p,
            "dCor + Pearson only": d & p & ~s, "Pearson + Spearman only": p & s & ~d,
            "dCor only": d & ~p & ~s, "Pearson only": p & ~d & ~s, "Spearman only": s & ~d & ~p,
        }
        union_count = int((d | p | s).sum())
        counts = {category: int(mask.sum()) for category, mask in categories.items()}
        assert sum(counts.values()) == union_count
        partitions.extend({
            "cohort": cohort, "q": q, "category": category, "count": count,
            "union_count": union_count, "percent_of_union": 100 * count / union_count if union_count else np.nan,
        } for category, count in counts.items())
        for first, second in combinations(SWEET_METHODS, 2):
            values = jaccard(
                np.bitwise_count(sets[q, first] & sets[q, second]).sum(axis=1, dtype=np.int32), k
            )
            if intermediate is not None:
                cross_values.append(values)
            a, b = high[q, first], high[q, second]
            union = int((a | b).sum())
            cross_method.append({
                "cohort": cohort, "q": q, "method_1": first, "method_2": second, "n_samples": n,
                "mean_within_sample_jaccard": values.mean(), "sd_within_sample_jaccard": values.std(ddof=1),
                "high_recurrence_intersection": int((a & b).sum()), "high_recurrence_union": union,
                "high_recurrence_jaccard": float((a & b).sum()) / union if union else np.nan,
            })
            delta = pair_values[q, first] - pair_values[q, second]
            paired.append({
                "cohort": cohort, "q": q, "method_1": first, "method_2": second,
                "n_sample_pairs": len(delta), "mean_paired_jaccard_difference": delta.mean(),
                "fraction_pairs_first_higher": (delta > 0).mean(),
            })
    if intermediate is not None:
        np.savez_compressed(
            intermediate / "within_sample_cross_method.npz", q=QS,
            method_pairs=list(combinations(SWEET_METHODS, 2)),
            jaccard=np.stack(cross_values).reshape(len(QS), -1, n),
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "between_sample_summary.csv", index=False)
    pd.DataFrame(cross_method).to_csv(output / "cross_method_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_structure_summary.csv", index=False)
    partition = pd.DataFrame(partitions)
    partition.to_csv(output / "high_recurrence_partition.csv", index=False)
    return {
        "method_resources": runtime, "total_peak_rss_mib": peak_rss_mib(),
        "platform": platform.platform(), "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "logical_cores": os.cpu_count(),
        "blas": [{k: lib[k] for k in ("internal_api", "version", "num_threads", "architecture")
                  if k in lib} for lib in threadpool_info()],
    }


def run_cohort(cohort, data_dir, output_dir):
    started = perf_counter()
    output = output_dir / cohort
    intermediate = output / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    abundance, ids, features, _, _, registration = load_cohort(cohort, data_dir, prediction=False)
    # Keep the original feature-by-sample convention for the population z-score.
    values = abundance.T
    x = np.ascontiguousarray(((values - values.mean(axis=1, keepdims=True)) /
                              values.std(axis=1, keepdims=True)).T)
    registration["standardization"] = "pooled population z-score, ddof=0; no additional log"
    n, p = x.shape
    features.insert(0, "feature_index", np.arange(p))
    features.to_csv(output / "input_features.csv", index=False)
    pd.DataFrame({"sample_index": np.arange(n), "sample_id": ids}).to_csv(
        output / "input_sample_order.csv", index=False
    )
    write_json(output / "input_registration.json", registration)
    np.savez_compressed(
        intermediate / "standardized_input_and_indices.npz",
        X=x, edge_index_to_features=np.column_stack(np.triu_indices(p, k=1)).astype(np.uint16),
        pair_index_to_samples=np.column_stack(np.triu_indices(n, k=1)).astype(np.uint16),
    )
    print(f"{cohort}: registered {n} samples and {p} features", flush=True)
    runtime = analyse(cohort, x, registration, output, intermediate=intermediate)
    runtime["total_seconds_including_archive"] = perf_counter() - started
    runtime["software"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    write_json(output / "runtime.json", runtime)
    print(f"{cohort}: saved intermediate arrays and tables", flush=True)


def make_figure(summary, output):
    widths = (8.6, 2.9)
    fig, axes = plt.subplots(1, 3, figsize=widths, sharex=True, sharey=True)
    methods = {
        "dcor": ("dCor", "#D55E00", "o"),
        "pearson": ("Pearson", "#0072B2", "s"),
        "spearman": ("Spearman", "#009E73", "^"),
    }
    for ax, cohort, label in zip(axes, COHORTS, "ABC"):
        cohort_table = summary.loc[summary["cohort"].eq(cohort)]
        ax.axvline(MAIN_Q * 100, color=".66", linestyle=":", linewidth=1, zorder=0)
        for method, (title, color, marker) in methods.items():
            sub = cohort_table.loc[cohort_table["method"].eq(method)].sort_values("q")
            ax.plot(sub["q"] * 100, sub["mean_jaccard"], color=color, marker=marker,
                    markersize=3.5, linewidth=1.35, label=title)
        ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(-8, 5),
                    textcoords="offset points", fontweight="bold")
        ax.set_title(COHORT_LABELS[cohort], fontsize=10)
        ax.set(xlabel="Top-q edges (%)", xticks=np.asarray(QS) * 100, ylim=(.55, .9))
        ax.grid(axis="y", color=".91", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean between-sample Jaccard")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .14, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"SectionC_cohort_concordance_scan.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)



def summarize(output_dir):
    tables, figures = output_dir / "tables", output_dir / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    for filename in SUMMARY_FILES:
        combined = pd.concat([pd.read_csv(output_dir / cohort / filename) for cohort in COHORTS],
                             ignore_index=True)
        combined.to_csv(tables / filename, index=False)
        if filename == "between_sample_summary.csv":
            primary = combined.loc[combined["q"].eq(MAIN_Q)]
            wide = primary.pivot(index=["cohort", "k", "n_sample_pairs"], columns="method",
                                 values="mean_jaccard")
            wide.reset_index().set_index("cohort").reindex(COHORTS).to_csv(tables / "primary_q10.csv")
            make_figure(combined, figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=(*COHORTS, "summarize"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "section_c")
    args = parser.parse_args()
    if args.task == "summarize":
        summarize(args.output_dir)
    else:
        run_cohort(args.task, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
