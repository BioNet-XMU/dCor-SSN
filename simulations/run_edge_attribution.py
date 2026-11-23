"""Evaluate fixed simulated edges from the original Section B score archive."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
SECTION_B = HERE.parent / "outputs" / "section_b"
METHODS = ("dCor-SWEET", "Pearson-SWEET", "Spearman-SWEET")
CLASSES = ("Linear", "Quadratic", "Absolute", "Sine", "Logarithmic")
SIGMAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ENCODINGS = ("absolute_score", "signed_score", "priority_rank")
BOOTSTRAP_SEED = 20261122
N_DRAWS = 10_000


def edge_aurocs(ranks, on):
    """Vectorized rank-sum AUROC; axis 2 indexes the 250 samples."""
    positive = on.sum(axis=1)
    assert (positive == 50).all()
    sum_positive = (ranks * on[:, None, :, None]).sum(axis=2)
    return (sum_positive - (positive * (positive + 1) / 2)[:, None, None]) / (
        positive * (on.shape[1] - positive)
    )[:, None, None]


def interval(values, draws):
    resampled = values[draws].mean(axis=1)
    return (*np.quantile(resampled, (0.025, 0.975)),)


def analyze_file(path, sigma, rng, edge_rows, repeat_rows, summary_rows):
    with np.load(path, allow_pickle=False) as data:
        assert float(data["sigma"]) == sigma
        assert tuple(data["classes"]) == CLASSES
        method_ids = [list(data["methods"]).index(method) for method in METHODS]
        scores = data["selected_edge_scores"][:, method_ids].astype(float)
        stored_rank = data["selected_edge_rank_times_two"][:, method_ids].astype(float)
        edge_indices = data["edge_indices"]
        labels, permutation, seeds = (data[k] for k in ("labels", "permuted_labels", "seeds"))
    n_repeat = len(seeds)
    assert scores.shape == stored_rank.shape == (100, 3, 250, 6, 5)
    assert edge_indices.shape == (6, 5) and np.unique(edge_indices).size == 30
    assert len(set(seeds)) == n_repeat and np.isfinite(scores).all()
    assert (stored_rank >= 2).all() and (stored_rank <= 2 * 2415).all()
    draws = rng.integers(0, n_repeat, size=(N_DRAWS, n_repeat))
    reference_checks = 0

    for encoding, score_array in zip(ENCODINGS, (np.abs(scores), scores, -stored_rank / 2)):
        ranks = rankdata(score_array, axis=2, method="average")
        for group_index, relation in enumerate(CLASSES):
            for analysis, edge_group, assigned_labels in (
                ("target", group_index, labels),
                ("shared_backbone", 5, labels),
                ("permuted_labels", group_index, permutation),
            ):
                on = assigned_labels == group_index
                auc = edge_aurocs(ranks[:, :, :, edge_group], on)
                assert (auc >= 0).all() and (auc <= 1).all()
                # Check the direct score direction and treatment of ties independently.
                for method_index in range(len(METHODS)):
                    observed = score_array[0, method_index, :, edge_group, 0]
                    expected = roc_auc_score(on[0], observed)
                    np.testing.assert_allclose(auc[0, method_index, 0], expected, atol=1e-12, rtol=0)
                    reference_checks += 1

                mean_edges = auc.mean(axis=-1)
                for method_index, method in enumerate(METHODS):
                    values = mean_edges[:, method_index]
                    lo, hi = interval(values, draws)
                    summary_rows.append((sigma, analysis, relation, encoding, "auroc", method, values.mean(), lo, hi))
                    for rep, seed in enumerate(seeds):
                        repeat_rows.append((sigma, rep + 1, seed, analysis, relation, encoding, "auroc", method, values[rep]))
                        for edge_id, value in zip(edge_indices[edge_group], auc[rep, method_index]):
                            edge_rows.append((sigma, rep + 1, seed, analysis, relation, encoding, method, edge_id, value))

                for method_index, method in enumerate(METHODS[1:], 1):
                    name = f"dCor-SWEET minus {method}"
                    diff = mean_edges[:, 0] - mean_edges[:, method_index]
                    lo, hi = interval(diff, draws)
                    summary_rows.append((sigma, analysis, relation, encoding, "paired_delta", name, diff.mean(), lo, hi))
                    repeat_rows.extend((sigma, rep + 1, seed, analysis, relation, encoding, "paired_delta", name, diff[rep])
                                       for rep, seed in enumerate(seeds))

    return reference_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SECTION_B / "intermediate")
    parser.add_argument("--output", type=Path, default=SECTION_B / "edge_attribution")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    edge_rows, repeat_rows, summary_rows = [], [], []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    input_hashes = {}
    checks = 0
    for sigma in SIGMAS:
        path = args.input / f"sigma_{round(10 * sigma):02d}.npz"
        with path.open("rb") as handle:
            input_hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        checks += analyze_file(path, sigma, rng, edge_rows, repeat_rows, summary_rows)
        print(f"Completed sigma={sigma:.1f}", flush=True)

    pd.DataFrame(edge_rows, columns=("sigma", "repeat", "seed", "analysis", "relation", "encoding",
                                     "method", "edge_index", "auroc")).to_csv(args.output / "auroc_by_edge.csv.gz", index=False)
    pd.DataFrame(repeat_rows, columns=("sigma", "repeat", "seed", "analysis", "relation", "encoding",
                                       "metric", "comparison", "value")).to_csv(args.output / "auroc_by_repeat.csv.gz", index=False)
    summary = pd.DataFrame(summary_rows, columns=("sigma", "analysis", "relation", "encoding", "metric",
                                                  "comparison", "mean", "ci95_low", "ci95_high"))
    summary.to_csv(args.output / "auroc_summary.csv", index=False)
    primary = summary[
        summary.sigma.eq(0.4) & summary.analysis.eq("target")
        & summary.relation.isin(("Quadratic", "Absolute"))
        & summary.encoding.isin(("absolute_score", "priority_rank"))
        & (summary.metric.eq("paired_delta") | summary.comparison.eq(METHODS[0]))
    ].copy()
    assert len(primary) == 12
    primary["lower_bound_target"] = np.where(primary.metric.eq("paired_delta"), 0.0, 0.5)
    primary["passes"] = primary.ci95_low.gt(primary.lower_bound_target)
    primary.to_csv(args.output / "primary_decision.csv", index=False)
    manifest = {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": N_DRAWS,
                "reference_auc_checks": checks, "input_sha256": input_hashes,
                "edge_rows": len(edge_rows), "repeat_rows": len(repeat_rows),
                "summary_rows": len(summary), "primary_passes": int(primary.passes.sum()),
                "main_text_criterion_met": bool(primary.passes.all())}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(primary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
