#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outer-training-reference SWEET Top-q PLS-LR analysis with stable VIP networks.

The workflow deliberately separates three layers:

1. Outer Monte Carlo splits estimate held-out classification performance and
   provide repeated training models for VIP stability.
2. Inner stratified CV selects the number of PLS components by the AUROC of
   the complete PLS-LR pipeline.
3. VIP edges recurring across outer training models are aggregated into
   q-specific stable networks, followed by across-q summaries.

The q values are prespecified sparsity thresholds. They are evaluated as a
sensitivity series and are not selected on the outer test data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
RAW_INPUT = Path(os.environ.get("BREAST_INPUT", BASE / "inputs" / "v5_analysis_data_raw.npz"))
sys.path.insert(0, str(BASE))

from sweet_plslr_common import (  # noqa: E402
    EPS_SWEET,
    K_SWEET,
    binq_rank_pct,
    corr_test,
    corr_train,
    dcor_test,
    dcor_train,
)
from make_figure5_classification_performance import generate_figure as generate_figure5  # noqa: E402


def _environment_output_dir() -> Path:
    """Resolve an optional isolated output directory relative to this script."""
    value = os.environ.get("ANALYSIS_OUTDIR")
    if not value:
        return BASE / "outputs"
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE / path


OUTDIR = _environment_output_dir()
FIG_DIR = OUTDIR / "figures"
VIP_DIR = OUTDIR / "stable_vip_networks"
for directory in (OUTDIR, FIG_DIR, VIP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# Reproducible defaults. Every setting can be overridden through an
# environment variable without editing this file.
SEED = int(os.environ.get("SEED", "42"))
N_SPLITS = int(os.environ.get("N_SPLITS", "100"))
N_PERM = int(os.environ.get("N_PERM", "100"))
N_JOBS = int(os.environ.get("N_JOBS", "8"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.30"))
Q_LIST = [
    float(value)
    for value in os.environ.get(
        "Q_LIST", "0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20"
    ).split(",")
]
MAX_N_COMP = int(os.environ.get("MAX_N_COMP", "8"))
INNER_FOLDS = int(os.environ.get("INNER_FOLDS", "5"))
VIP_CANDIDATE_TOP_N = int(os.environ.get("VIP_CANDIDATE_TOP_N", "500"))
VIP_NETWORK_TOP_NS = sorted(
    {int(value) for value in os.environ.get("VIP_NETWORK_TOP_NS", "100,200,300").split(",")}
)
VIP_PRIMARY_TOP_N = int(os.environ.get("VIP_PRIMARY_TOP_N", "200"))
HUB_TOP_N_PER_Q = int(os.environ.get("HUB_TOP_N_PER_Q", "20"))
RUN_PERM = os.environ.get("RUN_PERM", "1") == "1"

if VIP_PRIMARY_TOP_N not in VIP_NETWORK_TOP_NS:
    raise ValueError("VIP_PRIMARY_TOP_N must be included in VIP_NETWORK_TOP_NS")
if VIP_CANDIDATE_TOP_N < max(VIP_NETWORK_TOP_NS):
    raise ValueError(
        "VIP_CANDIDATE_TOP_N must be at least the largest VIP_NETWORK_TOP_NS value"
    )
if N_SPLITS < 1 or N_PERM < 0 or N_JOBS == 0:
    raise ValueError("N_SPLITS must be positive, N_PERM nonnegative, and N_JOBS nonzero")
if not 0.0 < TEST_SIZE < 1.0:
    raise ValueError("TEST_SIZE must lie strictly between 0 and 1")
if not Q_LIST or len(Q_LIST) != len(set(Q_LIST)) or any(not 0.0 < q <= 1.0 for q in Q_LIST):
    raise ValueError("Q_LIST must contain unique fractions in the interval (0, 1]")

METHODS = ("dCor", "Pearson", "Spearman")
METHOD_ORDER = ("dCor", "Spearman", "Pearson")
METHOD_COLORS = {
    "dCor": "#A23B72",
    "Spearman": "#3CA370",
    "Pearson": "#2E86AB",
}


def qtag(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_content_sha256(path: Path) -> str:
    data = np.load(path, allow_pickle=True)
    digest = hashlib.sha256()
    for name in ("X", "y"):
        array = np.ascontiguousarray(data[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    for name in ("metabolite_names", "sample_ids"):
        values = [str(value) for value in data[name]]
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def load_raw_with_names() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not RAW_INPUT.exists():
        raise FileNotFoundError(
            f"Missing classification input: {RAW_INPUT}. "
            "Run build_classification_input.py first."
        )
    data = np.load(RAW_INPUT, allow_pickle=True)
    required = {"X", "y", "metabolite_names", "sample_ids"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Input {RAW_INPUT} is missing arrays: {sorted(missing)}")
    loaded = (
        data["X"].astype(float),
        data["y"].astype(int),
        [str(value) for value in data["metabolite_names"]],
        [str(value) for value in data["sample_ids"]],
    )
    validate_input(*loaded)
    return loaded


def validate_input(
    x: np.ndarray,
    y: np.ndarray,
    metabolite_names: list[str],
    sample_ids: list[str],
) -> None:
    """Fail early when the packaged scientific input is malformed."""
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y):
        raise ValueError(f"Incompatible X/y shapes: X={x.shape}, y={y.shape}")
    if x.shape[1] != len(metabolite_names) or x.shape[0] != len(sample_ids):
        raise ValueError("Input names do not match the dimensions of X")
    if not np.isfinite(x).all():
        raise ValueError("X contains non-finite values")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"y must contain both binary classes 0 and 1; found {np.unique(y)}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample identifiers must be unique")
    if len(set(metabolite_names)) != len(metabolite_names):
        raise ValueError("Metabolite names must be unique")


def zscore_outer_train_test(
    x: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize raw abundances using outer-training parameters only."""
    mean = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    x_train = np.nan_to_num(
        (x[train_idx] - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0
    )
    x_test = np.nan_to_num(
        (x[test_idx] - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0
    )
    return x_train, x_test


def edge_profiles(
    method: str, x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build test edges relative to the outer-training reference only."""
    if method == "dCor":
        return dcor_train(x_train), dcor_test(x_train, x_test)
    if method == "Pearson":
        return corr_train(x_train, False), corr_test(x_train, x_test, False)
    if method == "Spearman":
        return corr_train(x_train, True), corr_test(x_train, x_test, True)
    raise ValueError(f"Unknown method: {method}")


def train_union_columns(x_train_binary: np.ndarray) -> np.ndarray:
    columns = np.flatnonzero(x_train_binary.sum(axis=0) > 0)
    if len(columns) == 0:
        raise ValueError("The outer-training Top-q union feature space is empty")
    return columns


def make_pls(n_components: int) -> PLSRegression:
    """Use one explicit scaling layer: StandardScaler, then PLS(scale=False)."""
    return PLSRegression(
        n_components=n_components,
        scale=False,
        max_iter=500,
        tol=1e-6,
    )


def make_logistic(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        class_weight=None,
        max_iter=2000,
        random_state=random_state,
    )


def inner_fold_count(y_train: np.ndarray) -> int:
    class_counts = np.bincount(y_train.astype(int))
    positive_counts = class_counts[class_counts > 0]
    if len(positive_counts) < 2:
        return 1
    return int(min(INNER_FOLDS, positive_counts.min()))


def select_n_components_by_plslr_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[int, float, list[dict]]:
    """Select PLS components using inner-CV AUROC of the full PLS-LR model.

    The outer-training Top-q feature space is fixed before this function.
    Within every inner fold, feature scaling, PLS fitting, and logistic fitting
    are learned from the inner-training fold only.
    """
    n_folds = inner_fold_count(y_train)
    if n_folds < 2:
        return 1, float("nan"), []

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_indices = list(cv.split(x_train, y_train))
    min_inner_train_size = min(len(train_idx) for train_idx, _ in fold_indices)
    max_components = int(
        min(MAX_N_COMP, x_train.shape[1], max(1, min_inner_train_size - 1))
    )

    candidate_rows: list[dict] = []
    for n_components in range(1, max_components + 1):
        fold_aurocs: list[float] = []
        for fold_number, (inner_train_idx, inner_valid_idx) in enumerate(
            fold_indices, start=1
        ):
            scaler = StandardScaler()
            x_inner_train = scaler.fit_transform(x_train[inner_train_idx])
            x_inner_valid = scaler.transform(x_train[inner_valid_idx])

            pls = make_pls(n_components)
            pls.fit(x_inner_train, y_train[inner_train_idx])
            logistic = make_logistic(random_state + fold_number)
            logistic.fit(pls.transform(x_inner_train), y_train[inner_train_idx])
            probability = logistic.predict_proba(pls.transform(x_inner_valid))[:, 1]
            fold_aurocs.append(
                float(roc_auc_score(y_train[inner_valid_idx], probability))
            )

        candidate_rows.append(
            {
                "n_components": n_components,
                "mean_inner_auroc": float(np.mean(fold_aurocs)),
                "sd_inner_auroc": float(np.std(fold_aurocs, ddof=1))
                if len(fold_aurocs) > 1
                else 0.0,
                "n_inner_folds": n_folds,
            }
        )

    # Deterministic parsimonious tie rule: highest mean AUROC, then fewer
    # components. Rounding avoids unstable choices from floating-point noise.
    best = sorted(
        candidate_rows,
        key=lambda row: (-round(row["mean_inner_auroc"], 12), row["n_components"]),
    )[0]
    return int(best["n_components"]), float(best["mean_inner_auroc"]), candidate_rows


def vip_scores(pls: PLSRegression) -> np.ndarray:
    weights = pls.x_weights_
    scores = pls.x_scores_
    y_loadings = pls.y_loadings_
    n_features = weights.shape[0]
    explained_y = (scores**2).sum(axis=0) * (y_loadings.ravel() ** 2)
    total = float(explained_y.sum())
    if total <= 0:
        return np.zeros(n_features, dtype=float)
    return np.sqrt(
        n_features * np.sum((weights**2) * explained_y, axis=1) / total
    )


def fit_outer_plslr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> dict:
    n_components, inner_auroc, tuning_rows = select_n_components_by_plslr_auc(
        x_train, y_train, random_state
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    pls = make_pls(n_components)
    pls.fit(x_train_scaled, y_train)
    logistic = make_logistic(random_state)
    logistic.fit(pls.transform(x_train_scaled), y_train)

    test_scores = pls.transform(x_test_scaled)
    probability = logistic.predict_proba(test_scores)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_test, probability)),
        "mcc": float(matthews_corrcoef(y_test, prediction)),
        "n_components": n_components,
        "inner_cv_auroc": inner_auroc,
        "vip": vip_scores(pls),
        "pls": pls,
        "logistic": logistic,
        "scaler": scaler,
        "tuning_rows": tuning_rows,
    }


def edge_name_lookup(n_metabolites: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_metabolites, 1)


def run_one_real_split(
    split_number: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    metabolite_names: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    random_state = SEED + split_number
    x_train, x_test = zscore_outer_train_test(x, train_idx, test_idx)
    y_train, y_test = y[train_idx], y[test_idx]
    tri_i, tri_j = edge_name_lookup(x.shape[1])

    result_rows: list[dict] = []
    vip_rows: list[dict] = []
    tuning_rows: list[dict] = []

    for method in METHODS:
        train_edges, test_edges = edge_profiles(method, x_train, x_test)
        for q in Q_LIST:
            train_binary_full = binq_rank_pct(train_edges, q)
            test_binary_full = binq_rank_pct(test_edges, q)
            columns = train_union_columns(train_binary_full)
            train_binary = train_binary_full[:, columns]
            test_binary = test_binary_full[:, columns]

            fitted = fit_outer_plslr(
                train_binary, y_train, test_binary, y_test, random_state
            )
            result_rows.append(
                {
                    "analysis": "real_labels",
                    "split": split_number,
                    "method": method,
                    "q": q,
                    "auroc": fitted["auroc"],
                    "mcc": fitted["mcc"],
                    "n_components": fitted["n_components"],
                    "inner_cv_auroc": fitted["inner_cv_auroc"],
                    "n_features": int(len(columns)),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_case": int((y_train == 1).sum()),
                    "n_train_control": int((y_train == 0).sum()),
                    "n_test_case": int((y_test == 1).sum()),
                    "n_test_control": int((y_test == 0).sum()),
                }
            )

            for row in fitted["tuning_rows"]:
                tuning_rows.append(
                    {
                        "split": split_number,
                        "method": method,
                        "q": q,
                        **row,
                        "selected": int(
                            row["n_components"] == fitted["n_components"]
                        ),
                    }
                )

            case_frequency = train_binary[y_train == 1].mean(axis=0)
            control_frequency = train_binary[y_train == 0].mean(axis=0)
            vip = fitted["vip"]
            top_local = np.argsort(-vip)[: min(VIP_CANDIDATE_TOP_N, len(vip))]
            for rank, local_idx in enumerate(top_local, start=1):
                edge_idx = int(columns[local_idx])
                node_a = metabolite_names[int(tri_i[edge_idx])]
                node_b = metabolite_names[int(tri_j[edge_idx])]
                vip_rows.append(
                    {
                        "split": split_number,
                        "method": method,
                        "q": q,
                        "rank_in_split": rank,
                        "edge_index": edge_idx,
                        "node_a": node_a,
                        "node_b": node_b,
                        "vip": float(vip[local_idx]),
                        "selected_frequency_case_train": float(
                            case_frequency[local_idx]
                        ),
                        "selected_frequency_control_train": float(
                            control_frequency[local_idx]
                        ),
                        "case_minus_control_train": float(
                            case_frequency[local_idx] - control_frequency[local_idx]
                        ),
                        "n_components": fitted["n_components"],
                        "n_features": int(len(columns)),
                    }
                )

    return result_rows, vip_rows, tuning_rows


def run_one_permutation(
    permutation_number: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> list[dict]:
    """Paired negative control with labels permuted within train and test.

    Within-set permutation preserves class counts in both partitions while
    breaking feature-label correspondence. This is a negative control, not a
    full permutation p-value for a statistic averaged across all outer splits.
    """
    random_state = SEED + 100_000 + permutation_number
    rng = np.random.default_rng(random_state)
    x_train, x_test = zscore_outer_train_test(x, train_idx, test_idx)
    y_train = rng.permutation(y[train_idx])
    y_test = rng.permutation(y[test_idx])
    rows: list[dict] = []

    for method in METHODS:
        train_edges, test_edges = edge_profiles(method, x_train, x_test)
        for q in Q_LIST:
            train_binary_full = binq_rank_pct(train_edges, q)
            test_binary_full = binq_rank_pct(test_edges, q)
            columns = train_union_columns(train_binary_full)
            fitted = fit_outer_plslr(
                train_binary_full[:, columns],
                y_train,
                test_binary_full[:, columns],
                y_test,
                random_state,
            )
            rows.append(
                {
                    "analysis": "permuted_labels",
                    "split": permutation_number,
                    "method": method,
                    "q": q,
                    "auroc": fitted["auroc"],
                    "mcc": fitted["mcc"],
                    "n_components": fitted["n_components"],
                    "inner_cv_auroc": fitted["inner_cv_auroc"],
                    "n_features": int(len(columns)),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_case": int((y_train == 1).sum()),
                    "n_train_control": int((y_train == 0).sum()),
                    "n_test_case": int((y_test == 1).sum()),
                    "n_test_control": int((y_test == 0).sum()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_classification(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["analysis", "method", "q"], as_index=False)
    summary = grouped.agg(
        n=("split", "count"),
        auroc_mean=("auroc", "mean"),
        auroc_sd=("auroc", "std"),
        auroc_median=("auroc", "median"),
        auroc_q025=("auroc", lambda values: values.quantile(0.025)),
        auroc_q975=("auroc", lambda values: values.quantile(0.975)),
        mcc_mean=("mcc", "mean"),
        mcc_sd=("mcc", "std"),
        n_components_mean=("n_components", "mean"),
        inner_cv_auroc_mean=("inner_cv_auroc", "mean"),
        n_features_mean=("n_features", "mean"),
    )
    return summary


def aggregate_stable_vip(vip_rows: list[dict]) -> pd.DataFrame:
    vip = pd.DataFrame(vip_rows)
    group_columns = ["method", "q", "edge_index", "node_a", "node_b"]
    stable = vip.groupby(group_columns, as_index=False).agg(
        top_vip_count=("split", "nunique"),
        mean_vip_when_selected=("vip", "mean"),
        median_vip_when_selected=("vip", "median"),
        sd_vip_when_selected=("vip", "std"),
        mean_rank_when_selected=("rank_in_split", "mean"),
        best_rank=("rank_in_split", "min"),
        mean_case_frequency_when_selected=("selected_frequency_case_train", "mean"),
        mean_control_frequency_when_selected=(
            "selected_frequency_control_train",
            "mean",
        ),
        mean_case_minus_control_when_selected=("case_minus_control_train", "mean"),
        mean_n_components_when_selected=("n_components", "mean"),
        mean_n_features_when_selected=("n_features", "mean"),
    )
    stable["n_outer_splits"] = N_SPLITS
    stable["top_vip_frequency"] = stable["top_vip_count"] / float(N_SPLITS)
    stable["abs_mean_case_minus_control_when_selected"] = stable[
        "mean_case_minus_control_when_selected"
    ].abs()
    return stable.sort_values(
        [
            "method",
            "q",
            "top_vip_frequency",
            "mean_vip_when_selected",
            "abs_mean_case_minus_control_when_selected",
            "edge_index",
        ],
        ascending=[True, True, False, False, False, True],
    )


def network_from_edges(edge_frame: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in edge_frame.itertuples(index=False):
        graph.add_edge(
            row.node_a,
            row.node_b,
            weight=float(row.mean_vip_when_selected),
            stable_frequency=float(row.top_vip_frequency),
            case_control_difference=float(
                row.mean_case_minus_control_when_selected
            ),
        )
    return graph


def network_metric_row(
    graph: nx.Graph, method: str, q: float, top_n: int
) -> dict:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    components = list(nx.connected_components(graph)) if n_nodes else []
    giant_size = max((len(component) for component in components), default=0)
    degrees = [degree for _node, degree in graph.degree()]
    return {
        "method": method,
        "q": q,
        "network_top_n": top_n,
        "nodes": n_nodes,
        "edges": n_edges,
        "components": int(nx.number_connected_components(graph)) if n_nodes else 0,
        "giant_component_nodes": giant_size,
        "giant_frac": float(giant_size / n_nodes) if n_nodes else 0.0,
        "density": float(nx.density(graph)) if n_nodes > 1 else 0.0,
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "avg_clustering": float(nx.average_clustering(graph, weight="weight"))
        if n_nodes > 1
        else 0.0,
    }


def topology_rows(graph: nx.Graph, method: str, q: float, top_n: int) -> list[dict]:
    if graph.number_of_nodes() == 0:
        return []
    betweenness = nx.betweenness_centrality(graph, weight=None)
    closeness = nx.closeness_centrality(graph)
    clustering = nx.clustering(graph, weight="weight")
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "method": method,
                "q": q,
                "network_top_n": top_n,
                "node": node,
                "degree": int(graph.degree(node)),
                "weighted_degree": float(graph.degree(node, weight="weight")),
                "betweenness": float(betweenness[node]),
                "closeness": float(closeness[node]),
                "clustering": float(clustering[node]),
            }
        )
    rows.sort(key=lambda row: (-row["weighted_degree"], -row["degree"], row["node"]))
    for rank, row in enumerate(rows, start=1):
        row["weighted_degree_rank"] = rank
    return rows


def build_stable_network_outputs(
    stable: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    topology_all: list[dict] = []
    hub_candidates: list[dict] = []

    stable.to_csv(VIP_DIR / "stable_vip_edges_all.csv", index=False)
    for (method, q), group in stable.groupby(["method", "q"], sort=True):
        for top_n in VIP_NETWORK_TOP_NS:
            selected = group.head(top_n).copy()
            selected.to_csv(
                VIP_DIR / f"stable_vip_edges_{method}_{qtag(float(q))}_top{top_n}.csv",
                index=False,
            )
            graph = network_from_edges(selected)
            metric_rows.append(network_metric_row(graph, method, float(q), top_n))
            node_rows = topology_rows(graph, method, float(q), top_n)
            topology_all.extend(node_rows)
            if top_n == VIP_PRIMARY_TOP_N:
                hub_candidates.extend(node_rows[:HUB_TOP_N_PER_Q])

    metrics = pd.DataFrame(metric_rows)
    topology = pd.DataFrame(topology_all)
    metrics.to_csv(VIP_DIR / "stable_vip_network_metrics_by_q.csv", index=False)
    topology.to_csv(VIP_DIR / "stable_vip_topology_by_q.csv", index=False)

    metric_columns = [
        "nodes",
        "edges",
        "components",
        "giant_component_nodes",
        "giant_frac",
        "density",
        "avg_degree",
        "avg_clustering",
    ]
    across_q = (
        metrics.groupby(["method", "network_top_n"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    across_q.columns = [
        "_".join(str(part) for part in column if part != "").rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in across_q.columns
    ]
    across_q["n_q"] = len(Q_LIST)
    across_q.to_csv(
        VIP_DIR / "stable_vip_network_metrics_across_q.csv", index=False
    )

    hub_frame = pd.DataFrame(hub_candidates)
    if hub_frame.empty:
        hubs = hub_frame
    else:
        hubs = (
            hub_frame.groupby(["method", "node"], as_index=False)
            .agg(
                q_count=("q", "nunique"),
                mean_hub_rank=("weighted_degree_rank", "mean"),
                mean_degree=("degree", "mean"),
                mean_weighted_degree=("weighted_degree", "mean"),
                mean_betweenness=("betweenness", "mean"),
            )
            .sort_values(
                ["method", "q_count", "mean_hub_rank", "mean_weighted_degree"],
                ascending=[True, False, True, False],
            )
        )
    hubs.to_csv(VIP_DIR / "stable_hubs_across_q.csv", index=False)
    return metrics, across_q, topology, hubs


def write_metadata(
    x: np.ndarray, y: np.ndarray, sample_ids: list[str], elapsed_seconds: float
) -> None:
    metadata = {
        "workflow": "new_sweet_topq_outer_mc_inner_plslr_auc_stable_vip",
        "seed": SEED,
        "n_outer_splits": N_SPLITS,
        "test_size": TEST_SIZE,
        "n_permutation_controls": N_PERM if RUN_PERM else 0,
        "q_list": Q_LIST,
        "q_role": "prespecified_sparsity_sensitivity_series_not_test_selected",
        "methods": METHODS,
        "output_directory": str(OUTDIR),
        "dependence_measure_policy": "method-matched_for_edges_and_sample_weights",
        "sweet_k": K_SWEET,
        "sweet_epsilon": EPS_SWEET,
        "max_pls_components": MAX_N_COMP,
        "inner_folds": INNER_FOLDS,
        "component_selection": "mean_inner_auroc_of_complete_plslr_pipeline",
        "outer_raw_standardization": "outer_training_only",
        "inner_binary_feature_standardization": "inner_training_fold_only",
        "pls_scale": False,
        "logistic": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": None,
            "max_iter": 2000,
            "classification_threshold": 0.5,
        },
        "permutation_control": "labels_permuted_separately_within_fixed_train_and_test_partitions",
        "vip_candidate_top_n_per_split": VIP_CANDIDATE_TOP_N,
        "vip_network_top_n_sensitivity": VIP_NETWORK_TOP_NS,
        "vip_primary_top_n": VIP_PRIMARY_TOP_N,
        "hub_top_n_per_q": HUB_TOP_N_PER_Q,
        "n_samples": int(x.shape[0]),
        "n_cases": int((y == 1).sum()),
        "n_controls": int((y == 0).sum()),
        "n_metabolites": int(x.shape[1]),
        "n_candidate_edges": int(x.shape[1] * (x.shape[1] - 1) // 2),
        "n_sample_ids": len(sample_ids),
        "input": RAW_INPUT.name,
        "input_archive_sha256": file_sha256(RAW_INPUT),
        "input_scientific_content_sha256": input_content_sha256(RAW_INPUT),
        "source_sha256": {
            Path(__file__).name: file_sha256(Path(__file__)),
            "sweet_plslr_common.py": file_sha256(BASE / "sweet_plslr_common.py"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "networkx": nx.__version__,
        },
        "elapsed_seconds": elapsed_seconds,
    }
    (OUTDIR / "new_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    started = time.time()
    x, y, metabolite_names, sample_ids = load_raw_with_names()
    outer_splits = list(
        StratifiedShuffleSplit(
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            random_state=SEED,
        ).split(x, y)
    )

    print(
        f"[real] {N_SPLITS} outer splits; methods={METHODS}; q={Q_LIST}",
        flush=True,
    )
    real_nested = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(run_one_real_split)(
            split_number,
            train_idx,
            test_idx,
            x,
            y,
            metabolite_names,
        )
        for split_number, (train_idx, test_idx) in enumerate(outer_splits, start=1)
    )
    result_rows = [row for results, _vip, _tuning in real_nested for row in results]
    vip_rows = [row for _results, vip, _tuning in real_nested for row in vip]
    tuning_rows = [
        row for _results, _vip, tuning in real_nested for row in tuning
    ]

    if RUN_PERM and N_PERM > 0:
        print(f"[permutation control] {N_PERM} paired runs", flush=True)
        permutation_jobs = []
        for permutation_number in range(1, N_PERM + 1):
            train_idx, test_idx = outer_splits[(permutation_number - 1) % N_SPLITS]
            permutation_jobs.append((permutation_number, train_idx, test_idx))
        permutation_nested = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
            delayed(run_one_permutation)(
                permutation_number, train_idx, test_idx, x, y
            )
            for permutation_number, train_idx, test_idx in permutation_jobs
        )
        result_rows.extend(row for group in permutation_nested for row in group)

    result_fields = [
        "analysis",
        "split",
        "method",
        "q",
        "auroc",
        "mcc",
        "n_components",
        "inner_cv_auroc",
        "n_features",
        "n_train",
        "n_test",
        "n_train_case",
        "n_train_control",
        "n_test_case",
        "n_test_control",
    ]
    write_csv(OUTDIR / "new_plslr_results.csv", result_rows, result_fields)
    write_csv(
        OUTDIR / "new_component_tuning_results.csv",
        tuning_rows,
        [
            "split",
            "method",
            "q",
            "n_components",
            "mean_inner_auroc",
            "sd_inner_auroc",
            "n_inner_folds",
            "selected",
        ],
    )

    summary = summarize_classification(result_rows)
    summary.to_csv(OUTDIR / "new_plslr_summary.csv", index=False)

    write_csv(
        VIP_DIR / "cv_top_vip_edges_per_split.csv",
        vip_rows,
        [
            "split",
            "method",
            "q",
            "rank_in_split",
            "edge_index",
            "node_a",
            "node_b",
            "vip",
            "selected_frequency_case_train",
            "selected_frequency_control_train",
            "case_minus_control_train",
            "n_components",
            "n_features",
        ],
    )
    stable = aggregate_stable_vip(vip_rows)
    build_stable_network_outputs(stable)
    if RUN_PERM and N_PERM > 0:
        generate_figure5(OUTDIR, FIG_DIR)
    else:
        print("[figures] Figure 5 skipped because permutation controls were disabled", flush=True)
    write_metadata(x, y, sample_ids, time.time() - started)
    print(f"[done] outputs written to {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
