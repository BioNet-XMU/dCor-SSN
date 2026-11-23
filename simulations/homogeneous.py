"""Homogeneous simulation generator, SWEET estimators and Figure 2A/B plotting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from numpy.random import default_rng
from openpyxl import load_workbook
from openpyxl.styles import Font
from scipy.stats import rankdata

class Config:
    N_SAMPLES = int(os.environ.get("SIM_N", 200))
    MODULE_SIZE = int(os.environ.get("SIM_MODULE_SIZE", 10))
    N_MODULES = 7
    M = MODULE_SIZE * N_MODULES
    LINEAR_COEF = float(os.environ.get("SIM_LINEAR_COEF", 0.7))
    LINEAR_INTERCEPT = float(os.environ.get("SIM_LINEAR_INTERCEPT", 1.0))
    SINE_FREQ = float(os.environ.get("SIM_SINE_FREQ", 2.0))
    LOG_SHIFT = float(os.environ.get("SIM_LOG_SHIFT", 1.0))
    SIGMA_GRID = tuple(
        float(x.strip())
        for x in os.environ.get("SIM_SIGMA_GRID", "0,0.2,0.4,0.6,0.8,1.0").split(",")
        if x.strip()
    )
    R = int(os.environ.get("SIM_R", 100))
    BASE_SEED = int(os.environ.get("SIM_BASE_SEED", 42))
    OUTPUT_PREFIX = os.environ.get("SIM_OUTPUT_PREFIX", "sim_ssn_sweet_edge_rank_log")
    SELF_TEST = os.environ.get("SIM_SELF_TEST", "0") not in ("0", "", "false", "False")
    # SWEET 尺度因子参数
    K = float(os.environ.get("SIM_K", 0.1))
    EPS = float(os.environ.get("SIM_EPS", 0.01))


cfg = Config()
OUT_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'section_a_original'


METHOD_ORDER = ["dcor", "pearson", "spearman"]
METHOD_LABELS = {"dcor": "dCor-SWEET", "pearson": "Pearson-SWEET", "spearman": "Spearman-SWEET"}
COLORS = {"dcor": "#D55E00", "pearson": "#0072B2", "spearman": "#009E73"}
MARKERS = {"dcor": "o", "pearson": "s", "spearman": "^"}
LINESTYLES = {"dcor": "-", "pearson": "-", "spearman": "-"}

B_RELATIONS = {
    "linear": {"target_module": 1, "label": "Linear: M1-M2"},
    "quadratic": {"target_module": 2, "label": "Quadratic: M1-M3"},
    "absolute": {"target_module": 3, "label": "Absolute: M1-M4"},
    "sine": {"target_module": 4, "label": "Sine(2x): M1-M5"},
    "logarithmic": {"target_module": 5, "label": "Logarithmic: M1-M6"},
}
B_RELATION_ORDER = ["linear", "quadratic", "absolute", "sine", "logarithmic"]

GROUP_ORDER = ["related_all", "null_m7", "null_cross_index"]
GROUP_LABELS = {
    "related_all": "Related edges (all)",
    "null_m7": "Null: M7 edges",
    "null_cross_index": "Null: cross-index edges",
}
GROUP_BETTER = {
    "related_all": "lower",
    "null_m7": "higher",
    "null_cross_index": "higher",
}


MODULE_DEFINITIONS = [
    {"module": "M1", "definition": "x_j^(1) ~ N(0, 1), j=1,...,10; 10 independent base variables"},
    {"module": "M2", "definition": "x_j^(2) = 0.7 * x_j^(1) + 1"},
    {"module": "M3", "definition": "x_j^(3) = (x_j^(1))^2"},
    {"module": "M4", "definition": "x_j^(4) = |x_j^(1)|"},
    {"module": "M5", "definition": "x_j^(5) = sin(2 * x_j^(1))"},
    {"module": "M6", "definition": "x_j^(6) = log(max(0, x_j^(1) + a) + 1), a=1.0"},
    {"module": "M7", "definition": "x_j^(7) ~ N(0, 1), independent null module"},
]

METHOD_DEFINITIONS = [
    {"method": "dcor", "definition": "distance correlation; SWEET single-sample edge weight, |.| for ranking"},
    {"method": "pearson", "definition": "Pearson correlation; SWEET single-sample edge weight, |.| for ranking"},
    {"method": "spearman", "definition": "Spearman correlation; SWEET single-sample edge weight, |.| for ranking"},
]


# --------------------------------------------------------------------------- #
# 数据生成
# --------------------------------------------------------------------------- #
def module_indices(module_id: int) -> np.ndarray:
    start = module_id * cfg.MODULE_SIZE
    end = start + cfg.MODULE_SIZE
    return np.arange(start, end)


def generate_clean_dataset(seed: int) -> np.ndarray:
    rng = default_rng(seed)
    n = cfg.N_SAMPLES
    mod = cfg.MODULE_SIZE
    x = np.zeros((n, cfg.M), dtype=float)

    base = rng.normal(0, 1, size=(n, mod))
    x[:, module_indices(0)] = base
    x[:, module_indices(1)] = cfg.LINEAR_COEF * base + cfg.LINEAR_INTERCEPT
    x[:, module_indices(2)] = base ** 2
    x[:, module_indices(3)] = np.abs(base)
    x[:, module_indices(4)] = np.sin(cfg.SINE_FREQ * base)
    x[:, module_indices(5)] = np.log1p(np.maximum(0.0, base + cfg.LOG_SHIFT))
    x[:, module_indices(6)] = rng.normal(0, 1, size=(n, mod))
    return x


def add_feature_scaled_noise(x: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = default_rng(seed)
    noisy = x.copy()
    stds = np.std(noisy, axis=0, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    noisy += rng.normal(0, sigma, size=noisy.shape) * stds[None, :]
    return np.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)


def generate_dataset(seed: int, sigma: float) -> np.ndarray:
    clean = generate_clean_dataset(seed)
    return add_feature_scaled_noise(clean, sigma=sigma, seed=seed + 1_000_000)


# --------------------------------------------------------------------------- #
# Ground truth 边分组
# --------------------------------------------------------------------------- #
def build_edge_lookup(triu_idx: Tuple[np.ndarray, np.ndarray]) -> Dict[Tuple[int, int], int]:
    rows, cols = triu_idx
    return {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}


def module_of(node: int) -> int:
    return node // cfg.MODULE_SIZE


def index_of(node: int) -> int:
    return node % cfg.MODULE_SIZE


def classify_edges(edge_lookup: Dict[Tuple[int, int], int]) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {g: [] for g in GROUP_ORDER}
    m7_mod = 6
    for (i, j), k in edge_lookup.items():
        mi, mj = module_of(i), module_of(j)
        if mi == m7_mod or mj == m7_mod:
            groups["null_m7"].append(k)
        elif index_of(i) == index_of(j):
            groups["related_all"].append(k)
        else:
            groups["null_cross_index"].append(k)
    return {g: np.array(sorted(v), dtype=int) for g, v in groups.items()}


def build_b_edges(edge_lookup: Dict[Tuple[int, int], int]) -> Dict[str, np.ndarray]:
    base_nodes = module_indices(0)
    b_edges: Dict[str, List[int]] = {}
    for relation in B_RELATION_ORDER:
        target_nodes = module_indices(B_RELATIONS[relation]["target_module"])
        idxs = []
        for base_node, target_node in zip(base_nodes, target_nodes):
            i, j = sorted((int(base_node), int(target_node)))
            idxs.append(edge_lookup[(i, j)])
        b_edges[relation] = np.array(idxs, dtype=int)
    return b_edges


# --------------------------------------------------------------------------- #
# 全网络边权（总网络 / 任意样本子集）
# --------------------------------------------------------------------------- #
def pearson_edges(data: np.ndarray, triu_idx) -> np.ndarray:
    mat = np.corrcoef(data, rowvar=False)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat[triu_idx]


def spearman_edges(data: np.ndarray, triu_idx) -> np.ndarray:
    ranked = np.apply_along_axis(rankdata, 0, data)
    return pearson_edges(ranked, triu_idx)


def dcor_edges(data: np.ndarray, triu_idx) -> np.ndarray:
    weights, _ = _dcor_edges_with_cache(data)
    return weights[triu_idx]


# --------------------------------------------------------------------------- #
# SWEET 单样本网络（增量留一提速 + 尺度因子）
# --------------------------------------------------------------------------- #
#
# 对每条边 e、每个样本 q：
#   w_q(e) = (K * S_q) * (N-1) * (w_all(e) - w_minus_q(e)) + w_minus_q(e)
# 排序时用 |w_q(e)|。
#
# S_q 由样本间底层依赖矩阵 min-max 归一化得到，跟随底层度量。


def _pearson_stats(data: np.ndarray):
    n = data.shape[0]
    Sx = data.sum(axis=0)
    Sxx = (data * data).sum(axis=0)
    Sxy = data.T @ data
    return n, Sx, Sxx, Sxy


def _pearson_from_stats(n, Sx, Sxx, Sxy, triu_idx) -> np.ndarray:
    mean = Sx / n
    cov = Sxy - n * np.outer(mean, mean)
    var = Sxx - n * mean * mean
    denom = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr[triu_idx]


def _pearson_full_matrix(data: np.ndarray) -> np.ndarray:
    mat = np.corrcoef(data, rowvar=False)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def _spearman_full_matrix(data: np.ndarray) -> np.ndarray:
    ranked = np.apply_along_axis(rankdata, 0, data)
    return _pearson_full_matrix(ranked)


def _sample_weight(data: np.ndarray, dep_matrix_fn) -> np.ndarray:
    """SWEET 样本权重 S_q。

    dep_matrix_fn 接受 (N, P) 返回 (P, P) 变量依赖矩阵；
    这里用 dep_matrix_fn(data.T) 得到 (N, N) 样本间依赖矩阵。
    """
    n = data.shape[0]
    R = np.asarray(dep_matrix_fn(data.T), dtype=np.float64)  # (N, N)
    mu = (R.sum(axis=1) - np.diag(R)) / (n - 1)
    lo = float(mu.min())
    hi = float(mu.max())
    return (mu - lo + cfg.EPS) / (hi - lo + cfg.EPS)


def pearson_sweet(data: np.ndarray, triu_idx) -> np.ndarray:
    """返回 (N, E) 的 SWEET 单样本边权矩阵（未取绝对值）。"""
    n, Sx, Sxx, Sxy = _pearson_stats(data)
    w_all = _pearson_from_stats(n, Sx, Sxx, Sxy, triu_idx)
    E = w_all.size
    S = _sample_weight(data, _pearson_full_matrix)
    ssn = np.empty((n, E), dtype=float)
    m = n - 1
    for q in range(n):
        xq = data[q]
        Sx_q = Sx - xq
        Sxx_q = Sxx - xq * xq
        Sxy_q = Sxy - np.outer(xq, xq)
        w_minus = _pearson_from_stats(n - 1, Sx_q, Sxx_q, Sxy_q, triu_idx)
        ssn[q] = (cfg.K * S[q]) * m * (w_all - w_minus) + w_minus
    return ssn


def spearman_sweet(data: np.ndarray, triu_idx) -> np.ndarray:
    """Spearman SWEET 留一（稳妥版）：去掉 q 后重排秩，再套 Pearson 增量。"""
    n = data.shape[0]
    ranked_all = np.apply_along_axis(rankdata, 0, data)
    na, Sxa, Sxxa, Sxya = _pearson_stats(ranked_all)
    w_all = _pearson_from_stats(na, Sxa, Sxxa, Sxya, triu_idx)
    E = w_all.size
    S = _sample_weight(data, _spearman_full_matrix)
    ssn = np.empty((n, E), dtype=float)
    m = n - 1
    for q in range(n):
        sub = np.delete(data, q, axis=0)
        ranked_sub = np.apply_along_axis(rankdata, 0, sub)
        ns, Sxs, Sxxs, Sxys = _pearson_stats(ranked_sub)
        w_minus = _pearson_from_stats(ns, Sxs, Sxxs, Sxys, triu_idx)
        ssn[q] = (cfg.K * S[q]) * m * (w_all - w_minus) + w_minus
    return ssn


def _dcor_edges_with_cache(data: np.ndarray):
    values = data.T.astype(float)
    M, n = values.shape
    dist = np.abs(values[:, :, None] - values[:, None, :])
    mat = _dcor_from_dist(dist, n)
    return mat, dist


def _dcor_from_dist(dist: np.ndarray, n: int) -> np.ndarray:
    centered = (
        dist
        - dist.mean(axis=2, keepdims=True)
        - dist.mean(axis=1, keepdims=True)
        + dist.mean(axis=(1, 2))[:, None, None]
    )
    flat = centered.reshape(dist.shape[0], -1)
    dcov2 = flat @ flat.T / float(n * n)
    dvar2 = np.diag(dcov2)
    denom = np.sqrt(np.outer(dvar2, dvar2))
    with np.errstate(divide="ignore", invalid="ignore"):
        dcor_sqr = np.divide(dcov2, denom, out=np.zeros_like(dcov2), where=denom > 0)
    mat = np.sqrt(np.clip(dcor_sqr, 0, 1))
    return mat


def _centered_inner_from_sufficient(raw_inner, row_inner, total, n):
    """由充分统计量算双中心化距离内积 <A,B>（对所有变量对同时计算）。

    <A,B> = sum(a*b) - (2/n)*sum(rowsum(a)*rowsum(b)) + total(a)*total(b)/n^2
    """
    nf = float(n)
    M = raw_inner - (2.0 / nf) * row_inner + np.outer(total, total) / (nf * nf)
    return 0.5 * (M + M.T)


def _dcor_from_centered_inner(M):
    """把双中心化距离内积矩阵转成 dCor 矩阵（对角 = 1）。"""
    M = np.asarray(M, dtype=np.float64)
    diag = np.clip(np.diag(M), 0.0, None)
    denom = np.sqrt(np.outer(diag, diag))
    R2 = np.zeros_like(M)
    valid = denom > 0.0
    R2[valid] = M[valid] / denom[valid]
    R2 = np.clip(R2, 0.0, 1.0)
    R = np.sqrt(R2)
    return 0.5 * (R + R.T)


def _dcor_edges_with_cache(data: np.ndarray):
    values = data.T.astype(float)
    M, n = values.shape
    dist = np.abs(values[:, :, None] - values[:, None, :])
    mat = _dcor_from_dist(dist, n)
    return mat, dist


def _dcor_full_matrix(data: np.ndarray) -> np.ndarray:
    """全量 dCor 矩阵（供 _sample_weight 使用）。"""
    values = data.T.astype(float)
    M, n = values.shape
    dist = np.abs(values[:, :, None] - values[:, None, :])
    return _dcor_from_dist(dist, n)


def dcor_sweet(data: np.ndarray, triu_idx) -> np.ndarray:
    """dCor SWEET 留一（充分统计量增量版，O(P^2 N^2)）。

    距离矩阵只算一次；留一时用充分统计量做 O(P^2 N) 增量更新，
    而非切片后重新双中心化。SWEET 组装公式不变。
    """
    values = data.T.astype(float)            # (P, N)
    P, n = values.shape
    dist = np.abs(values[:, :, None] - values[:, None, :])   # (P, N, N)

    # 全量充分统计量
    row_sum = dist.sum(axis=2)               # (P, N)
    total = row_sum.sum(axis=1)              # (P,)
    F = dist.reshape(P, -1)                  # (P, N^2)
    raw_inner = F @ F.T                      # S_uv
    row_inner = row_sum @ row_sum.T          # R_uv

    M_full = _centered_inner_from_sufficient(raw_inner, row_inner, total, n)
    mat_all = _dcor_from_centered_inner(M_full)
    w_all = mat_all[triu_idx]
    E = w_all.size
    S = _sample_weight(data, _dcor_full_matrix)
    ssn = np.empty((n, E), dtype=float)
    m = n - 1

    for q in range(n):
        H = dist[:, q, :]                    # (P, N)
        raw_inner_loo = raw_inner - 2.0 * (H @ H.T)
        total_loo = total - 2.0 * row_sum[:, q]

        reduced_rows = row_sum - dist[:, :, q]
        reduced_rows[:, q] = 0.0
        row_inner_loo = reduced_rows @ reduced_rows.T

        M_loo = _centered_inner_from_sufficient(raw_inner_loo, row_inner_loo, total_loo, m)
        mat_minus = _dcor_from_centered_inner(M_loo)
        w_minus = mat_minus[triu_idx]
        ssn[q] = (cfg.K * S[q]) * m * (w_all - w_minus) + w_minus
    return ssn


SWEET_ESTIMATORS = {
    "pearson": pearson_sweet,
    "spearman": spearman_sweet,
    "dcor": dcor_sweet,
}


# --------------------------------------------------------------------------- #
# 排名
# --------------------------------------------------------------------------- #
def rank_edges_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=int)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def group_mean_ranks(ranks: np.ndarray, edge_idx: np.ndarray) -> float:
    return float(ranks[edge_idx].mean())


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
def run_one(seed: int, sigma: float, edge_groups, b_edges, triu_idx) -> List[Dict]:
    data = generate_dataset(seed, sigma)
    n = data.shape[0]
    rows = []

    for method in METHOD_ORDER:
        ssn = SWEET_ESTIMATORS[method](data, triu_idx)
        abs_ssn = np.abs(ssn)

        per_sample_group = {g: np.empty(n) for g in GROUP_ORDER}
        per_sample_b = {r: np.empty(n) for r in B_RELATION_ORDER}
        for q in range(n):
            ranks = rank_edges_desc(abs_ssn[q])
            for g in GROUP_ORDER:
                per_sample_group[g][q] = group_mean_ranks(ranks, edge_groups[g])
            for r in B_RELATION_ORDER:
                per_sample_b[r][q] = group_mean_ranks(ranks, b_edges[r])

        for g in GROUP_ORDER:
            rows.append({
                "seed": seed, "sigma": sigma, "method": method,
                "method_label": METHOD_LABELS[method],
                "kind": "group", "name": g, "label": GROUP_LABELS[g],
                "better": GROUP_BETTER[g],
                "rep_mean_rank": float(per_sample_group[g].mean()),
                "total_edges": int(abs_ssn.shape[1]),
                "n_edges": int(edge_groups[g].size),
            })
        for r in B_RELATION_ORDER:
            rows.append({
                "seed": seed, "sigma": sigma, "method": method,
                "method_label": METHOD_LABELS[method],
                "kind": "b_relation", "name": r, "label": B_RELATIONS[r]["label"],
                "better": "lower",
                "rep_mean_rank": float(per_sample_b[r].mean()),
                "total_edges": int(abs_ssn.shape[1]),
                "n_edges": int(b_edges[r].size),
            })
    return rows


def run_simulation() -> pd.DataFrame:
    triu_idx = np.triu_indices(cfg.M, k=1)
    edge_lookup = build_edge_lookup(triu_idx)
    edge_groups = classify_edges(edge_lookup)
    b_edges = build_b_edges(edge_lookup)

    assert edge_groups["related_all"].size == 150, edge_groups["related_all"].size
    assert edge_groups["null_m7"].size == 645, edge_groups["null_m7"].size
    assert edge_groups["null_cross_index"].size == 1620, edge_groups["null_cross_index"].size

    rows = []
    for sigma in cfg.SIGMA_GRID:
        print(f"\nSigma = {sigma}")
        for rep in range(cfg.R):
            seed = cfg.BASE_SEED + rep
            print(f"  repeat {rep + 1}/{cfg.R} (seed={seed})...", end=" ", flush=True)
            rows.extend(run_one(seed, sigma, edge_groups, b_edges, triu_idx))
            print("done", flush=True)
    return pd.DataFrame(rows)


def summarize(rep_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rep_df.groupby(["kind", "name", "label", "better", "sigma", "method", "method_label"])
        .agg(
            rank_mean=("rep_mean_rank", "mean"),
            rank_std=("rep_mean_rank", "std"),
            rank_sem=("rep_mean_rank", lambda s: s.std(ddof=1) / np.sqrt(s.size) if s.size > 1 else 0.0),
            rank_median=("rep_mean_rank", "median"),
            n_repeats=("rep_mean_rank", "size"),
            n_edges=("n_edges", "first"),
            total_edges=("total_edges", "first"),
        )
        .reset_index()
    )
    return summary


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
#
# 主文 Figure 2 版式（与 make_figure2_manuscript_preview.py 一致）：
#   A：全部 150 条相关边的归一化平均秩（%）随噪声 σ 的变化，误差棒 = SEM（跨重复）；
#   B：五类 M1 锚定关系在 6 个噪声水平上的平均值（点）与最小–最大范围（竖线）。
# 纵轴 = rank_mean / total_edges × 100；虚线 = 随机排序期望 (E+1)/(2E) × 100。
# 图中方法标签用 “-SSN”；结果表中的 method_label 仍保留 “-SWEET”，不受影响。

FIG_DIR = OUT_DIR / 'figures'
FIG_STEM = "Figure2_homogeneous_edge_ranking"
FIG_LABELS = {"dcor": "dCor-SSN", "pearson": "Pearson-SSN", "spearman": "Spearman-SSN"}
FIG_MARKERS = {"dcor": "o", "pearson": "s", "spearman": "^"}
FIG_LINESTYLES = {"dcor": "-", "pearson": (0, (5, 2.5)), "spearman": (0, (1.5, 1.8))}
FIG_RELATION_TICKS = {
    "linear": "Linear",
    "quadratic": "Quadratic",
    "absolute": "Absolute",
    "sine": "Sine",
    "logarithmic": "Logarithmic",
}
RANDOM_COLOR = "#8C8C8C"
RANDOM_LINESTYLE = (0, (5, 5))
Y_TICKS = (0, 10, 20, 30, 40, 50, 60)


def set_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _pct(values, total_edges) -> np.ndarray:
    return np.asarray(values, dtype=float) / float(total_edges) * 100.0


def figure2_plot_data(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """整理 Figure 2 的绘图数据：A 面板逐 σ 数据，B 面板跨 σ 的均值/最小/最大。"""
    df = summary_df.copy()
    df["sigma"] = df["sigma"].astype(float)
    df["pct"] = _pct(df["rank_mean"], 1.0) / df["total_edges"].to_numpy(dtype=float)
    df["sem_pct"] = _pct(df["rank_sem"].fillna(0.0), 1.0) / df["total_edges"].to_numpy(dtype=float)

    panel_a = (
        df[(df["kind"] == "group") & (df["name"] == "related_all")]
        .sort_values(["method", "sigma"])
        .reset_index(drop=True)
    )
    panel_b = (
        df[df["kind"] == "b_relation"]
        .groupby(["name", "method"])["pct"]
        .agg(mean="mean", minimum="min", maximum="max")
        .reset_index()
    )
    total_edges = int(df["total_edges"].iloc[0])
    random_pct = (total_edges + 1.0) / 2.0 / total_edges * 100.0
    return panel_a, panel_b, random_pct


def save_figure2_plot_data(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> Path:
    rows = []
    for method in METHOD_ORDER:
        for _, r in panel_a[panel_a["method"] == method].iterrows():
            rows.append({
                "panel": "A", "relationship": "all_related", "method": method,
                "method_label": FIG_LABELS[method], "sigma": f"{r['sigma']:.1f}",
                "normalized_mean_rank_pct": f"{r['pct']:.6f}",
                "sem_pct": f"{r['sem_pct']:.6f}",
                "across_sigma_min_pct": "", "across_sigma_max_pct": "",
            })
    for relation in B_RELATION_ORDER:
        for method in METHOD_ORDER:
            r = panel_b[(panel_b["name"] == relation) & (panel_b["method"] == method)].iloc[0]
            rows.append({
                "panel": "B", "relationship": relation, "method": method,
                "method_label": FIG_LABELS[method], "sigma": "mean_across_sigma",
                "normalized_mean_rank_pct": f"{r['mean']:.6f}", "sem_pct": "",
                "across_sigma_min_pct": f"{r['minimum']:.6f}",
                "across_sigma_max_pct": f"{r['maximum']:.6f}",
            })
    out_csv = FIG_DIR / f"{FIG_STEM}_plot_data.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def plot_figure2(summary_df: pd.DataFrame, top5: pd.DataFrame | None = None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel_a, panel_b, random_pct = figure2_plot_data(summary_df)
    save_figure2_plot_data(panel_a, panel_b)

    set_plot_style()
    if top5 is None:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    else:
        fig = plt.figure(figsize=(7.2, 5.2))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.76, wspace=0.28)
        ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[1, :])
        groups = ["quadratic", "absolute"]
        panel_c = top5.query("sigma == 1 and method == 'dcor' and name in @groups")
        panel_c.to_csv(FIG_DIR / "Figure2C_top5_plot_data.csv", index=False)
        y = np.arange(len(groups))
        for frame, offset, color, label in (
            ("population", -0.17, "#BF5600", "Cohort dCor"),
            ("SWEET", 0.17, "#F5D0AC", "dCor-SWEET"),
        ):
            group = panel_c[panel_c["framework"].eq(frame)].set_index("name").loc[groups]
            mean, sem = group["top5_rate"] * 100, group["top5_rate_sem"] * 100
            ax_c.barh(y + offset, mean, xerr=sem, height=0.29, color=color, label=label,
                      edgecolor="#A84A00", linewidth=0.5, error_kw={"elinewidth": 0.7, "capsize": 2})
            for pos, value, err in zip(y + offset, mean, sem):
                ax_c.text(value + err + 1, pos, f"{value:.1f}%", va="center", fontsize=7)
        ax_c.set(yticks=y, yticklabels=["Quadratic", "Absolute value"], xlim=(0, 68),
                 xlabel="Share of each edge group in top 121 (%)")
        ax_c.invert_yaxis()
        ax_c.spines[["top", "right"]].set_visible(False)
        ax_c.set_title("C      Top 5% at σ = 1.0", loc="left", fontsize=8)
        ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=2)

    # Panel A：全部相关边的边排序随噪声的变化
    for method in METHOD_ORDER:
        sub = panel_a[panel_a["method"] == method]
        ax_a.errorbar(
            sub["sigma"].to_numpy(), sub["pct"].to_numpy(), yerr=sub["sem_pct"].to_numpy(),
            color=COLORS[method], marker=FIG_MARKERS[method], linestyle=FIG_LINESTYLES[method],
            markersize=4.5, linewidth=1.45, capsize=1.7, capthick=0.7, elinewidth=0.7,
            label=FIG_LABELS[method], zorder=3,
        )
    ax_a.axhline(random_pct, color=RANDOM_COLOR, linewidth=0.9, linestyle=RANDOM_LINESTYLE, zorder=1)
    ax_a.set(
        xlim=(min(cfg.SIGMA_GRID) - 0.035, max(cfg.SIGMA_GRID) + 0.035),
        ylim=(0, 60), xticks=cfg.SIGMA_GRID, yticks=Y_TICKS,
        xlabel="Noise level, σ", ylabel="Normalized mean rank (%)",
    )
    ax_a.grid(axis="y", color="#E8E8E8", linewidth=0.5)

    # Panel B：五类关系跨 σ 的均值（点）与最小–最大范围（竖线）
    positions = np.arange(len(B_RELATION_ORDER), dtype=float)
    offsets = {"dcor": -0.22, "pearson": 0.0, "spearman": 0.22}
    for method in METHOD_ORDER:
        for pos, relation in zip(positions, B_RELATION_ORDER):
            r = panel_b[(panel_b["name"] == relation) & (panel_b["method"] == method)].iloc[0]
            x = pos + offsets[method]
            ax_b.vlines(x, r["minimum"], r["maximum"], color=COLORS[method],
                        linewidth=1.45, alpha=0.70, zorder=2)
            ax_b.scatter(x, r["mean"], color=COLORS[method], marker=FIG_MARKERS[method],
                         s=25, edgecolor="white", linewidth=0.35, zorder=3)
    ax_b.axhline(random_pct, color=RANDOM_COLOR, linewidth=0.9, linestyle=RANDOM_LINESTYLE, zorder=1)
    ax_b.set(
        xlim=(-0.55, len(B_RELATION_ORDER) - 0.45), ylim=(0, 60),
        xticks=positions, xticklabels=[FIG_RELATION_TICKS[r] for r in B_RELATION_ORDER],
        yticks=Y_TICKS, xlabel="Relationship type",
    )
    ax_b.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    if top5 is None:
        ax_b.tick_params(axis="y", left=False, labelleft=False)

    for ax, panel in ((ax_a, "A"), (ax_b, "B")):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.115 if top5 is None else 0.0, 1.03 if top5 is None else 1.08, panel, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left")

    handles = [
        Line2D([0], [0], color=COLORS[m], marker=FIG_MARKERS[m], linestyle=FIG_LINESTYLES[m],
               linewidth=1.45, markersize=4.3, label=FIG_LABELS[m])
        for m in METHOD_ORDER
    ]
    handles.append(Line2D([0], [0], color=RANDOM_COLOR, linewidth=0.9,
                          linestyle=RANDOM_LINESTYLE, label="Random ranking (50%)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01 if top5 is None else 0.40), ncol=4,
               frameon=False, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.965, bottom=0.235 if top5 is None else 0.09)
    stem = FIG_STEM if top5 is None else "Figure2_rank_and_top5"

    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=360 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 结果导出
# --------------------------------------------------------------------------- #
def config_dataframe() -> pd.DataFrame:
    rows = [
        {"parameter": "MODEL", "value": "SWEET single-sample network (SSN with scale factor)"},
        {"parameter": "N_SAMPLES", "value": cfg.N_SAMPLES},
        {"parameter": "MODULE_SIZE", "value": cfg.MODULE_SIZE},
        {"parameter": "N_MODULES", "value": cfg.N_MODULES},
        {"parameter": "N_VARIABLES", "value": cfg.M},
        {"parameter": "TOTAL_EDGES", "value": cfg.M * (cfg.M - 1) // 2},
        {"parameter": "RELATED_EDGES", "value": 150},
        {"parameter": "NULL_M7_EDGES", "value": 645},
        {"parameter": "NULL_CROSS_INDEX_EDGES", "value": 1620},
        {"parameter": "LINEAR_COEF", "value": cfg.LINEAR_COEF},
        {"parameter": "LINEAR_INTERCEPT", "value": cfg.LINEAR_INTERCEPT},
        {"parameter": "SINE_FREQ", "value": cfg.SINE_FREQ},
        {"parameter": "LOG_SHIFT", "value": cfg.LOG_SHIFT},
        {"parameter": "LOG_TRANSFORM", "value": "log(max(0, x + LOG_SHIFT) + 1); a=1.0 -> curved regime + soft low-value truncation"},
        {"parameter": "SIGMA_GRID", "value": ",".join(map(str, cfg.SIGMA_GRID))},
        {"parameter": "R", "value": cfg.R},
        {"parameter": "BASE_SEED", "value": cfg.BASE_SEED},
        {"parameter": "K", "value": cfg.K},
        {"parameter": "EPS", "value": cfg.EPS},
        {"parameter": "SSN_FORMULA", "value": "w_q = (K*S_q)*(N-1)*(w_all - w_minus_q) + w_minus_q; |w_q| for ranking"},
        {"parameter": "SAMPLE_WEIGHT", "value": "S_q = sample-sample dependency (follows method) -> row mean -> min-max + EPS"},
        {"parameter": "NOISE_MODEL", "value": "z = z + epsilon * sd(z), epsilon ~ N(0, sigma^2)"},
        {"parameter": "AGGREGATION", "value": "per-sample group mean rank -> per-repeat mean over N samples -> mean+SEM over R repeats"},
        {"parameter": "EDGE_RANKING", "value": "within each sample's own network; all 2415 edges by |weight| desc; rank 1 = strongest"},
    ]
    return pd.DataFrame(rows)


def format_workbook(path: Path) -> None:
    wb = load_workbook(path)
    font = Font(name="Arial", size=10)
    header_font = Font(name="Arial", size=10, bold=True)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for row in ws.iter_rows():
            for cell in row:
                cell.font = header_font if cell.row == 1 else font
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells[:200]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 55)
    wb.save(path)


def save_results(rep_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{cfg.OUTPUT_PREFIX}_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="rank_summary", index=False)
        rep_df.to_excel(writer, sheet_name="rank_by_repeat", index=False)
        pd.DataFrame(MODULE_DEFINITIONS).to_excel(writer, sheet_name="module_definitions", index=False)
        pd.DataFrame(METHOD_DEFINITIONS).to_excel(writer, sheet_name="method_definitions", index=False)
        config_dataframe().to_excel(writer, sheet_name="config", index=False)
    format_workbook(out_path)
    print(f"Results saved to: {out_path}")
    return out_path


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n============= Mean rank summary (per-repeat mean +- SEM) =============")
    order = [("group", g) for g in GROUP_ORDER] + [("b_relation", r) for r in B_RELATION_ORDER]
    for kind, name in order:
        sub = summary_df[(summary_df["kind"] == kind) & (summary_df["name"] == name)]
        if sub.empty:
            continue
        better = sub["better"].iloc[0]
        print(f"\n{sub['label'].iloc[0]}  ({better} rank is better)")
        for _, row in sub.sort_values(["sigma", "method"]).iterrows():
            print(f"  sigma={row['sigma']:.1f}, {row['method_label']}: "
                  f"{row['rank_mean']:.1f} +- {row['rank_sem']:.1f}")


# --------------------------------------------------------------------------- #
# 增量留一自检（SWEET 版）
# --------------------------------------------------------------------------- #
def _brute_sweet(data, triu_idx, edge_fn, full_matrix_fn):
    """从头重算的 SWEET，用于校验增量版本。"""
    n = data.shape[0]
    m = n - 1
    w_all = edge_fn(data, triu_idx)
    S = _sample_weight(data, full_matrix_fn)
    ssn = np.empty((n, w_all.size))
    for q in range(n):
        sub = np.delete(data, q, axis=0)
        w_minus = edge_fn(sub, triu_idx)
        ssn[q] = (cfg.K * S[q]) * m * (w_all - w_minus) + w_minus
    return ssn


def self_test() -> None:
    print("Running self-test: incremental SWEET leave-one-out vs brute-force ...")
    triu_idx = np.triu_indices(cfg.M, k=1)
    data = generate_dataset(cfg.BASE_SEED, sigma=0.4)

    checks = [
        ("pearson", pearson_sweet, pearson_edges, _pearson_full_matrix),
        ("spearman", spearman_sweet, spearman_edges, _spearman_full_matrix),
        ("dcor", dcor_sweet, dcor_edges, _dcor_full_matrix),
    ]
    for name, fast_fn, edge_fn, full_matrix_fn in checks:
        fast = fast_fn(data, triu_idx)
        brute = _brute_sweet(data, triu_idx, edge_fn, full_matrix_fn)
        max_diff = float(np.max(np.abs(fast - brute)))
        ok = np.allclose(fast, brute, atol=1e-8, rtol=1e-6)
        print(f"  {name:9s}: max|diff| = {max_diff:.3e}  -> {'OK' if ok else 'MISMATCH'}")
        assert ok, f"{name} incremental SWEET mismatch (max diff {max_diff})"
    print("Self-test passed.\n")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "", "false", "False", "FALSE")


def load_existing_summary() -> pd.DataFrame:
    result_path = OUT_DIR / f"{cfg.OUTPUT_PREFIX}_results.xlsx"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Cannot replot because results file is missing: {result_path}. "
            "Run the simulation once, or unset SIM_REPLOT_ONLY."
        )
    summary_df = pd.read_excel(result_path, sheet_name="rank_summary")
    print(f"Loaded existing results from: {result_path}")
    return summary_df


def plot_all_figures(summary_df: pd.DataFrame) -> None:
    plot_figure2(summary_df)


def print_generated_figure_paths() -> None:
    print("\nGenerated figure files:")
    for ext in (".png", ".pdf", ".svg"):
        print(f"  {FIG_DIR / f'{FIG_STEM}{ext}'}")
    print(f"  {FIG_DIR / f'{FIG_STEM}_plot_data.csv'}")
    print(f"  {OUT_DIR / f'{cfg.OUTPUT_PREFIX}_results.xlsx'}")
    print(f"  {Path(__file__).resolve()}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.SELF_TEST:
        self_test()

    if _env_flag("SIM_REPLOT_ONLY"):
        summary_df = load_existing_summary()
    else:
        rep_df = run_simulation()
        summary_df = summarize(rep_df)
        save_results(rep_df, summary_df)

    plot_all_figures(summary_df)
    print_summary(summary_df)
    print_generated_figure_paths()


if __name__ == "__main__":
    main()
