#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SWEET edge-construction kernel for the training-restricted PLS-LR workflow.

Contains only the dependence measures, SWEET training (leave-one-out) and test
(projection) edge scores, and Top-q binarization. Model fitting, PLS component
selection (inner-CV AUROC of the full PLS-LR pipeline) and VIP are implemented in
``run_new_plslr_sweet_rankunion_nestedcv.py``.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import rankdata


K_SWEET = float(os.environ.get("K_SWEET", "0.1"))
EPS_SWEET = float(os.environ.get("EPS_SWEET", "0.01"))


def corr_matrix(x: np.ndarray) -> np.ndarray:
    z = x - x.mean(0, keepdims=True)
    norm = np.sqrt(np.sum(z * z, 0))
    den = np.outer(norm, norm)
    return np.divide(z.T @ z, den, out=np.zeros((x.shape[1], x.shape[1])), where=den > 0)


def ranks(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda a: rankdata(a, method="average"), 0, x)


def sample_weight(dep_samp_samp: np.ndarray) -> np.ndarray:
    n = dep_samp_samp.shape[0]
    mu = (dep_samp_samp.sum(1) - np.diag(dep_samp_samp)) / (n - 1.0)
    lo = float(mu.min())
    hi = float(mu.max())
    return (mu - lo + EPS_SWEET) / (hi - lo + EPS_SWEET)


def corr_sample_similarity(x: np.ndarray, spearman: bool = False) -> np.ndarray:
    """Sample-by-sample dependence matrix used for the SWEET weight S_q.

    Samples are treated as variables and metabolites as observations, exactly as
    in ``03_network_construction/dcor_ssn.py::sample_weight`` and the two
    simulations. For Spearman, metabolites are therefore ranked *within each
    sample* before the sample-sample Pearson correlation is computed.

    Revision 2026-09-29: the earlier version ranked each metabolite across
    samples (``ranks(x)``) before transposing, which is not the sample-sample
    Spearman correlation and differed from the other analyses.
    """
    samples_as_columns = x.T
    xx = ranks(samples_as_columns) if spearman else samples_as_columns
    return corr_matrix(xx)


def dcor_from(sab: np.ndarray, sr: np.ndarray, sa: np.ndarray, n: int) -> np.ndarray:
    dc = sab / (n * n) - 2 * sr / (n**3) + np.outer(sa, sa) / (n**4)
    dv = np.diag(dc)
    den = np.sqrt(np.maximum(np.outer(dv, dv), 0))
    ratio = np.divide(dc, den, out=np.zeros_like(dc), where=den > 0)
    return np.sqrt(np.clip(ratio, 0, 1))


def dstats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vals = x.T
    dist = np.abs(vals[:, :, None] - vals[:, None, :])
    ra = dist.sum(2)
    sa = ra.sum(1)
    flat = dist.reshape(vals.shape[0], -1)
    sab = flat @ flat.T
    sr = ra @ ra.T
    return dist, ra, sa, sab, sr, dcor_from(sab, sr, sa, vals.shape[1])


def dcor_sample_similarity(x: np.ndarray) -> np.ndarray:
    return dstats(x.T)[-1]


def dcor_train(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    tri = np.triu_indices(p, 1)
    dist, ra, sa, sab, _sr, full = dstats(x)
    weights = sample_weight(dcor_sample_similarity(x))
    out = np.empty((n, len(tri[0])), dtype=np.float32)
    keep = np.ones(n, dtype=bool)
    for q in range(n):
        keep[q] = False
        c = dist[:, :, q]
        saq = sa - 2 * ra[:, q]
        sabq = sab - 2 * (c @ c.T)
        rq = ra[:, keep] - c[:, keep]
        loo = dcor_from(sabq, rq @ rq.T, saq, n - 1)
        out[q] = ((K_SWEET * weights[q]) * (n - 1) * (full - loo) + loo)[tri]
        keep[q] = True
    return out


def dcor_test(xtr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    n, p = xtr.shape
    tri = np.triu_indices(p, 1)
    _dist, ra, sa, sab, _sr, full = dstats(xtr)
    vals = xtr.T
    out = np.empty((len(xte), len(tri[0])), dtype=np.float32)
    for k, row in enumerate(xte):
        c = np.abs(vals - row[:, None])
        sc = c.sum(1)
        sabc = sab + 2 * (c @ c.T)
        ro = ra + c
        src = ro @ ro.T + np.outer(sc, sc)
        sac = sa + 2 * sc
        comb = dcor_from(sabc, src, sac, n + 1)
        sk = sample_weight(dcor_sample_similarity(np.vstack([xtr, row])))[-1]
        out[k] = ((K_SWEET * sk) * n * (comb - full) + full)[tri]
    return out


def corr_train(x: np.ndarray, spearman: bool = False) -> np.ndarray:
    n, p = x.shape
    tri = np.triu_indices(p, 1)
    xx = ranks(x) if spearman else x
    full = corr_matrix(xx)
    weights = sample_weight(corr_sample_similarity(x, spearman))
    out = np.empty((n, len(tri[0])), dtype=np.float32)
    for q in range(n):
        sub = np.delete(x, q, 0)
        loo = corr_matrix(ranks(sub) if spearman else sub)
        out[q] = ((K_SWEET * weights[q]) * (n - 1) * (full - loo) + loo)[tri]
    return out


def corr_test(xtr: np.ndarray, xte: np.ndarray, spearman: bool = False) -> np.ndarray:
    n, p = xtr.shape
    tri = np.triu_indices(p, 1)
    full = corr_matrix(ranks(xtr) if spearman else xtr)
    out = np.empty((len(xte), len(tri[0])), dtype=np.float32)
    for k, row in enumerate(xte):
        combined = np.vstack([xtr, row])
        comb = corr_matrix(ranks(combined) if spearman else combined)
        sk = sample_weight(corr_sample_similarity(combined, spearman))[-1]
        out[k] = ((K_SWEET * sk) * n * (comb - full) + full)[tri]
    return out


def binq_rank_pct(e: np.ndarray, q: float) -> np.ndarray:
    n, m = e.shape
    k = int(np.ceil(q * m))
    idx = np.argpartition(np.abs(e), m - k, axis=1)[:, m - k:]
    out = np.zeros_like(e, dtype=np.float32)
    out[np.repeat(np.arange(n), k), idx.reshape(-1)] = 1.0
    return out
