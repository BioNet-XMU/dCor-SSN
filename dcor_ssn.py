"""
dcor_ssn.py
===========

NumPy-only reference implementation of dCor-SSN, the distance-correlation
sample-specific network estimator used in the manuscript, together with the
Pearson-SSN and Spearman-SSN comparators.

Contents
--------
* Distance correlation (dCor) and its exact leave-one-out update.
* SWEET sample-specific networks (the method reported in the manuscript) for
  dCor, Pearson and Spearman.
* dCor-LIONESS, a separate estimator that shares the same leave-one-out update.
* Brute-force leave-one-out recomputation for validation.
* A runnable validation example and a runtime / memory benchmark of the three
  SWEET estimators.

Run
---
    python dcor_ssn.py                 # exact update vs brute force (LIONESS + SWEET)
    python dcor_ssn.py --benchmark     # also time the three SWEET estimators

Definitions
-----------
For an N x P data matrix X (rows = samples, columns = variables), D_all is the
dependence matrix estimated from all N samples and D_-q the same matrix after
removing sample q. The population-level dCor is the standard biased
(V-statistic) sample distance correlation; Pearson and Spearman are signed.

SWEET (Chen et al., 2023, Brief Bioinform, bbad032):

    e_q,ij = (K * S_q) * (N - 1) * (D_all,ij - D_-q,ij) + D_-q,ij

LIONESS:

    e_q,ij = N * D_all,ij - (N - 1) * D_-q,ij

S_q is a sample weight obtained from the sample-sample dependence matrix of the
same measure (X transposed, so samples become the variables), averaged over the
other samples and min-max scaled with a small eps. For Spearman this ranks the
features within each sample. Defaults: K = 0.1, eps = 0.01.

Exact dCor leave-one-out update
-------------------------------
All P pairwise-distance matrices are computed once. Removing a sample only
changes row sums, totals and raw inner products, from which the double-centred
leave-one-out inner products are rebuilt exactly. This reduces the dominant
cost of all N leave-one-out dCor networks from O(P^2 N^3) to O(P^2 N^2) while
matching brute-force recomputation up to floating-point roundoff.

No top-q encoding, classification, pathway analysis or other downstream step
is performed here. No normalisation, absolute-value transform or
measure-specific post-processing is applied beyond the formulas above.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

DependencyFn = Callable[[np.ndarray], np.ndarray]

__all__ = [
    "distance_correlation_matrix",
    "pearson_matrix",
    "spearman_matrix",
    "sample_weight",
    "upper_triangle_edges",
    "dcor_ssn_sweet",
    "dcor_ssn_sweet_bruteforce",
    "pearson_ssn_sweet",
    "spearman_ssn_sweet",
    "dcor_ssn_lioness",
    "dcor_ssn_lioness_bruteforce",
    "validate_exact_update",
    "benchmark",
]


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #
def _validate_X(X: np.ndarray, min_samples: int = 3) -> np.ndarray:
    """Validate and convert an input data matrix to contiguous float64."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_variables).")
    n, p = X.shape
    if n < min_samples:
        raise ValueError(f"At least {min_samples} samples are required.")
    if p < 2:
        raise ValueError("At least 2 variables are required.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return np.ascontiguousarray(X)


def _validate_sweet_params(K: float, eps: float) -> tuple[float, float]:
    """Validate SWEET hyperparameters."""
    K = float(K)
    eps = float(eps)
    if not np.isfinite(K):
        raise ValueError("K must be a finite float.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive float.")
    return K, eps


def upper_triangle_edges(p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (i, j) indices of the undirected upper-triangle edges."""
    if p < 2:
        raise ValueError("p must be >= 2.")
    return np.triu_indices(p, k=1)


# --------------------------------------------------------------------------- #
# Dependence measures
# --------------------------------------------------------------------------- #
def _distance_stack(X: np.ndarray) -> np.ndarray:
    """
    Pairwise absolute-distance matrices for all scalar variables.

    Returns D with shape (P, N, N), D[v, a, b] = |X[a, v] - X[b, v]|.
    """
    XT = X.T
    return np.abs(XT[:, :, None] - XT[:, None, :])


def _centered_inner_from_sufficient(
    raw_inner: np.ndarray,
    row_inner: np.ndarray,
    total: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    sum_{a,b} A_ab B_ab for double-centred distance matrices A, B.

    For raw symmetric distance matrices a, b with zero diagonal,

        <A, B> = sum(a*b) - (2/n) sum(rowsum(a) * rowsum(b))
                 + total(a) * total(b) / n^2,

    applied simultaneously to every variable pair.
    """
    nf = float(n)
    M = raw_inner - (2.0 / nf) * row_inner + np.outer(total, total) / (nf * nf)
    # Symmetrise to suppress tiny BLAS roundoff asymmetry.
    return 0.5 * (M + M.T)


def _dcor_from_centered_inner(M: np.ndarray) -> np.ndarray:
    """
    Convert centred distance inner products to biased sample dCor.

    dCor(i, j) = sqrt(M_ij / sqrt(M_ii * M_jj)); the n^-2 factor of distance
    covariance cancels. Zero-distance-variance variables get dCor = 0.
    """
    M = np.asarray(M, dtype=np.float64)
    diag = np.clip(np.diag(M), 0.0, None)      # guard tiny negative roundoff
    denom = np.sqrt(np.outer(diag, diag))

    R2 = np.zeros_like(M)
    valid = denom > 0.0
    R2[valid] = M[valid] / denom[valid]
    R = np.sqrt(np.clip(R2, 0.0, 1.0))
    return 0.5 * (R + R.T)


def distance_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Standard biased (V-statistic) sample dCor matrix, shape (P, P).

    Rows of X are samples and columns are variables. A zero-variance variable
    has a zero row/column.
    """
    X = _validate_X(X, min_samples=2)
    n, p = X.shape
    D = _distance_stack(X)
    row_sum = D.sum(axis=2)
    total = row_sum.sum(axis=1)
    F = D.reshape(p, -1)
    M = _centered_inner_from_sufficient(F @ F.T, row_sum @ row_sum.T, total, n)
    return _dcor_from_centered_inner(M)


def pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Signed Pearson correlation matrix; zero-variance pairs are set to 0."""
    X = _validate_X(X, min_samples=2)
    Z = X - X.mean(axis=0, keepdims=True)
    ss = np.sum(Z * Z, axis=0)
    denom = np.sqrt(np.outer(ss, ss))
    C = Z.T @ Z
    R = np.zeros_like(C)
    valid = denom > 0.0
    R[valid] = C[valid] / denom[valid]
    R = np.clip(R, -1.0, 1.0)
    return 0.5 * (R + R.T)


def _rank_columns(X: np.ndarray) -> np.ndarray:
    """
    Column-wise 1-based average ranks with exact tie handling (NumPy only).

    Tied values receive the mean of the positions they occupy, identical to
    scipy.stats.rankdata(method="average") applied column by column.
    """
    X = _validate_X(X, min_samples=2)
    n, p = X.shape
    order = np.argsort(X, axis=0, kind="mergesort")
    xs = np.take_along_axis(X, order, axis=0)

    pos = np.arange(n)[:, None]
    new_group = np.ones((n, p), dtype=bool)
    new_group[1:] = xs[1:] != xs[:-1]
    # First and last sorted position of each tie group.
    first = np.maximum.accumulate(np.where(new_group, pos, 0), axis=0)
    end_group = np.ones((n, p), dtype=bool)
    end_group[:-1] = new_group[1:]
    last = np.minimum.accumulate(
        np.where(end_group, pos, n - 1)[::-1], axis=0
    )[::-1]

    ranks_sorted = 0.5 * (first + last) + 1.0
    R = np.empty_like(ranks_sorted)
    np.put_along_axis(R, order, ranks_sorted, axis=0)
    return R


def spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Signed Spearman correlation: Pearson on column-wise average ranks."""
    return pearson_matrix(_rank_columns(X))


# --------------------------------------------------------------------------- #
# SWEET sample weights and assembly
# --------------------------------------------------------------------------- #
def sample_weight(X: np.ndarray, dep: DependencyFn, eps: float = 0.01) -> np.ndarray:
    """
    SWEET sample weights S_q in (0, 1].

    dep(X.T) gives the (N, N) sample-sample dependence matrix R of the same
    measure. mu_q is the mean of R_qr over r != q, and

        S_q = (mu_q - min(mu) + eps) / (max(mu) - min(mu) + eps).
    """
    X = _validate_X(X, min_samples=2)
    _, eps = _validate_sweet_params(K=0.1, eps=eps)
    n = X.shape[0]

    R = np.asarray(dep(X.T), dtype=np.float64)
    if R.shape != (n, n):
        raise ValueError("dep(X.T) must return an (n_samples, n_samples) matrix.")

    mu = (R.sum(axis=1) - np.diag(R)) / (n - 1)
    lo, hi = float(mu.min()), float(mu.max())
    return (mu - lo + eps) / (hi - lo + eps)


def _finalize(E, edge_pairs, D_full, S, *, return_full: bool, return_weights: bool):
    """Assemble optional return values."""
    out = [E, edge_pairs]
    if return_full:
        out.append(D_full)
    if return_weights:
        out.append(S)
    return tuple(out)


def _dcor_full_and_loo(X: np.ndarray) -> tuple[np.ndarray, Iterator[np.ndarray]]:
    """
    Full-cohort dCor matrix and an iterator over the exact leave-one-out dCor
    matrices D_-q, q = 0..N-1, via the sufficient-statistic update.

    With D_v the N x N raw distance matrix of variable v:
      T_v       = sum(D_v)
      R_uv      = sum_i rowsum(D_u)[i] * rowsum(D_v)[i]
      S_uv      = sum_{i,j} D_u[i,j] * D_v[i,j]
    After removing sample q:
      T_v^(-q)  = T_v - 2 * rowsum(D_v)[q]
      S_uv^(-q) = S_uv - 2 * sum_j D_u[q,j] D_v[q,j]
      rowsum(D_v^(-q))[i] = rowsum(D_v)[i] - D_v[i,q]   for i != q.
    Each q costs only P x N matrix products, O(P^2 N^2) in total.
    """
    n, p = X.shape
    m = n - 1

    D = _distance_stack(X)                  # (P, N, N)
    row_sum = D.sum(axis=2)                 # (P, N)
    total = row_sum.sum(axis=1)             # (P,)
    F = D.reshape(p, -1)
    raw_inner = F @ F.T                     # S_uv
    row_inner = row_sum @ row_sum.T         # R_uv

    D_full = _dcor_from_centered_inner(
        _centered_inner_from_sufficient(raw_inner, row_inner, total, n)
    )

    def loo() -> Iterator[np.ndarray]:
        for q in range(n):
            H = D[:, q, :]                              # distances from q, (P, N)
            raw_inner_loo = raw_inner - 2.0 * (H @ H.T)
            total_loo = total - 2.0 * row_sum[:, q]
            reduced_rows = row_sum - D[:, :, q]
            reduced_rows[:, q] = 0.0                    # q contributes nothing
            row_inner_loo = reduced_rows @ reduced_rows.T
            yield _dcor_from_centered_inner(
                _centered_inner_from_sufficient(
                    raw_inner_loo, row_inner_loo, total_loo, m
                )
            )

    return D_full, loo()


def _bruteforce_full_and_loo(
    X: np.ndarray, dep: DependencyFn
) -> tuple[np.ndarray, Iterator[np.ndarray]]:
    """Full-cohort matrix and leave-one-out matrices recomputed from scratch."""
    D_full = dep(X)
    loo = (dep(np.delete(X, q, axis=0)) for q in range(X.shape[0]))
    return D_full, loo


def _assemble_sweet(X, dep, full_and_loo, *, K, eps, return_full, return_weights):
    """Shared SWEET assembly for exact and brute-force leave-one-out paths."""
    X = _validate_X(X, min_samples=3)
    K, eps = _validate_sweet_params(K, eps)
    n, p = X.shape
    m = n - 1

    S = sample_weight(X, dep, eps=eps)
    D_full, loo = full_and_loo(X)

    iu = upper_triangle_edges(p)
    E = np.empty((n, len(iu[0])), dtype=np.float64)
    for q, D_loo in enumerate(loo):
        E[q, :] = ((K * S[q]) * m * (D_full - D_loo) + D_loo)[iu]

    return _finalize(E, np.column_stack(iu), D_full, S,
                     return_full=return_full, return_weights=return_weights)


# --------------------------------------------------------------------------- #
# Public SWEET estimators
# --------------------------------------------------------------------------- #
def dcor_ssn_sweet(X, *, K=0.1, eps=0.01, return_full=False, return_weights=False):
    """
    dCor-SWEET sample-specific networks with the exact leave-one-out update.

    Returns
    -------
    E : ndarray, shape (N, P(P-1)/2)
        Edge contributions of every sample in upper-triangle order.
    edges : ndarray, shape (P(P-1)/2, 2)
        Variable-index pair (i, j) of each column of E.
    D_full : ndarray, shape (P, P), only if return_full=True
    S : ndarray, shape (N,), only if return_weights=True
    """
    return _assemble_sweet(X, distance_correlation_matrix, _dcor_full_and_loo,
                           K=K, eps=eps, return_full=return_full,
                           return_weights=return_weights)


def dcor_ssn_sweet_bruteforce(X, *, K=0.1, eps=0.01, return_full=False, return_weights=False):
    """dCor-SWEET with every leave-one-out dCor recomputed (validation only)."""
    dep = distance_correlation_matrix
    return _assemble_sweet(X, dep, lambda A: _bruteforce_full_and_loo(A, dep),
                           K=K, eps=eps, return_full=return_full,
                           return_weights=return_weights)


def pearson_ssn_sweet(X, *, K=0.1, eps=0.01, return_full=False, return_weights=False):
    """Pearson-SWEET sample-specific networks (same interface as dcor_ssn_sweet)."""
    dep = pearson_matrix
    return _assemble_sweet(X, dep, lambda A: _bruteforce_full_and_loo(A, dep),
                           K=K, eps=eps, return_full=return_full,
                           return_weights=return_weights)


def spearman_ssn_sweet(X, *, K=0.1, eps=0.01, return_full=False, return_weights=False):
    """
    Spearman-SWEET sample-specific networks. Ranks are recomputed within every
    leave-one-out sample set; sample weights rank features within each sample.
    """
    dep = spearman_matrix
    return _assemble_sweet(X, dep, lambda A: _bruteforce_full_and_loo(A, dep),
                           K=K, eps=eps, return_full=return_full,
                           return_weights=return_weights)


# --------------------------------------------------------------------------- #
# dCor-LIONESS (separate comparator sharing the exact update)
# --------------------------------------------------------------------------- #
def _assemble_lioness(X, full_and_loo, *, return_full):
    X = _validate_X(X, min_samples=3)
    n, p = X.shape
    D_full, loo = full_and_loo(X)
    iu = upper_triangle_edges(p)
    E = np.empty((n, len(iu[0])), dtype=np.float64)
    for q, D_loo in enumerate(loo):
        E[q, :] = (n * D_full - (n - 1) * D_loo)[iu]
    return _finalize(E, np.column_stack(iu), D_full, None,
                     return_full=return_full, return_weights=False)


def dcor_ssn_lioness(X, *, return_full=False):
    """dCor-LIONESS networks with the exact leave-one-out update."""
    return _assemble_lioness(X, _dcor_full_and_loo, return_full=return_full)


def dcor_ssn_lioness_bruteforce(X, *, return_full=False):
    """dCor-LIONESS with every leave-one-out dCor recomputed (validation only)."""
    return _assemble_lioness(
        X, lambda A: _bruteforce_full_and_loo(A, distance_correlation_matrix),
        return_full=return_full,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_exact_update(
    *,
    seed: int = 42,
    n: int = 18,
    p: int = 7,
    K: float = 0.1,
    eps: float = 0.01,
    tol: float = 1e-10,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compare the exact dCor leave-one-out update with brute-force recomputation
    for both SWEET and LIONESS assembly. Two nonlinear columns (quadratic and
    sinusoidal) are included so the check is not purely Gaussian-linear.

    Returns {"sweet": max_abs_error, "lioness": max_abs_error}.
    """
    if n < 5 or p < 4:
        raise ValueError("Use n >= 5 and p >= 4 for the validation example.")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] ** 2 + 0.35 * rng.normal(size=n)
    X[:, 3] = np.sin(1.5 * X[:, 2]) + 0.35 * rng.normal(size=n)

    pairs = {
        "sweet": (dcor_ssn_sweet(X, K=K, eps=eps),
                  dcor_ssn_sweet_bruteforce(X, K=K, eps=eps)),
        "lioness": (dcor_ssn_lioness(X), dcor_ssn_lioness_bruteforce(X)),
    }
    errors = {}
    for name, ((E_a, ed_a), (E_b, ed_b)) in pairs.items():
        if not np.array_equal(ed_a, ed_b):
            raise AssertionError(f"{name}: edge ordering mismatch.")
        errors[name] = float(np.max(np.abs(E_a - E_b)))

    if verbose:
        print("dCor exact leave-one-out update vs brute-force recomputation")
        print(f"  X shape              : {X.shape}  (seed={seed})")
        print(f"  undirected edges/net : {p * (p - 1) // 2}")
        print(f"  SWEET  (K={K}, eps={eps}) max abs error : {errors['sweet']:.3e}")
        print(f"  LIONESS                  max abs error : {errors['lioness']:.3e}")
        ok = max(errors.values()) < tol
        print(f"  status               : {'PASS' if ok else 'CHECK'} (tol={tol:g})")
    return errors


# --------------------------------------------------------------------------- #
# Runtime / memory benchmark of the three SWEET estimators
# --------------------------------------------------------------------------- #
def benchmark(
    n: int = 214,
    p: int = 227,
    repeats: int = 3,
    seed: int = 0,
    K: float = 0.1,
    eps: float = 0.01,
) -> list[dict]:
    """
    Time building all N SWEET sample-specific networks for each method.

    Defaults match the ST000355 cohort size (N = 214, P = 227, 25,651 edges).
    Wall time is the median over `repeats` runs. Peak memory is the peak of
    NumPy/Python allocations during one additional run, measured with
    tracemalloc so that each method is measured independently. Absolute values
    are machine-dependent.
    """
    import time
    import tracemalloc

    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n, p))
    X = base.copy()
    # A few nonlinear dependencies so dCor has non-trivial work.
    for k in range(1, min(p, 6)):
        col = base[:, 0] ** 2 if k % 2 else np.sin(2.0 * base[:, 0])
        col = (col - col.mean()) / (col.std() or 1.0)
        X[:, k] = col + 0.3 * base[:, k]

    funcs = {
        "dCor-SWEET": dcor_ssn_sweet,
        "Pearson-SWEET": pearson_ssn_sweet,
        "Spearman-SWEET": spearman_ssn_sweet,
    }
    print(f"SWEET benchmark: N={n}, P={p}, edges={p * (p - 1) // 2}, "
          f"K={K}, eps={eps}, repeats={repeats}")
    rows: list[dict] = []
    for name, fn in funcs.items():
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(X, K=K, eps=eps)
            times.append(time.perf_counter() - t0)
        tracemalloc.start()
        fn(X, K=K, eps=eps)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row = {"method": name, "median_seconds": float(np.median(times)),
               "peak_alloc_gb": peak / 1024.0 ** 3}
        rows.append(row)
        print(f"  {name:15s}: median {row['median_seconds']:.3f} s, "
              f"peak allocation {row['peak_alloc_gb']:.3f} GB")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", action="store_true",
                        help="Also time the three SWEET estimators.")
    parser.add_argument("--n", type=int, default=214, help="Benchmark samples.")
    parser.add_argument("--p", type=int, default=227, help="Benchmark variables.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    errors = validate_exact_update()
    if max(errors.values()) >= 1e-10:
        raise SystemExit("Validation failed: exact update does not match brute force.")
    if args.benchmark:
        print()
        benchmark(n=args.n, p=args.p, repeats=args.repeats)
