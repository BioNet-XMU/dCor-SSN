"""Fold-local SWEET Top-q and PLS-LR for the paired increment analysis."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def profile_distances(rows):
    """Pack symmetric sample-profile distances and compute their row sums."""
    p = rows.shape[1]
    i, j = np.triu_indices(p, 1)
    distances = np.empty((len(rows), len(i)))
    row_sums = np.empty_like(rows)
    position = np.arange(p)
    for s, row in enumerate(rows):
        distances[s] = np.abs(row[i] - row[j])
        order = np.argsort(row)
        values = row[order]
        cumulative = np.cumsum(values)
        row_sums[s, order] = (2 * position - p + 2) * values + cumulative[-1] - 2 * cumulative
    return distances, row_sums


def profile_weights(x, z, eps):
    """Use the original normalization on the reference and on each reference + target."""
    n, p = x.shape
    packed, rows = profile_distances(np.vstack((x, z)))
    train, target = packed[:n], packed[n:]
    tr_rows, te_rows = rows[:n], rows[n:]
    ts, vs = tr_rows.sum(axis=1), te_rows.sum(axis=1)
    tr_ab = (2.0 / p**2) * (train @ train.T)
    tr_dc = tr_ab - (2.0 / p**3) * (tr_rows @ tr_rows.T) + np.outer(ts, ts) / p**4
    dtr = np.maximum(np.diag(tr_dc), 0)
    dte = np.maximum(
        (2.0 / p**2) * np.einsum("ij,ij->i", target, target)
        - (2.0 / p**3) * np.einsum("ij,ij->i", te_rows, te_rows) + (vs / p**2) ** 2, 0,
    )

    def normalized(dc, denominator):
        ratio = np.divide(dc, denominator, out=np.zeros_like(dc), where=denominator > 0)
        return np.sqrt(np.clip(ratio, 0, 1))

    tr_sim = normalized(tr_dc, np.sqrt(np.outer(dtr, dtr)))
    te_dc = ((2.0 / p**2) * (target @ train.T) - (2.0 / p**3) * (te_rows @ tr_rows.T)
             + np.outer(vs, ts) / p**4)
    te_sim = normalized(te_dc, np.sqrt(np.outer(dte, dtr)))
    sums = tr_sim.sum(axis=1) - tr_sim.diagonal()
    avg = sums / (n - 1)
    train_weight = (avg - avg.min() + eps) / (avg.max() - avg.min() + eps)
    each_avg = (sums[None, :] + te_sim) / n
    target_avg = te_sim.mean(axis=1)
    lo = np.minimum(each_avg.min(axis=1), target_avg)
    hi = np.maximum(each_avg.max(axis=1), target_avg)
    return train_weight, (target_avg - lo + eps) / (hi - lo + eps)


def dcor_edges(ab, sr, totals, n, tri):
    i, j = tri
    variance = np.diag(ab) / n**2 - 2.0 * np.diag(sr) / n**3 + (totals / n**2)**2
    covariance = ab[tri] / n**2 - 2.0 * sr[tri] / n**3 + totals[i] * totals[j] / n**4
    denominator = np.sqrt(np.maximum(variance[i] * variance[j], 0))
    ratio = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    return np.sqrt(np.clip(ratio, 0, 1))


def edge_design(x, z, q, scale=.1, eps=.01):
    """Exact leave-one-out training and one-target projection, retaining binary Top-q."""
    n,p = x.shape
    tri = np.triu_indices(p,1)
    size = len(tri[0])
    k = int(np.ceil(q * size))
    w_train,w_test = profile_weights(x,z,eps)
    vals = np.ascontiguousarray(x.T)
    dist = vals[:,:,None]-vals[:,None,:]
    np.abs(dist,out=dist)
    ra = dist.sum(axis=2)
    totals = ra.sum(axis=1)
    flat = dist.reshape(p,-1)
    ab = flat @ flat.T
    full = dcor_edges(ab, ra @ ra.T, totals, n, tri)
    output = np.zeros((n+len(z),size),dtype=np.bool_)

    def retain(index, weights):
        weights = np.abs(weights.astype(np.float32))
        output[index,np.argpartition(weights,size-k)[size-k:]] = True

    for i in range(n):
        c = np.ascontiguousarray(dist[:,:,i])
        reduced = ra-c
        reduced[:,i] = 0.0
        loo = dcor_edges(ab-2*(c@c.T),reduced@reduced.T,totals-2*ra[:,i],n-1,tri)
        retain(i,(scale*w_train[i])*(n-1)*(full-loo)+loo)
    del dist, flat
    for i,row in enumerate(z):
        c = np.abs(vals-row[:,None])
        sc = c.sum(axis=1)
        enlarged = ra+c
        comb = dcor_edges(ab+2*(c@c.T),enlarged@enlarged.T+np.outer(sc,sc),totals+2*sc,n+1,tri)
        retain(n+i,(scale*w_test[i])*n*(comb-full)+full)
    assert np.all(output.sum(axis=1) == k)
    return output


def standardized_gram(x, z, edges=None):
    """Return XX^T and ZX^T after training-only feature scaling."""
    scaler = StandardScaler()
    tr = scaler.fit_transform(x)
    te = scaler.transform(z)
    k = tr @ tr.T
    kz = te @ tr.T
    n = len(x)
    union_count = 0
    if edges is not None:
        rates = edges[:n].mean(axis=0)
        union_count = int((rates>0).sum())
        selected = np.flatnonzero((rates>0)&(rates<1))
        for start in range(0,len(selected),16000):
            idx = selected[start:start+16000]
            block = edges[:,idx].astype(float)
            block -= rates[idx]
            block /= np.sqrt(rates[idx]*(1-rates[idx]))
            k += block[:n] @ block[:n].T
            kz += block[n:] @ block[:n].T
    return k, kz, union_count


def orthogonal_coordinates(k, kz):
    """Represent the same scaled design on its training row space."""
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    keep = eigenvalues > (eigenvalues[-1] * 1e-12)
    roots = np.sqrt(eigenvalues[keep])
    basis = eigenvectors[:,keep]
    return basis*roots, (kz@basis)/roots


def pls_scores(k, kz, y, seed, choices):
    tr,te = orthogonal_coordinates(k,kz)
    model = PLSRegression(n_components=min(max(choices),tr.shape[1]),scale=False,
                          max_iter=500,tol=1e-6)
    model.fit(tr,y)
    a,b = model.transform(tr),model.transform(te)
    pred = {}
    for c in choices:
        if c>a.shape[1]:
            continue
        lr = LogisticRegression(C=1,solver="lbfgs",max_iter=2000,random_state=seed)
        lr.fit(a[:,:c],y)
        pred[c] = lr.predict_proba(b[:,:c])[:,1]
    return pred


def fold_predictions(x, z, clinical_tr, clinical_te, y, seed, q, choices):
    """Fit both arms against the same reference; z is never part of that reference."""
    scale = x.std(axis=0)
    scale[scale==0] = 1
    center = x.mean(axis=0)
    tr,te = (x-center)/scale,(z-center)/scale
    a,b = np.column_stack((clinical_tr,x)),np.column_stack((clinical_te,z))
    edges = edge_design(tr,te,q)
    k,kz,_ = standardized_gram(a,b)
    base = pls_scores(k,kz,y,seed,choices[0])
    k,kz,union = standardized_gram(a,b,edges)
    increment = pls_scores(k,kz,y,seed,choices[1])
    return base,increment,union
