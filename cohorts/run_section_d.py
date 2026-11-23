"""Paired covariate-plus-abundance vs covariate-plus-abundance-plus-network discrimination."""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_info

from cohort_data import COHORTS, ROOT, load_cohort, write_json
from fold_models import edge_design, fold_predictions, pls_scores, standardized_gram
from resources import peak_rss_mib

N_SPLITS, TEST_SIZE, SEED, Q = 100, .30, 42, .10
COMPONENTS = tuple(range(1,9))
N_BOOTSTRAP, BOOTSTRAP_SEED = 10000, 20261122



def prepare(cohort, data_dir, output_dir):
    x, ids, features, y, cov, info = load_cohort(cohort, data_dir, prediction=True)
    outer = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    splits = list(outer.split(x, y))
    train = np.stack([i for i, _ in splits])
    test = np.stack([j for _, j in splits])
    folds = []
    for split, idx in enumerate(train, 1):
        assignment = np.empty(len(idx), dtype=np.uint8)
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + split)
        for k, (_, valid) in enumerate(inner.split(x[idx], y[idx])):
            assignment[valid] = k
        folds.append(assignment)
    directory = output_dir / cohort
    (directory / "splits").mkdir(parents=True, exist_ok=True)
    packaged = directory / "inputs_and_splits.npz"
    arrays = dict(X=np.ascontiguousarray(x), y=y, feature_names=features.feature.to_numpy(dtype=str),
                  sample_ids=ids, covariates=cov, train=train, test=test, inner_fold=np.stack(folds))
    if any((directory / "splits").glob("split_*.npz")):
        with np.load(packaged, allow_pickle=False) as previous:
            for key, array in arrays.items():
                np.testing.assert_array_equal(previous[key], array, err_msg="Saved split inputs changed")
    else:
        staging = packaged.with_name(f"inputs_and_splits.{os.getpid()}.tmp.npz")
        np.savez_compressed(staging, **arrays)
        staging.replace(packaged)
    if cohort == "ST000355":
        info["reconstructed_age_only_auroc"] = np.mean([roc_auc_score(y[j], cov[j, 0]) for j in test])
    write_json(directory / "input_registration.json", info)
    print(f"{cohort}: prepared {len(splits)} splits, {x.shape[0]} samples and {x.shape[1]} features")


def self_check():
    sys.path.insert(0, str(ROOT))
    from dcor_ssn import dcor_ssn_sweet
    rng=np.random.default_rng(192)
    for n,p,m in ((19,31,8),(30,45,10)):
        x=rng.normal(size=(n,p))
        z=rng.normal(size=(m,p))
        train_scores, _ = dcor_ssn_sweet(x)
        test_scores = [dcor_ssn_sweet(np.vstack([x, row]))[0][-1] for row in z]
        scores = np.abs(np.vstack([train_scores, test_scores]).astype(np.float32))
        count = int(np.ceil(Q * scores.shape[1]))
        top = np.argpartition(scores, scores.shape[1] - count, axis=1)[:, -count:]
        expected = np.zeros_like(scores, dtype=bool)
        np.put_along_axis(expected, top, True, axis=1)
        actual=edge_design(x,z,Q)
        np.testing.assert_array_equal(actual,expected)
        x2,z2=rng.normal(size=(n,3)),rng.normal(size=(m,3))
        y=np.arange(n)%2
        train=np.column_stack((x2,actual[:n]))
        test=np.column_stack((z2,actual[n:]))
        k,kz,_=standardized_gram(x2,z2,actual)
        from sklearn.preprocessing import StandardScaler
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import LogisticRegression
        scaler=StandardScaler().fit(train)
        a,b=scaler.transform(train),scaler.transform(test)
        tested=pls_scores(k,kz,y,42,COMPONENTS)
        for c,prob in tested.items():
            pls=PLSRegression(n_components=c,scale=False,max_iter=500,tol=1e-6).fit(a,y)
            model=LogisticRegression(C=1,solver="lbfgs",max_iter=2000,random_state=42)
            model.fit(pls.transform(a),y)
            np.testing.assert_allclose(prob,model.predict_proba(pls.transform(b))[:,1],atol=1e-9,rtol=0)
    print("Cached Top-q scores and row-space PLS-LR agree with direct reference calculations",flush=True)


def init_worker(cohort, output_dir):
    global work, split_dir
    directory=output_dir/cohort
    split_dir=directory/"splits"
    with np.load(directory/"inputs_and_splits.npz",allow_pickle=False) as z:
        work={k:z[k] for k in ("X","y","covariates","train","test","inner_fold")}


def analyse_split(i):
    started=perf_counter()
    file=split_dir/f"split_{i+1:03d}.npz"
    if file.is_file():
        with np.load(file,allow_pickle=False) as saved:
            if saved["probability"].shape!=(2,len(work["test"][i])):
                raise ValueError(f"Invalid saved split {file}")
            np.testing.assert_array_equal(saved["test_ids"],work["test"][i],err_msg=str(file))
        return i+1,0
    x,y,cov=[work[k] for k in ("X","y","covariates")]
    train,test,fold=[work[k][i] for k in ("train","test","inner_fold")]
    seed=SEED+i+1
    cv_scores=np.full((2,5,len(COMPONENTS)),np.nan)
    union_counts=[]
    for k in range(5):
        tr,va=train[fold!=k],train[fold==k]
        assert len(np.intersect1d(tr,va)) == len(np.intersect1d(tr,test)) == 0
        p0,p1,union=fold_predictions(x[tr],x[va],cov[tr],cov[va],y[tr],seed+k+1,Q,(COMPONENTS,COMPONENTS))
        for arm,predictions in enumerate((p0,p1)):
            for comp,prob in predictions.items():
                cv_scores[arm,k,comp-1]=roc_auc_score(y[va],prob)
        union_counts.append(union)
    mean_auroc=cv_scores.mean(axis=1)
    best=tuple(int(np.nanargmax(np.round(arr,12))+1) for arr in mean_auroc)
    p0,p1,union=fold_predictions(x[train],x[test],cov[train],cov[test],y[train],seed,Q,((best[0],),(best[1],)))
    probability=np.vstack([p0[best[0]],p1[best[1]]])
    elapsed=perf_counter()-started
    staging=file.with_name(f"{file.stem}.{os.getpid()}.working.npz")
    np.savez_compressed(staging,probability=probability,selected_components=best,cv_scores=cv_scores,
                        network_union_counts=[*union_counts,union],test_ids=test,
                        duration_seconds=elapsed,peak_rss_mib=peak_rss_mib(),
                        blas_threads=[lib["num_threads"] for lib in threadpool_info() if lib["user_api"]=="blas"])
    staging.replace(file)
    return i+1,elapsed


def run(cohort, workers, benchmark, output_dir):
    if benchmark:
        init_worker(cohort, output_dir)
        i,elapsed=analyse_split(0)
        file=output_dir/cohort/"splits"/"split_001.npz"
        with np.load(file) as z:
            info={"full_nested_splits_measured":1,"seconds":float(z["duration_seconds"]),
                  "peak_rss_mib":float(z["peak_rss_mib"]),"network_union_counts":z["network_union_counts"].tolist(),
                  "blas_threads":z["blas_threads"].tolist()}
        write_json(output_dir/cohort/"resource_trial.json",info)
        print(cohort,"resource trial",info,flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(cohort,output_dir)) as pool:
        futures=[pool.submit(analyse_split,i) for i in range(N_SPLITS)]
        for done,job in enumerate(as_completed(futures),1):
            i,elapsed=job.result()
            print(f"{cohort}: completed split {i:03d}, {done}/{N_SPLITS}, {elapsed:.2f} s",flush=True)


def bootstrap(y, tests, probability):
    weights=np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        len(y),np.full(len(y),1/len(y)),size=N_BOOTSTRAP
    )
    delta=np.zeros(N_BOOTSTRAP)
    valid=np.ones(N_BOOTSTRAP,dtype=bool)
    for split,idx in enumerate(tests):
        pos=y[idx]==1
        cp,cn=weights[:,idx[pos]],weights[:,idx[~pos]]
        denom=cp.sum(axis=1)*cn.sum(axis=1)
        valid &= denom>0
        scores=probability[split]
        wins=(np.sign(scores[:,pos,None]-scores[:,None,~pos])+1)/2
        contrast=wins[1]-wins[0]
        numer=((cp@contrast)*cn).sum(axis=1)
        delta+=np.divide(numer,denom,out=np.zeros_like(numer),where=denom>0)
    return delta[valid]/len(tests),valid


def summarize(cohort, output_dir):
    directory=output_dir/cohort
    with np.load(directory/"inputs_and_splits.npz",allow_pickle=False) as z:
        y,tests,ids=z["y"],z["test"],z["sample_ids"]
    probability,splits,tuning=[],[],[]
    for i,test in enumerate(tests,1):
        with np.load(directory/"splits"/f"split_{i:03d}.npz",allow_pickle=False) as z:
            np.testing.assert_array_equal(z["test_ids"],test)
            prob=z["probability"]
            auc=[roc_auc_score(y[test],p) for p in prob]
            splits.append((i,*auc,auc[1]-auc[0],*z["selected_components"],z["network_union_counts"][-1],
                           z["duration_seconds"].item(),z["peak_rss_mib"].item()))
            probability.append(prob)
            values=z["cv_scores"]
            tuning.extend((i,a+1,f+1,c+1,values[a,f,c])
                          for a in range(2) for f in range(5) for c in range(len(COMPONENTS)))
    ar=np.stack(probability)
    split_frame=pd.DataFrame(splits,columns=["split","baseline_auroc","increment_auroc","delta_auroc",
                                            "baseline_components","increment_components","network_union_edges",
                                            "seconds","peak_rss_mib"])
    split_frame.to_csv(directory/"paired_split_aurocs.csv",index=False)
    pd.DataFrame(tuning,columns=["split","arm","inner_fold","n_components","inner_auroc"]).to_csv(
        directory/"inner_tuning.csv",index=False)
    pd.DataFrame({"split":np.repeat(np.arange(1,N_SPLITS+1),tests.shape[1]),
                  "sample_id":ids[tests].ravel(),"label":y[tests].ravel(),
                  "baseline_probability":ar[:,0].ravel(),"increment_probability":ar[:,1].ravel()}).to_csv(
        directory/"paired_test_predictions.csv",index=False)
    d,valid=bootstrap(y,tests,ar)
    np.savez_compressed(directory/"subject_bootstrap.npz",delta_auroc=d,accepted_draw=valid,seed=BOOTSTRAP_SEED)
    lower,upper=np.quantile(d,[.025,.975])
    delta=float(split_frame["delta_auroc"].mean())
    summary={"cohort":cohort,"n_outer_splits":len(tests),
             "baseline_mean_auroc":float(split_frame["baseline_auroc"].mean()),
             "increment_mean_auroc":float(split_frame["increment_auroc"].mean()),
             "mean_paired_delta_auroc":delta,"ci95_lower":float(lower),"ci95_upper":float(upper),
             "bootstrap_accepted":int(valid.sum()),"bootstrap_requested":len(valid),
             "practical_gain":bool(delta>=.02 and lower>0)}
    write_json(directory/"summary.json",summary)
    print(summary,flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare", "check", "benchmark", "run", "summarize"))
    parser.add_argument("--cohort", choices=COHORTS, default="ST000355")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "section_d")
    parser.add_argument("--workers", type=int, default=1, help="Parallel outer splits (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.task == "prepare":
        prepare(args.cohort, args.data_dir, args.output_dir)
    elif args.task == "check":
        self_check()
    elif args.task == "summarize":
        summarize(args.cohort, args.output_dir)
    else:
        run(args.cohort, args.workers, args.task == "benchmark", args.output_dir)


if __name__ == "__main__":
    main()
