# dCor-SSN

Python code for **Distance Correlation Extends Sample-Specific Metabolic Networks to Nonlinear Dependence**, by Shiqi Zhuo, Genjin Lin, Liheng Dong, Lingli Deng, Daniel Raftery, and Jiyang Dong.

dCor-SSN ranks metabolite pairs for each sample relative to a reference cohort. It uses distance correlation (dCor) in two places in the SWEET calculation: to measure associations between metabolites across samples and to measure similarity between samples across their metabolite profiles. The paper uses **dCor-SWEET** for this same estimator when comparing it with Pearson-SWEET and Spearman-SWEET. dCor-LIONESS is a separate simulation comparator.

## Run the estimator

The reference implementation requires only NumPy. The complete analysis environment below was tested with Python 3.12. Run commands from the repository root.

```sh
python -m venv .venv
# Activate with source .venv/bin/activate on macOS/Linux, or .\.venv\Scripts\Activate.ps1 in PowerShell.
python -m pip install -r requirements.txt
python dcor_ssn.py
```

The last command compares the exact dCor leave-one-out update with direct recomputation for both SWEET and LIONESS on a small simulated matrix. It exits successfully when both maximum absolute discrepancies are below `1e-10`.

```python
import numpy as np
from dcor_ssn import dcor_ssn_sweet

# Example abundance matrix: 24 samples, 6 metabolites
rng = np.random.default_rng(42)
X = rng.normal(size=(24, 6))
X[:, 1] = X[:, 0] ** 2 + 0.2 * rng.normal(size=len(X))
X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

scores, pairs = dcor_ssn_sweet(X)
top = np.argsort(-np.abs(scores[0]), kind="stable")[:3]
for edge in top:
    i, j = pairs[edge]
    print(i, j, scores[0, edge])
```

Supply a finite numeric matrix with samples in rows and metabolites in columns. `scores[s, e]` is the sample-network score and `pairs[e]` gives its two zero-based column indices. Each undirected pair occurs once. A score can be negative even with dCor. Use its absolute value to rank edges within a sample, as in the paper. To compare the three measures on the same matrix, use `pearson_ssn_sweet` and `spearman_ssn_sweet` with the same interface.

The default estimator uses the biased sample dCor, a leave-one-out perturbation, and SWEET parameters `K=0.1` and `eps=0.01`. Pass `return_full=True, return_weights=True` to obtain the cohort association matrix and the per-sample weights as the third and fourth return values. The function accepts preprocessed values. For held-out prediction, the cohort scripts below fit preprocessing and construct the network reference within the specified training split.

## Reproduce the manuscript analyses

| Question | Entry point | Manuscript output |
| --- | --- | --- |
| How do the three measures rank known edges in cohort and sample networks? | `python simulations/run_section_a.py --workers 4` | Figure 2, Tables S2 and S8 |
| Can network features distinguish five simulated dependence patterns? | `python simulations/run_section_b.py --workers 4`, then `python simulations/run_edge_attribution.py` | Figure 3, Tables S3, S9 and S11 |
| Which high-ranking edges recur within each measured cohort? | Run `cohorts/run_section_c.py` for each cohort, then its `summarize` command and `python figures/plot_figure4.py` | Figure 4, Table 1 and the concordance data for Tables S4 and S10 |
| How well do network features alone distinguish breast cancer cases and controls? | Prepare the workbook, run the nested analysis, and plot with `--include-baselines` | Figure 5, Table S5 and training-derived VIP results for Table S6 |
| Does adding network features improve internal discrimination over abundance and covariates? | Run the `prepare`, `run` and `summarize` commands of `cohorts/run_section_d.py` for each cohort | Table 2 |

The task guides give exact commands, file layouts, model settings and output names. See [simulations](simulations/README.md), [cohort analyses](cohorts/README.md) and [breast cancer classification and VIP](breast_cancer/README.md). The last guide documents which historical inputs are available for the age-restricted and pathway analyses in Figures S1 and S2.

The five-class experiment constructs networks from the full, unlabeled simulated cohort before cross-validation. Figure 5 uses the outer training cohort as the network reference, including its inner validation samples. Table 2 rebuilds the reference within each training fold and projects held-out individuals separately. Use the entry point for the intended protocol.

## Cohort data

Obtain the source workbooks from the data providers and place them in the local `data/` directory as described in the [cohort guide](cohorts/README.md). The simulations generate their own inputs.

| Cohort | Analysis matrix (samples × features) | Data source |
| --- | ---: | --- |
| Breast cancer plasma, ST000355 | 214 × 227 | [Metabolomics Workbench deposit](https://doi.org/10.21228/M86K6W) |
| Type 2 diabetes serum, ST003390 | 300 × 1,286 | [Metabolomics Workbench deposit](https://doi.org/10.21228/M81V7G), post-processed abundances and clinical baseline workbooks |
| Gastric cancer plasma | 521 × 146 | [Chen et al., Supplementary Data 1](https://doi.org/10.1038/s41467-024-46043-y), diagnostic worksheet. Section D also uses the authors’ [diagnostic model inputs](https://github.com/Yangzi-Chen2023/GC_NC-Res/tree/main/GC_diagnosis_model/Data) to verify case and control labels. |

## Results to check

At the highest simulated noise level, the group of ten quadratic anchor edges had mean top-121 inclusion rates of 54.1% for cohort dCor and 36.2% for dCor-SWEET. At the prespecified top 10% in ST003390, mean between-sample Jaccard similarity was 0.817 for dCor-SWEET and 0.864 for Spearman-SWEET. The mean paired AUROC differences after adding dCor-SSN to each cohort’s abundance and covariate baseline were +0.0033 for breast cancer, +0.0182 for type 2 diabetes and +0.0176 for gastric cancer. The prespecified gain required both a mean difference of at least +0.02 and a strictly positive lower limit of the participant-paired 95% bootstrap interval. None of the three cohorts met both conditions.
