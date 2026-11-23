# Breast cancer network classification and VIP analysis

These scripts generate Figure 5 and training-model variable importance in projection (VIP) scores for study ST000355. For each outer split, the network-only models predict case–control status from binary indicators of high-ranking edges. The separate paired analysis in the paper adds network features to an abundance and covariate baseline.

## Input and commands

Place the public `COH plasma data.xlsx` workbook in `data/ST000355/` under the repository root. Use the original `Sheet1` with its sample identifiers, group labels, and metabolite names. The following commands run from the repository root.

```sh
python breast_cancer/build_classification_input.py
python breast_cancer/run_new_plslr_sweet_rankunion_nestedcv.py
python breast_cancer/make_figure5_classification_performance.py --include-baselines
```

The preparation step accepts `--source PATH` for a workbook stored elsewhere. It writes `breast_cancer/inputs/v5_analysis_data_raw.npz` and a manifest. For the published input, expect 214 samples (138 cases, 76 controls), 227 metabolites, and an array-content SHA-256 of `3ac83470b2689cb208d92a11c3e6dd9ebc801b50a776cbfae5265c7a08e03007`.

## Protocol and outputs

The run uses 100 stratified 70/30 outer splits (seed 42), 100 label-permutation controls, and ten edge fractions from 2% to 20%. Each fraction is evaluated separately. Within each outer training set, the script standardizes abundances, builds training networks, and projects each outer test sample individually onto that reference. Inner fivefold validation chooses the number of PLS components by AUROC, with binary-feature scaling fitted inside each inner training fold. Inner validation samples contribute to the outer training network reference. The paired increment experiment uses a separate fold-specific network protocol.

Outputs are written under `breast_cancer/outputs/`.

- `new_plslr_results.csv` and `new_plslr_summary.csv` contain the split-level and summarized network-only results.
- `stable_vip_networks/` contains per-split VIP edges, their frequency among outer training fits, and node and network summaries. The primary exploratory graph uses 200 edges per method and edge fraction. `stable_hubs_across_q.csv` summarizes recurrence of highly ranked hubs.
- `figures/` contains Figure 5 and the aggregate values plotted in each panel. With `--include-baselines`, panel A adds the reported age-only (0.949) and abundance-only (0.965) mean AUROCs from the original analysis. These two values are fixed figure inputs. Panel B averages the ten fractions within each permutation before plotting.

To check the asparagine–serine example in the paper, select this metabolite pair in the VIP output. The published counts of outer training models placing the pair among their top 500 VIP edges were 95 for dCor, 9 for Pearson, and 32 for Spearman.

```python
import pandas as pd

edges = pd.read_csv("breast_cancer/outputs/stable_vip_networks/stable_vip_edges_all.csv")
pair = ["Asparagine", "Serine"]
selected = edges["q"].eq(0.10) & edges["node_a"].isin(pair) & edges["node_b"].isin(pair)
print(edges.loc[selected, ["method", "top_vip_count"]])
```

For the full run, the default is eight parallel workers. When memory is limited, set the environment variable `N_JOBS=1`. To rebuild the figure from an existing pair of result CSVs, pass `--input-dir PATH --output-dir PATH --include-baselines` to the plotting script.

The historical archive lacks the individual membership and matching rule for age-restricted subset A in Figure S1. The stable hubs can be regenerated from training models. The original pathway service export for the exploratory enrichment in Figure S2 was unavailable for independent reconstruction.
