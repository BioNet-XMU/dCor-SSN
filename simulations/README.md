# Simulations

Run the following commands from the repository root with the dependencies listed in the root README. Both simulations generate their input matrices from fixed seeds. Each full run covers six noise levels and 100 replicates per level.

| Manuscript analysis | Command | Main output |
| --- | --- | --- |
| Homogeneous dependence and cohort-level comparison (Figure 2, Tables S2 and S8) | `python simulations/run_section_a.py --workers 4` | `outputs/section_a/tables/summary.csv` and `paired_summary.csv`; `figures/Figure2_rank_and_top5.pdf` under the same output directory |
| Five-class discrimination and matched SWEET/LIONESS comparison (Figure 3, Tables S3 and S9) | `python simulations/run_section_b.py --workers 4` | `outputs/section_b/tables/summary.csv`, `paired_summary.csv`, `LIONESS_SI_table.csv` and `figures/Figure3_heterogeneous_sample_classification.pdf` |
| Single-edge on/off assignment (Table S11) | `python simulations/run_edge_attribution.py` after the five-class simulation | `outputs/section_b/edge_attribution/auroc_summary.csv` and `primary_decision.csv` |

Lower `--workers` to reduce peak memory. On machines using OpenBLAS, set `OPENBLAS_NUM_THREADS=1` before starting parallel runs. Both simulation drivers accept `--output DIR`. If you change the Section B output directory, pass `--input DIR/intermediate` and the desired `--output` directory to the single-edge analysis.

### Homogeneous dependence

Each replicate contains 200 samples and 70 variables. Three disjoint groups partition the 2,415 candidate edges into 150 dependent edges, 645 null edges that involve M7, and 1,620 null edges that join different indices outside M7. Five groups of ten M1-anchored edges distinguish the individual relationship types. For each association measure the same matrix yields a single cohort network and 200 SWEET networks. Ranking uses absolute edge scores, with ties resolved by upper-triangle order. Each group’s inclusion rate is the number of its edges among the first 121 divided by its own size. The figure summarizes normalized ranks, relationship types, and cohort versus sample dCor inclusion at the highest noise level.

### Heterogeneous dependence

Each replicate contains 250 samples, five equally sized classes, five shared backbone edges and five additional prespecified edges per class. The original simulation estimators in `five_class.py` reconstruct the three SWEET networks. The paired driver also builds Pearson- and dCor-LIONESS networks on the same unlabeled samples. Each sample is represented by the mean absolute-score rank in each of the five class-specific edge groups, with average ranks for ties. The evaluator uses stratified five-fold cross-validation, fold-wise standardization, and L2 logistic regression. Shared-backbone and shuffled-label controls use the same evaluation procedure.

The `intermediate/sigma_*.npz` files store the generated inputs, class labels, fold assignments, edge scores, ranks, and out-of-fold predictions. The single-edge analysis reads these files to compare 50 active samples with 200 inactive samples for each target edge. It reports AUROC and paired method differences across replicates, with percentile intervals from 10,000 bootstrap resamples of the replicates.

The original SWEET-only five-class workflow is also available as `python simulations/five_class.py`. The historical comparison of the two runs matched all primary balanced accuracies, macro-F1 values, class recalls, confusion counts and label-permutation scores. For the shared-backbone-only control, 222 of 1,800 replicate scores differed, with a maximum absolute difference of 0.00044 in a method-by-noise mean. The original workflow writes to `outputs/section_b_original/`.
