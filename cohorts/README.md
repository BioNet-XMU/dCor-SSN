# Three-cohort analyses

Section C measures the concordance of sample-specific edge sets in a pooled cohort. Section D tests the incremental discrimination obtained by adding dCor-SWEET network features to a covariate-plus-abundance PLS-logistic model. Run the commands below from the repository root using the environment specified there.

## Source workbooks

Place the following files below `data/`, or pass the parent of the three cohort folders with `--data-dir`. Keep the default workbook names and sheets.

```text
data/
  ST000355/
    COH plasma data.xlsx
  ST003390/
    YLab_T2DM_Metabolomics Post-Processed Data.xlsx
    YLab_T2DM_Clinical Baseline Information.xlsx
  GC/
    41467_2024_46043_MOESM5_ESM.xlsx
    model_inputs/
      Xtrain.xlsx
      ytrain.xlsx
      Xtest.xlsx
      ytest.xlsx
      Xexternal_test.xlsx
      yexternal_test.xlsx
```

- **Breast cancer, ST000355.** Use the COH plasma workbook, sheet `Sheet1`. Sample IDs are in Excel row 1, columns I onward. Clinical field names occupy column A in rows 1–17, including `Group (1/0)` and `Age (BOD-OD)`. Measurements start in row 18, with annotation IDs in column B, names in C and abundances from I onward. Retain named features, suffix duplicate names in occurrence order, and preserve reported zeros. The final matrix contains 138 cases and 76 controls by 227 features. Within each group, sample order follows the sheet. Section C places cases first and Section D places controls first, matching the published analyses. [Metabolomics Workbench ST000355](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST000355)
- **Type 2 diabetes, ST003390.** Extract the two named workbooks from the study download. The post-processed workbook contains six sheets, M1 through M6, with metabolite names in `Metabolite Name`, optional `HMDB ID`, and one column per participant or QC sample. The loader joins participant columns to `Sheet1` of the clinical workbook by `Sample ID` and uses `Age`, `Gender` and `Group`. It places group 2 controls before group 1 cases, preserving clinical sheet order within each group. QC columns and rows with no participant measurements are excluded. M1–M6 contribute 103, 98, 69, 70, 165 and 781 features, respectively. Both M4 bilirubin measurements remain. The input contains 200 controls and 100 cases by 1,286 features, using the authors’ filled values in the source units. [Metabolomics Workbench ST003390](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST003390)
- **Gastric cancer, GC.** Download Supplementary Data 1 from Chen et al. The loader reads the `MS dataset for diagnosis` sheet, whose first row contains measurement names and first column contains sample IDs. An extracted copy of that sheet named `data_diagnosis.xlsx` is also accepted. The analysis orders IDs lexicographically, excludes the unnamed measurement at Excel column 70, and retains 521 participants by 146 named features. Section C requires only this workbook. [Supplementary Data 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10891053/)
- **GC diagnosis labels for Section D.** Copy the six `X*.xlsx` and `y*.xlsx` files above from the authors’ model repository. Each X file has an identifier in its first column, 147 measurements in the middle and a final metadata column. Each y file has an identifier and a `state` column. The loader verifies each participant against all 147 published values, joins the labels by the corresponding IDs, and verifies the P/N suffix and the 22 batch prefixes. This produces 208 cases and 313 controls. To write the audit tables separately, run `python cohorts/audit_gc.py`. Section D preparation performs the same checks automatically. [Published diagnosis-model inputs](https://github.com/Yangzi-Chen2023/GC_NC-Res/tree/main/GC_diagnosis_model/Data)

## Section C · Edge concordance

```sh
python cohorts/run_section_c.py ST000355
python cohorts/run_section_c.py ST003390
python cohorts/run_section_c.py GC
python cohorts/run_section_c.py summarize
```

Each run standardizes each metabolite with its pooled population mean and SD and computes dCor-, Pearson- and Spearman-SWEET leave-one-out scores (`K=0.1`, `eps=0.01`). Within each sample and method, it retains the largest absolute scores using `ceil(q × number_of_edges)` for q = 5%, 10%, 15%, 20%, 30%, 40% and 50%. The primary q is 10%. Between-sample Jaccard is averaged over all unordered sample pairs. Cross-method Jaccard compares edge sets within the same sample. An edge is highly recurrent when at least half the samples retain it.

The default output is `outputs/section_c/`. Per-cohort folders include input registration, feature/sample order and intermediate arrays. Packed masks have axes (q, sample, byte); decode with `np.unpackbits(mask, axis=-1, count=n_edges)`. Edges and sample pairs follow the zero-based upper triangle. The final command pools summaries in `tables/`, including `primary_q10.csv` and `high_recurrence_partition.csv`, and writes the seven-fraction scan in `figures/`. Use a consistent `--output-dir` on every command to change the destination.

## Section D · Paired discrimination

```sh
python cohorts/run_section_d.py check
python cohorts/run_section_d.py prepare --cohort ST000355
python cohorts/run_section_d.py run --cohort ST000355
python cohorts/run_section_d.py summarize --cohort ST000355
python cohorts/run_section_d.py prepare --cohort ST003390
python cohorts/run_section_d.py run --cohort ST003390
python cohorts/run_section_d.py summarize --cohort ST003390
python cohorts/run_section_d.py prepare --cohort GC
python cohorts/run_section_d.py run --cohort GC
python cohorts/run_section_d.py summarize --cohort GC
```

`check` runs two small synthetic comparisons of the optimized training/held-out edge masks and row-space PLS-LR predictions against direct calculations. `prepare` reproduces 100 stratified 70/30 splits with seed 42 and five shuffled inner folds per outer training set, seeded with 42 plus the one-based split number. Both arms use the same participants and splits. The baseline includes abundance and age for ST000355, abundance, age and sex for ST003390, and abundance and batch indicators for GC.

The augmented arm adds binary dCor-SWEET features from the top 10% of absolute per-sample scores. For each inner or outer fit, training participants determine the scaling, reference network, candidate edge union and model. Each held-out participant is projected against the training reference individually. The two arms independently select 1–8 PLS components by mean inner-fold AUROC, breaking ties at 12 decimal places in favor of the smaller component count. PLS scores feed logistic regression with `C=1`.

`run` saves each outer split atomically and resumes completed splits after an interruption. `summarize` requires all 100 splits. It averages the splitwise difference (augmented AUROC minus baseline AUROC), then recomputes that mean on 10,000 participant bootstrap samples with seed 20261122. Participant weights are shared across arms and splits and held-out predictions are fixed. The reported limits are percentile 95% intervals. The prespecified gain requires a mean increment of at least 0.02 and a lower limit above zero.

Results are in `outputs/section_d/<cohort>/`. `summary.json`, `paired_split_aurocs.csv`, `inner_tuning.csv` and `subject_bootstrap.npz` give the aggregate and split-level results. `inputs_and_splits.npz` and `paired_test_predictions.csv` contain participant-level data. Keep generated cohort outputs in the local output directory.

## Runtime, memory and numerical checks

Section C's largest study (ST003390) has 826,255 candidate edges and used approximately 7.5 GiB peak resident memory in the reference run. Allow additional memory for the operating system. Section D defaults to one worker. After `prepare`, run `python cohorts/run_section_d.py benchmark --cohort ST003390` to time the first complete nested split on your machine. The reference trial took about 108 seconds and 2.03 GiB with four OpenBLAS threads. The benchmark saves split 001 for reuse by `run`. Use `--workers N` to parallelize outer splits if available memory and cores permit. Set BLAS thread counts before launching Python, for example `OPENBLAS_NUM_THREADS=4` on a four-thread OpenBLAS run. Installation, hardware and thread counts affect timing and floating-point rounding.

| Cohort | Mean Jaccard at 10% (dCor / Pearson / Spearman) | Mean paired ΔAUROC (95% interval) |
| --- | --- | --- |
| ST000355 | 0.723 / 0.614 / 0.617 | +0.0033 (−0.0142, +0.0260) |
| ST003390 | 0.817 / 0.664 / 0.864 | +0.0182 (+0.0017, +0.0407) |
| GC | 0.638 / 0.597 / 0.586 | +0.0176 (+0.0064, +0.0299) |
