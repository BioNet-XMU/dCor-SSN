"""Reconstruct the study matrices, in analysis order, from the source workbooks."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("ST000355", "ST003390", "GC")
SHAPES = dict(zip(COHORTS, ((214, 227), (300, 1286), (521, 146))))
GC_SHEET = "MS dataset for diagnosis"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def identify_sources(paths):
    records = {}
    for path in paths:
        with path.open("rb") as handle:
            records[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return records


def unique_names(values):
    seen, result = {}, []
    for name in values:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


def breast(data, prediction):
    source = data / "COH plasma data.xlsx"
    raw = pd.read_excel(source, sheet_name="Sheet1", header=None)
    ids = raw.iloc[0, 8:].astype(str)
    clinical = raw.iloc[:17, 8:].set_axis(raw.iloc[:17, 0].astype(str), axis=0)
    groups = pd.to_numeric(clinical.loc["Group (1/0)"], errors="coerce")
    # The pooled concordance and discrimination analyses used different row orders.
    first = 0 if prediction else 1
    cols = [*groups.index[groups.eq(first)], *groups.index[groups.eq(1 - first)]]
    names = raw.iloc[17:, 2].dropna().astype(str)
    names = names.loc[~names.str.strip().str.lower().isin(["", "nan", "none"])]
    features = pd.DataFrame({
        "feature": unique_names(names), "original_name": names.to_numpy(),
        "excel_row": names.index + 1, "annotation_id": raw.loc[names.index, 1].to_numpy(),
    })
    x = raw.loc[names.index, cols].to_numpy(dtype=float).T
    age = clinical.loc["Age (BOD-OD)", cols].to_numpy(dtype=float)[:, None]
    return x, ids.loc[cols], features, groups.loc[cols].to_numpy(int), age, ["age"], [source]


def diabetes(data):
    source = data / "YLab_T2DM_Metabolomics Post-Processed Data.xlsx"
    clinical_file = data / "YLab_T2DM_Clinical Baseline Information.xlsx"
    clinical = pd.read_excel(clinical_file, sheet_name="Sheet1",
                             usecols=["Sample ID", "Age", "Group", "Gender"]).dropna(how="all")
    clinical["Sample ID"] = clinical["Sample ID"].astype(str).str.strip()
    clinical = clinical.set_index("Sample ID", verify_integrity=True)
    if set(clinical.Group) != {1, 2}:
        raise ValueError("Expected group 1 (T2DM) and 2 (healthy) in the clinical workbook")
    clinical = clinical.sort_values("Group", ascending=False, kind="stable")
    ids = clinical.index
    sex = clinical.Gender.str.strip().str.lower()
    if set(sex) != {"male", "female"}:
        raise ValueError("Unrecognized gender in the clinical workbook")
    features, blocks = [], []
    with pd.ExcelFile(source) as workbook:
        for module in ("M1", "M2", "M3", "M4", "M5", "M6"):
            sheet = workbook.parse(module)
            participant_ids = [col for col in sheet.columns if re.fullmatch(r"[DH]\d+", str(col))]
            if set(participant_ids) != set(ids):
                raise ValueError(f"{module} participant columns differ from the clinical IDs")
            values = sheet.loc[:, ids].dropna(how="all")
            rows = sheet.loc[values.index]
            features.append(pd.DataFrame({
                "original_name": rows["Metabolite Name"], "module": module,
                "source_row": rows.index + 2, "annotation_id": rows.get("HMDB ID", ""),
            }))
            blocks.append(values.to_numpy(dtype=float))
    features = pd.concat(features, ignore_index=True)
    features.insert(0, "feature", unique_names(features.original_name))
    cov = np.column_stack([clinical.Age.to_numpy(float), sex.eq("male").to_numpy(float)])
    return (np.vstack(blocks).T, ids, features, clinical.Group.eq(1).to_numpy(int), cov,
            ["age", "male"], [source, clinical_file])


def gastric_workbook(data):
    published = data / "41467_2024_46043_MOESM5_ESM.xlsx"
    return published if published.is_file() else data / "data_diagnosis.xlsx"


def gastric(data, prediction):
    source = gastric_workbook(data)
    raw = pd.read_excel(source, sheet_name=GC_SHEET, header=None)
    headers = raw.iloc[0, 1:]
    if headers.index[headers.isna()].tolist() != [69]:
        raise ValueError("Expected one unnamed measurement at Excel column 70")
    names = headers.dropna()
    features = pd.DataFrame({"feature": names.to_numpy(), "original_name": names.to_numpy(),
                             "excel_column": names.index + 1})
    # The analysis worksheet was sorted lexicographically by participant ID.
    ids = raw.iloc[1:, 0].astype(str).sort_values(kind="stable")
    x = raw.loc[ids.index, names.index].to_numpy(float)
    y, cov, cov_names, sources = None, None, [], [source]
    if prediction:
        from audit_gc import verify_labels
        verified, audit_sources = verify_labels(source, data / "model_inputs")
        verified = verified.loc[ids]
        y = verified["state"].to_numpy(int)
        batch = pd.get_dummies(verified["batch"], dtype=float)
        cov, cov_names = batch.to_numpy(), batch.columns.tolist()
        sources.extend(audit_sources)
    return x, ids, features, y, cov, cov_names, sources


def load_cohort(cohort, data_dir, *, prediction):
    directory = Path(data_dir) / cohort
    if cohort == "ST000355":
        fields = breast(directory, prediction)
    elif cohort == "ST003390":
        fields = diabetes(directory)
    elif cohort == "GC":
        fields = gastric(directory, prediction)
    else:
        raise ValueError(cohort)
    x, ids, features, y, cov, cov_names, sources = fields
    if x.shape != SHAPES[cohort] or not np.isfinite(x).all():
        raise ValueError(f"{cohort} requires a complete {SHAPES[cohort]} abundance matrix, got {x.shape}")
    if not pd.Index(ids).is_unique or not features.feature.is_unique or (x.std(axis=0) == 0).any():
        raise ValueError(f"Duplicated IDs, feature names or constant abundances in {cohort}")
    info = {"cohort": cohort, "samples": len(x), "features": x.shape[1],
            "n_edges": x.shape[1] * (x.shape[1] - 1) // 2,
            "missing_at_input": int(np.isnan(x).sum()), "zeros_at_input": int((x == 0).sum()),
            "source_sha256": identify_sources(sources)}
    if prediction:
        if not np.isfinite(cov).all() or set(y) != {0, 1}:
            raise ValueError(f"Invalid clinical data in {cohort}")
        info.update(cases=int(y.sum()), controls=int((y == 0).sum()), covariates=cov_names)
    return x, np.asarray(ids, dtype=str), features, y, cov, info
