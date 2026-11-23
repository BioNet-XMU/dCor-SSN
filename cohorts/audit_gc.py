"""Match each gastric cohort profile to its published diagnosis label."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_data import GC_SHEET, ROOT, gastric_workbook, identify_sources, write_json


def verify_labels(source, model_dir):
    local = pd.read_excel(source, sheet_name=GC_SHEET)
    local_ids = local.iloc[:, 0].astype(str)
    values = local.iloc[:, 1:].to_numpy(float)
    if values.shape != (521, 147) or not local_ids.is_unique:
        raise ValueError("Expected 521 unique profiles with 147 measurements")
    lookup = {tuple(row[:12]): i for i, row in enumerate(values)}
    if len(lookup) != len(local_ids):
        raise ValueError("The matching key does not uniquely identify diagnostic profiles")
    matches, paths = [], []
    for split in ("train", "test", "external_test"):
        x_file, y_file = model_dir / f"X{split}.xlsx", model_dir / f"y{split}.xlsx"
        paths.extend([x_file, y_file])
        x, y = pd.read_excel(x_file), pd.read_excel(y_file)
        full = x.iloc[:, 1:-1].to_numpy(float)
        rows = [lookup[tuple(row[:12])] for row in full]
        np.testing.assert_array_equal(full, values[rows])
        ids = local_ids.iloc[rows].to_numpy()
        codes = ids if split == "external_test" else x.iloc[:, 0].astype(str).to_numpy()
        labels = y.set_index(y.iloc[:, 0].astype(str), verify_integrity=True)
        if set(labels.index) != set(codes):
            raise ValueError(f"Published labels do not match the {split} profile IDs")
        matches.append(pd.DataFrame({"sample_id": ids, "state": labels.loc[codes, "state"].to_numpy(),
                                     "published_split": split}))
    labels = pd.concat(matches).set_index("sample_id", verify_integrity=True).loc[local_ids]
    labels.index.name = "sample_id"
    batch = local_ids.str.extract(r"^(Batch[0-9]{2})_", expand=False)
    if batch.isna().any() or batch.nunique() != 22 or not local_ids.str.endswith(("P", "N")).all():
        raise ValueError("Unexpected gastric cohort batch or status coding")
    np.testing.assert_array_equal(local_ids.str.endswith("P").astype(int), labels.state)
    labels["batch"] = batch.to_numpy()
    return labels, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "section_d" / "gc_audit")
    args = parser.parse_args()
    gc_dir = args.data_dir / "GC"
    source = gastric_workbook(gc_dir)
    verified, paths = verify_labels(source, gc_dir / "model_inputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verified.to_csv(args.output_dir / "verified_sample_batch_labels.csv")
    counts = pd.crosstab(verified.batch, verified.state).rename(columns={0: "controls", 1: "cases"})
    counts.to_csv(args.output_dir / "batch_label_counts.csv")
    write_json(args.output_dir / "eligibility.json", {
        "n_samples": len(verified), "num_batches": len(counts),
        "batches_with_no_cases": int(counts.cases.eq(0).sum()),
        "source_sha256": identify_sources([source, *paths]),
    })
    print(f"Verified {len(verified)} complete profiles and labels across {len(counts)} batches")


if __name__ == "__main__":
    main()
