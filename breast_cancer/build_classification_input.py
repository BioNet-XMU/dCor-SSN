#!/usr/bin/env python3
"""Build the unstandardized ST000355 matrix used by the classification workflow.

The output contains raw abundances only. Standardization is deliberately deferred
to each outer training split in the main analysis script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "ST000355" / "COH plasma data.xlsx"
DEFAULT_OUTPUT = BASE / "inputs" / "v5_analysis_data_raw.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, anchor: Path) -> str:
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return str(path)


def array_content_sha256(
    x: np.ndarray, y: np.ndarray, metabolite_names: list[str], sample_ids: list[str]
) -> str:
    """Hash scientific array content without depending on ZIP timestamps or paths."""
    digest = hashlib.sha256()
    for name, array in (("X", x), ("y", y)):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    for name, values in (("metabolite_names", metabolite_names), ("sample_ids", sample_ids)):
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def load_raw_breast(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    sample_ids = raw.iloc[0, 8:].astype(str).tolist()

    clinical = raw.iloc[0:17, 8:].copy()
    clinical.index = raw.iloc[0:17, 0].astype(str).tolist()
    clinical.columns = sample_ids
    clinical = clinical.T

    metabolite_names = raw.iloc[17:, 2].astype(str).tolist()
    abundance = raw.iloc[17:, 8:].apply(pd.to_numeric, errors="coerce")
    valid = np.array(
        [str(name).strip().lower() not in ("", "nan", "none") for name in metabolite_names]
    )
    abundance = abundance.loc[valid].copy()
    metabolite_names = [name for name, keep in zip(metabolite_names, valid) if keep]

    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in metabolite_names:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    abundance.index = unique_names

    abundance.columns = sample_ids
    if abundance.isna().any().any():
        medians = abundance.median(axis=1)
        abundance = abundance.T.fillna(medians).T.fillna(0.0)

    group = clinical["Group (1/0)"].astype(str).str.strip()
    case_ids = [sid for sid, value in zip(clinical.index.astype(str), group) if value in ("1", "1.0")]
    control_ids = [sid for sid, value in zip(clinical.index.astype(str), group) if value in ("0", "0.0")]
    ordered_ids = control_ids + case_ids

    x = abundance[ordered_ids].T.to_numpy(dtype=float)
    y = np.array([0] * len(control_ids) + [1] * len(case_ids), dtype=int)
    return x, y, unique_names, ordered_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Raw ST000355 workbook not found: {source}")

    x, y, metabolite_names, sample_ids = load_raw_breast(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        X=x,
        y=y,
        metabolite_names=np.asarray(metabolite_names, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        source=str(source),
        preprocessing="raw_abundance_no_all_sample_zscore",
    )
    manifest = {
        "source": portable_path(source, PROJECT_ROOT),
        "source_sha256": sha256(source),
        "output": portable_path(output, BASE),
        "output_sha256": sha256(output),
        "scientific_content_sha256": array_content_sha256(x, y, metabolite_names, sample_ids),
        "shape": list(x.shape),
        "n_case": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "preprocessing": "raw abundance; no all-sample standardization",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
