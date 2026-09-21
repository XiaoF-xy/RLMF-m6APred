from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("H_b", "H_k", "H_l", "M_b", "M_h", "M_k", "M_l", "M_t", "R_b", "R_k", "R_l")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    failures: list[str] = []
    weight_rows = read_manifest(ROOT / "manifests" / "weights_sha256.csv")
    if len(weight_rows) != 55:
        failures.append(f"expected 55 weights, found {len(weight_rows)}")
    for row in weight_rows:
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            failures.append(f"weight hash mismatch: {row['path']}")
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict) or not state or any(
            not torch.is_tensor(value) for value in state.values()
        ):
            failures.append(f"weight is not tensor-only: {row['path']}")
    for row in read_manifest(ROOT / "manifests" / "data_sha256.csv"):
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            failures.append(f"data hash mismatch: {row['path']}")
    for dataset in DATASETS:
        if not (ROOT / "data" / "internal_benchmark" / f"{dataset}.csv").is_file():
            failures.append(f"missing benchmark: {dataset}")
        if not (ROOT / "data" / "independent_test" / f"{dataset}.csv").is_file():
            failures.append(f"missing independent test: {dataset}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("Release verification passed: 55 tensor-only weights and all data hashes are valid.")


if __name__ == "__main__":
    main()
