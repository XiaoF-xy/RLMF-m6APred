from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rlmf.inference import DATASETS, evaluate_dataset
from rlmf.metrics import format_metrics


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce RLMF-m6APred independent-test results from released weights"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASETS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_results")
    return parser.parse_args()


def expected_results() -> dict[str, dict[str, float]] | None:
    path = ROOT / "expected_results" / "independent_test_metrics.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["dataset"]: {
                key: float(row[key]) for key in ("ACC", "MCC", "AUC", "AUPRC")
            }
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    datasets = DATASETS if args.all else (args.dataset,)
    expected = expected_results()
    failed = []
    for dataset in datasets:
        result = evaluate_dataset(
            dataset=dataset,
            data_path=ROOT / "data" / "independent_test" / f"{dataset}.csv",
            weights_root=ROOT / "weights",
            model_dir=ROOT / "pretrained" / "rna_language_model",
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        metrics = result["metrics"]
        if expected is None:
            print(f"{dataset}: {format_metrics(metrics)}")
            continue
        deltas = {
            key: abs(float(metrics[key]) - expected[dataset][key])
            for key in expected[dataset]
        }
        status = "REPRODUCED" if max(deltas.values()) <= 1e-5 else "MISMATCH"
        print(f"{dataset}: {format_metrics(metrics)} [{status}]")
        if status != "REPRODUCED":
            failed.append((dataset, deltas))
    if failed:
        details = "; ".join(f"{dataset}: {deltas}" for dataset, deltas in failed)
        raise SystemExit("Released-weight reproduction failed: " + details)


if __name__ == "__main__":
    main()
