from __future__ import annotations

import argparse
from pathlib import Path

from rlmf.inference import DATASETS, evaluate_dataset
from rlmf.metrics import format_metrics


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a labeled 41-nt CSV with a released ensemble")
    parser.add_argument("--dataset", required=True, choices=DATASETS, help="Checkpoint ensemble to use")
    parser.add_argument("--input", required=True, type=Path, help="CSV containing sequence and label columns")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prediction_results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = evaluate_dataset(
        dataset=args.dataset,
        data_path=args.input,
        weights_root=ROOT / "weights",
        model_dir=ROOT / "pretrained" / "rna_language_model",
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(format_metrics(result["metrics"]))


if __name__ == "__main__":
    main()
