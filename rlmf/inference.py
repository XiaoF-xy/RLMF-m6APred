from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .features import feature_matrix
from .metrics import compute_binary_metrics, json_safe_metrics
from .model import RLMFm6APred


DATASETS = ("H_b", "H_k", "H_l", "M_b", "M_h", "M_k", "M_l", "M_t", "R_b", "R_k", "R_l")
FOLDS = 5
SEQUENCE_LENGTH = 41
THRESHOLD = 0.5


@dataclass(frozen=True)
class Sample:
    sequence: str
    label: int


def normalize_sequence(sequence: str) -> str:
    sequence = "".join(str(sequence).upper().split()).replace("U", "T")
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected a {SEQUENCE_LENGTH}-nt sequence, got {len(sequence)}")
    invalid = sorted(set(sequence) - set("ACGT"))
    if invalid:
        raise ValueError(f"Invalid nucleotide symbols: {invalid}")
    return sequence


def read_labeled_csv(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"sequence", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain sequence and label columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                sequence = normalize_sequence(row["sequence"])
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row {line_number} in {path}: {exc}") from exc
            if label not in (0, 1):
                raise ValueError(f"Invalid label at row {line_number}: {label}")
            samples.append(Sample(sequence, label))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


class _SequenceDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class _Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[Sample]) -> dict:
        sequences = [sample.sequence for sample in batch]
        encoded = self.tokenizer(
            [" ".join(sequence) for sequence in sequences],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        encoded["handcrafted_features"] = torch.tensor(
            np.stack([feature_matrix(sequence) for sequence in sequences]),
            dtype=torch.float32,
        )
        encoded["labels"] = torch.tensor([sample.label for sample in batch], dtype=torch.long)
        encoded["sequences"] = sequences
        return encoded


def checkpoint_paths(weights_root: Path, dataset: str) -> list[Path]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of: {', '.join(DATASETS)}")
    paths = [Path(weights_root) / dataset / f"fold_{fold:02d}.pt" for fold in range(1, FOLDS + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing release checkpoints: " + ", ".join(missing))
    return paths


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid release checkpoint: {path}")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"Checkpoint is not tensor-only: {path}")
    return state


@torch.inference_mode()
def _predict_fold(model: RLMFm6APred, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for batch in loader:
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "labels"
        }
        logits = model(**inputs)["logits"]
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def evaluate_dataset(
    *,
    dataset: str,
    data_path: Path,
    weights_root: Path,
    model_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> dict:
    samples = read_labeled_csv(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer.model_max_length = 64
    loader = DataLoader(
        _SequenceDataset(samples),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=_Collator(tokenizer),
    )
    model = RLMFm6APred(model_dir).to(device)
    fold_probabilities = []
    for fold, path in enumerate(checkpoint_paths(weights_root, dataset), start=1):
        model.load_release_state_dict(_load_state(path))
        fold_probabilities.append(_predict_fold(model, loader, device))
        print(f"{dataset}: fold {fold}/{FOLDS} complete", flush=True)
    probabilities = np.mean(np.stack(fold_probabilities), axis=0)
    labels = np.asarray([sample.label for sample in samples], dtype=int)
    metrics = compute_binary_metrics(labels, probabilities, THRESHOLD)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{dataset}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sequence", "label", "probability", "prediction"])
        for sample, probability in zip(samples, probabilities):
            writer.writerow([sample.sequence, sample.label, float(probability), int(probability >= THRESHOLD)])
    payload = {
        "dataset": dataset,
        "device": str(device),
        "samples": len(samples),
        "ensemble": "mean probability across five released fold models",
        "threshold": THRESHOLD,
        "metrics": metrics,
    }
    with (output_dir / f"{dataset}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe_metrics(payload), handle, indent=2, ensure_ascii=False)
    return payload
