from __future__ import annotations

from collections import Counter

import numpy as np


FEATURE_ORDER = ("onehot", "chemical", "eiip", "enac")
FEATURE_CHANNELS = (4, 4, 1, 4)

_BASE_PROPERTIES = {
    "A": (1.0, 1.0, 1.0),
    "C": (0.0, 1.0, 0.0),
    "G": (1.0, 0.0, 0.0),
    "T": (0.0, 0.0, 1.0),
    "N": (0.0, 0.0, 0.0),
}
_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
_EIIP_VALUES = {"A": 0.1260, "C": 0.1340, "G": 0.0806, "T": 0.1335, "N": 0.0}


def _normalize_sequence(sequence: str) -> str:
    normalized = str(sequence).upper().replace("U", "T")
    return "".join(base if base in _BASE_PROPERTIES else "N" for base in normalized)


def onehot4_encode(sequence: str) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    features = np.zeros((4, len(sequence)), dtype=np.float32)
    for position, base in enumerate(sequence):
        if base in _BASE_TO_INDEX:
            features[_BASE_TO_INDEX[base], position] = 1.0
    return features


def chemical4_encode(sequence: str) -> np.ndarray:
    """Return the public repository's 3 NCP properties plus cumulative frequency."""
    sequence = _normalize_sequence(sequence)
    counts = Counter()
    rows: list[list[float]] = []
    for position, base in enumerate(sequence, start=1):
        counts[base] += 1
        rows.append([
            *_BASE_PROPERTIES[base],
            float(np.round(counts[base] / position, 3)),
        ])
    if not rows:
        return np.zeros((4, 0), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32).T


def eiip1_encode(sequence: str) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    return np.asarray([[_EIIP_VALUES[base] for base in sequence]], dtype=np.float32)


def enac4_encode(sequence: str, window_size: int = 5) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got: {window_size}")
    sequence = _normalize_sequence(sequence)
    radius = window_size // 2
    padded = "N" * radius + sequence + "N" * radius
    features = np.zeros((4, len(sequence)), dtype=np.float32)
    for position in range(len(sequence)):
        counts = Counter(padded[position:position + window_size])
        for base, channel in _BASE_TO_INDEX.items():
            features[channel, position] = counts[base] / window_size
    return features


def feature_matrix(sequence: str, enac_window_size: int = 5) -> np.ndarray:
    sequence = _normalize_sequence(sequence)
    features = (
        onehot4_encode(sequence),
        chemical4_encode(sequence),
        eiip1_encode(sequence),
        enac4_encode(sequence, window_size=enac_window_size),
    )
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)
