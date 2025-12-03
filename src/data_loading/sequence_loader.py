from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    "hadoop": {
        "dir": "hadoop_loghub",
        "prefix": "hadoop",
    },
    "openstack": {
        "dir": "openstack_loghub",
        "prefix": "openstack",
    },
    "adfa": {
        "dir": "adfa_verazuo",
        "prefix": "adfa",
    },
}


@dataclass
class DatasetSplit:
    sequences: List[List[int]]
    labels: List[str]


@dataclass
class DatasetBundle:
    name: str
    train: DatasetSplit
    test: DatasetSplit
    test_by_type: Optional[Dict[str, DatasetSplit]]
    stats: Dict[str, Any]


def load_sequences(file_path: str) -> List[List[int]]:
    """Load space-separated integer event id sequences from a text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence file: {path}")

    sequences: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            raw_tokens = line.split()
            if "," in raw_tokens[0]:
                _, first_event = raw_tokens[0].split(",", 1)
                raw_tokens = [first_event] + raw_tokens[1:]
            tokens = [int(token) for token in raw_tokens]
            sequences.append(tokens)
    return sequences


def _build_paths(dataset_name: str, base_dir: str) -> Dict[str, Path]:
    cfg = DATASET_CONFIG[dataset_name]
    dataset_dir = Path(base_dir).expanduser().resolve() / cfg["dir"]
    prefix = cfg["prefix"]
    return {
        "train": dataset_dir / f"{prefix}_train",
        "test_normal": dataset_dir / f"{prefix}_test_normal",
        "test_abnormal": dataset_dir / f"{prefix}_test_abnormal",
        "dataset_dir": dataset_dir,
        "prefix": prefix,
    }


def _compute_length_stats(sequences: List[List[int]]) -> Dict[str, float]:
    if not sequences:
        return {
            "count": 0,
            "avg_len": 0.0,
            "max_len": 0,
            "total_events": 0,
        }
    lengths = [len(seq) for seq in sequences]
    total_events = sum(lengths)
    return {
        "count": len(sequences),
        "avg_len": total_events / len(sequences),
        "max_len": max(lengths),
        "total_events": total_events,
    }


def build_dataset(dataset_name: str, base_dir: str) -> DatasetBundle:
    """
    Load train/test splits for a dataset from the AIT repository layout.

    Returns a DatasetBundle with sequences, labels, and basic stats.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Options: {list(DATASET_CONFIG)}")

    paths = _build_paths(dataset_name, base_dir)
    train_sequences = load_sequences(paths["train"])
    test_normal_sequences = load_sequences(paths["test_normal"])
    test_abnormal_sequences = load_sequences(paths["test_abnormal"])

    test_sequences = test_normal_sequences + test_abnormal_sequences
    test_labels = ["normal"] * len(test_normal_sequences) + ["anomaly"] * len(test_abnormal_sequences)

    test_by_type: Dict[str, DatasetSplit] = {}
    subtype_prefix = f"{paths['prefix']}_test_abnormal_"
    for subtype_file in sorted(paths["dataset_dir"].glob(f"{subtype_prefix}*")):
        subtype = subtype_file.name.replace(subtype_prefix, "")
        subtype_sequences = load_sequences(str(subtype_file))
        test_by_type[subtype] = DatasetSplit(
            sequences=subtype_sequences,
            labels=[subtype] * len(subtype_sequences),
        )

    stats = {
        "dataset": dataset_name,
        "n_train": len(train_sequences),
        "n_test": len(test_sequences),
        "n_test_normal": len(test_normal_sequences),
        "n_test_anomaly": len(test_abnormal_sequences),
        "train_length_stats": _compute_length_stats(train_sequences),
        "test_length_stats": _compute_length_stats(test_sequences),
        "test_normal_length_stats": _compute_length_stats(test_normal_sequences),
        "test_anomaly_length_stats": _compute_length_stats(test_abnormal_sequences),
        "total_events_test": _compute_length_stats(test_sequences)["total_events"],
    }

    return DatasetBundle(
        name=dataset_name,
        train=DatasetSplit(sequences=train_sequences, labels=["normal"] * len(train_sequences)),
        test=DatasetSplit(sequences=test_sequences, labels=test_labels),
        test_by_type=test_by_type or None,
        stats=stats,
    )
