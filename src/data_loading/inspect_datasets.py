from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from data_loading.sequence_loader import DATASET_CONFIG, build_dataset  # noqa: E402


def default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "anomaly-detection-log-datasets"


def inspect(datasets: List[str], base_dir: Path) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for dataset in datasets:
        bundle = build_dataset(dataset, str(base_dir))
        results[dataset] = bundle.stats
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset stats from the AIT splits.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG.keys()),
        help=f"Datasets to inspect. Options: {list(DATASET_CONFIG.keys())}",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=default_base_dir(),
        help="Path to the anomaly-detection-log-datasets repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON stats (e.g., results/metrics/dataset_stats.json).",
    )
    args = parser.parse_args()

    stats = inspect(args.datasets, args.base_dir)
    print("=== Dataset stats ===")
    for name, info in stats.items():
        print(f"- {name}")
        print(f"  n_train: {info['n_train']}")
        print(f"  n_test: {info['n_test']} (normal {info['n_test_normal']}, anomaly {info['n_test_anomaly']})")
        test_stats = info["test_length_stats"]
        print(
            f"  test seq lens: avg {test_stats['avg_len']:.2f}, "
            f"max {test_stats['max_len']}, total events {test_stats['total_events']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"\nWrote stats to {args.output}")


if __name__ == "__main__":
    main()
