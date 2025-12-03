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
from llm_interface.openrouter_client import INPUT_RATE_PER_TOKEN, OUTPUT_RATE_PER_TOKEN  # noqa: E402

OVERHEAD_TOKENS = 100
COMPLETION_TOKENS_PER_CALL = 64


def default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "anomaly-detection-log-datasets"


def estimate_dataset_cost(dataset: str, base_dir: Path) -> Dict[str, float]:
    bundle = build_dataset(dataset, str(base_dir))
    n_test = len(bundle.test.sequences)
    total_prompt_tokens = sum(len(seq) + OVERHEAD_TOKENS for seq in bundle.test.sequences)
    total_completion_tokens = COMPLETION_TOKENS_PER_CALL * n_test

    cost_input = total_prompt_tokens * INPUT_RATE_PER_TOKEN
    cost_output = total_completion_tokens * OUTPUT_RATE_PER_TOKEN

    return {
        "dataset": dataset,
        "n_test": n_test,
        "n_test_normal": bundle.stats["n_test_normal"],
        "n_test_anomaly": bundle.stats["n_test_anomaly"],
        "approx_prompt_tokens": total_prompt_tokens,
        "approx_completion_tokens": total_completion_tokens,
        "estimated_cost_input": cost_input,
        "estimated_cost_output": cost_output,
        "estimated_total_cost": cost_input + cost_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate OpenRouter cost before running evaluations.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG.keys()),
        help=f"Datasets to include. Options: {list(DATASET_CONFIG.keys())}",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=default_base_dir(),
        help="Path to the anomaly-detection-log-datasets repository.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/metrics/cost_estimate.json"),
        help="Where to write the JSON cost estimate.",
    )
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=Path("results/metrics/cost_estimate.txt"),
        help="Where to write the human-readable summary.",
    )
    args = parser.parse_args()

    estimates: List[Dict[str, float]] = []
    for dataset in args.datasets:
        estimates.append(estimate_dataset_cost(dataset, args.base_dir))

    total_cost = sum(item["estimated_total_cost"] for item in estimates)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump({"datasets": estimates, "total_estimated_cost": total_cost}, handle, indent=2)

    lines = [
        "===== Cost estimate (DeepSeek V3.2 on OpenRouter) =====",
        "Pricing: $0.28 / 1M input tokens, $0.40 / 1M output tokens.",
    ]
    for item in estimates:
        lines.append(f"\nDataset: {item['dataset']}")
        lines.append(f"  n_test = {item['n_test']} (normal {item['n_test_normal']}, anomaly {item['n_test_anomaly']})")
        lines.append(f"  approx_prompt_tokens = {item['approx_prompt_tokens']:,}")
        lines.append(f"  approx_completion_tokens = {item['approx_completion_tokens']:,}")
        lines.append(f"  estimated_cost_input = ${item['estimated_cost_input']:.4f}")
        lines.append(f"  estimated_cost_output = ${item['estimated_cost_output']:.4f}")
        lines.append(f"  estimated_total_cost = ${item['estimated_total_cost']:.4f}")
    lines.append(f"\nGlobal total (all planned runs): ${total_cost:.4f}")
    lines.append("Budget cap: $10.00")
    lines.append("=======================================================")
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_txt.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote estimates to {args.output_json} and {args.output_txt}")


if __name__ == "__main__":
    main()
