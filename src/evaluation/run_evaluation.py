from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from data_loading.sequence_loader import DATASET_CONFIG, build_dataset  # noqa: E402
from llm_interface.classifier import classify_sequence  # noqa: E402
from llm_interface.openrouter_client import OpenRouterClient  # noqa: E402

BUDGET_STOP_USD = 9.0


def evaluate_dataset(
    dataset: str,
    base_dir: Path,
    client: OpenRouterClient,
    predictions_path: Path,
    metrics_path: Path,
    skip_on_error: bool = False,
    max_workers: int = 10,
    max_completion_tokens: int = 64,
) -> Dict[str, float]:
    bundle = build_dataset(dataset, str(base_dir))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    y_true: List[int] = []
    y_pred: List[int] = []
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    skipped = 0

    def _infer(idx_seq_label: Tuple[int, List[int], str]):
        idx, seq, true_label = idx_seq_label
        return idx, true_label, classify_sequence(
            client,
            dataset,
            seq,
            sequence_index=idx,
            skip_on_error=skip_on_error,
            max_completion_tokens=max_completion_tokens,
        )

    tasks = [(idx, seq, label) for idx, (seq, label) in enumerate(zip(bundle.test.sequences, bundle.test.labels))]

    results: Dict[int, Tuple[str, Dict[str, float], str, str]] = {}

    # Pre-load valid existing predictions to skip re-run
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    idx_existing = int(row["idx"])
                except Exception:
                    continue
                error = row.get("error", "")
                pred_label = row.get("pred_label", "")
                anomaly_pred = row.get("anomaly_pred", "")
                true_label_existing = row.get("true_label", "")
                if pred_label and not error:
                    usage = {
                        "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(row.get("completion_tokens", 0) or 0),
                        "total_tokens": int(row.get("total_tokens", 0) or 0),
                        "cost_input_usd": float(row.get("cost_input_usd", 0.0) or 0.0),
                        "cost_output_usd": float(row.get("cost_output_usd", 0.0) or 0.0),
                    }
                    result_obj = {
                        "label": pred_label,
                        "anomaly": str(anomaly_pred).lower() == "true",
                        "usage": usage,
                        "raw_content": row.get("raw_content", ""),
                    }
                    results[idx_existing] = (true_label_existing, result_obj, row.get("raw_content", ""), "")

    tasks = [(idx, seq, label) for idx, (seq, label) in enumerate(zip(bundle.test.sequences, bundle.test.labels)) if idx not in results]

    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_infer, t): t for t in tasks}
            progress = tqdm(total=len(futures), desc=f"{dataset} eval", leave=True)
            valid_count = 0
            error_samples = []
            for fut in as_completed(futures):
                idx_submit, _, true_label_submit = futures[fut]
                try:
                    idx_res, true_label_res, result = fut.result(timeout=180)
                except Exception as exc:  # noqa: BLE001
                    results[idx_submit] = (true_label_submit, {}, f"error: {exc}", "error")
                    if len(error_samples) < 10:
                        error_samples.append(f"idx {idx_submit}: {exc}")
                        print(f"[{dataset}] error {len(error_samples)}/{len(futures)}: idx {idx_submit}: {exc}")
                    progress.update(1)
                    pct = (valid_count / progress.n * 100) if progress.n else 0.0
                    progress.set_postfix(valid_pct=f"{pct:.1f}")
                    continue
                results[idx_res] = (true_label_res, result, result.get("raw_content", ""), result.get("error", ""))
                if "error" not in result:
                    valid_count += 1
                elif len(error_samples) < 10:
                    msg = result.get("error") or result.get("raw_content")
                    error_samples.append(f"idx {idx_res}: {msg}")
                    print(f"[{dataset}] error {len(error_samples)}/{len(futures)}: idx {idx_res}: {msg}")
                progress.update(1)
                pct = (valid_count / progress.n * 100) if progress.n else 0.0
                progress.set_postfix(valid_pct=f"{pct:.1f}")
            progress.close()

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "idx",
                "true_label",
                "pred_label",
                "anomaly_pred",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost_input_usd",
                "cost_output_usd",
                "raw_content",
                "error",
            ]
        )

        for idx in sorted(results):
            true_label, result, raw_content, error = results[idx]
            if error:
                skipped += 1
                writer.writerow(
                    [
                        idx,
                        true_label,
                        "",
                        "",
                        0,
                        0,
                        0,
                        "0.000000",
                        "0.000000",
                        raw_content,
                        error,
                    ]
                )
                continue

            pred_label = result["label"]
            anomaly_pred = bool(result["anomaly"])
            usage = result.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            cost_in = float(usage.get("cost_input_usd", 0.0) or 0.0)
            cost_out = float(usage.get("cost_output_usd", 0.0) or 0.0)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_cost += cost_in + cost_out

            y_true.append(1 if true_label == "anomaly" else 0)
            y_pred.append(1 if anomaly_pred else 0)

            writer.writerow(
                [
                    idx,
                    true_label,
                    pred_label,
                    anomaly_pred,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    f"{cost_in:.6f}",
                    f"{cost_out:.6f}",
                    raw_content,
                    "",
                ]
            )

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0

    tp = int(np.logical_and(np.array(y_true) == 1, np.array(y_pred) == 1).sum())
    tn = int(np.logical_and(np.array(y_true) == 0, np.array(y_pred) == 0).sum())
    fp = int(np.logical_and(np.array(y_true) == 0, np.array(y_pred) == 1).sum())
    fn = int(np.logical_and(np.array(y_true) == 1, np.array(y_pred) == 0).sum())

    metrics = {
        "dataset": dataset,
        "n_test": len(bundle.test.sequences),
        "n_test_normal": bundle.stats["n_test_normal"],
        "n_test_anomaly": bundle.stats["n_test_anomaly"],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_cost_usd": total_cost,
        "model": client.model,
        "notes": "zero-shot, per-sequence classification based on event ID sequences",
        "skipped": skipped,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-based anomaly detection evaluation.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG.keys()),
        help=f"Datasets to run. Options: {list(DATASET_CONFIG.keys())}",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "anomaly-detection-log-datasets",
        help="Path to the anomaly-detection-log-datasets repository.",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("results/predictions"),
        help="Directory to store per-sequence predictions.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to store metrics JSON files.",
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v3.2",
        help="OpenRouter model to use.",
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip sequences with API/parse errors instead of labeling them anomaly.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Max concurrent requests.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=64,
        help="Max completion tokens per request.",
    )
    args = parser.parse_args()

    def sanitize_model_tag(model: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_") or "model"

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

    client = OpenRouterClient(api_key=api_key, model=args.model)
    model_tag = sanitize_model_tag(args.model)
    pred_root = args.pred_dir / model_tag
    metrics_root = args.metrics_dir / model_tag
    pred_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    for dataset in args.datasets:
        pred_path = pred_root / f"{dataset}_predictions.csv"
        metrics_path = metrics_root / f"{dataset}_metrics.json"
        metrics = evaluate_dataset(
            dataset,
            args.base_dir,
            client,
            pred_path,
            metrics_path,
            skip_on_error=args.skip_errors,
            max_workers=args.max_workers,
            max_completion_tokens=args.max_completion_tokens,
        )
        all_metrics.append(metrics)
        print(f"{dataset}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    summary_path = metrics_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
