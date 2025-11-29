#!/usr/bin/env python3
import os
import csv
import json
import argparse
import requests
from typing import Optional, Dict, Any
from transformers import pipeline
import torch

# -----------------------------
# LLM call helper
# -----------------------------
def call_llm_on_log(
    log_text: str,
    api_key: str,
    api_url: str,
    model: str,
) -> Dict[str, Any]:
    """
    Send a single log line to the LLM and ask it to detect errors.

    Returns a Python dict parsed from the model's JSON response.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # System + user prompt. We force JSON output.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a log analysis assistant. "
                "Given a single log line, you must detect whether it indicates "
                "an error, failure, or abnormal behavior. "
                "If there is no error, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze the following log entry and detect ANY errors or abnormal behavior.\n\n"
                "Log entry:\n"
                "```log\n"
                f"{log_text}\n"
                "```\n\n"
                "Respond ONLY with a JSON object with the following keys:\n"
                "  - \"has_error\": true or false\n"
                "  - \"error_type\": short string label (e.g., \"network\", \"configuration\", \"none\", etc.)\n"
                "  - \"severity\": one of [\"info\", \"warning\", \"error\", \"critical\"]\n"
                "  - \"explanation\": short natural language explanation\n"
            ),
        },
    ]

    payload = {
        "model": model,
        "messages": messages,
        # Depending on your provider, this may differ (e.g. `temperature`, `max_tokens`, etc.)
        "temperature": 0.1,
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # Model should return JSON, but we guard against slight deviations
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "has_error": None,
            "error_type": "parse_error",
            "severity": "info",
            "explanation": f"Failed to parse model output as JSON: {content}",
        }

    return data

def gpt_oss20b_test(log_text: str, pipe):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a log analysis assistant. "
                "Given a single log line, you must detect whether it indicates "
                "an error or failure with the system you may ignore any formatting errors in the logs"
                "If there is no error, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze the following log entry and detect ANY errors or abnormal behavior, keep the answers short and simple.\n\n"
                "Log entry:\n"
                "```log\n"
                f"{log_text}\n"
                "```\n\n"
                "Respond ONLY with a JSON object with the following keys:\n"
                "  - \"has_error\": true or false\n"
                "  - \"error_type\": short string label (e.g., \"network\", \"configuration\", \"none\", etc.)\n"
                "  - \"explanation\": short natural language explanation\n"
            ),
        },
    ]

    outputs = pipe(
        messages,
        max_new_tokens=1024,
    )
    return outputs[0]["generated_text"][-1]


# -----------------------------
# Main CSV processing
# -----------------------------
def process_csv(
    csv_path: str,
    model: str,
    log_col: Optional[int] = None,
    has_header: bool = False,
    max_rows: Optional[int] = None,
) -> None:
    """
    Loop through CSV rows and send each log line to the LLM.

    Prints one JSON object per line with:
      {
        "row_index": ...,
        "original_row": [...],
        "log_text": "...",
        "llm_result": {...}
      }
    """
    if "gpt-oss-20b" == model:
        model_id = "openai/gpt-oss-20b"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    elif model == "gpt-oss-120b":
        model_id = "openai/gpt-oss-120b"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    elif model == "deepseek-3.1-7b-instruct":
        model_id = "deepseek-ai/DeepSeek-V3-7B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
            
    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        if has_header:
            header = next(reader, None)
            # print(f"# Skipping header: {header}", flush=True)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            # Use entire row EXCEPT first two columns
            if len(row) > 4:
                log_text = ",".join(row[5:])
            else:
                log_text = ",".join(row)

            llm_result = gpt_oss20b_test(log_text, pipe)

            output = {
                "row_index": i,
                "original_row": row,
                "log_text": log_text,
                "llm_result": llm_result,
            }

            results.append(output)

            # Every 50 iterations, save all accumulated results
            if i % 50 == 0:
                with open("output.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)



# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Loop through a CSV and feed each row (or a column) to an LLM for log error detection."
    )
    parser.add_argument("--csv_path", help="Path to input CSV file.")
    parser.add_argument(
        "--log-col",
        type=int,
        default=None,
        help="0-based column index to treat as log text. If omitted, the whole row is used.",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Set this flag if the first row is a header and should be skipped.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of rows to process.",
    )
    parser.add_argument(
        "--model",
        help="Model name to use (e.g., deepseek-v3.1-instruct).",
    )
    
    args = parser.parse_args()

    process_csv(
        csv_path=args.csv_path,
        model=args.model,
        log_col=args.log_col,
        has_header=args.has_header,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
