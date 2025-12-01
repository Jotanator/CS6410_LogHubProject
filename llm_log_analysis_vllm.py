from vllm import LLM, SamplingParams
import csv
import json
from typing import List, Dict, Any
import argparse

import json
import re

def extract_json_from_text(text: str) -> dict:
    # Strip some common junk token that shows up in some chat templates
    text = text.replace("assistantfinal", "").strip()

    # First try: last {...} block in the string
    start = text.rfind("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: regex over all {...} blocks, try from the end backwards
    matches = re.findall(r"\{.*?\}", text, flags=re.S)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # If everything fails, raise or return a default
    raise ValueError(f"Could not extract valid JSON from LLM output: {text[:200]!r}")


# 2) Sampling params
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 3) Build a *single-string* prompt for vLLM.generate
PROMPT_TEMPLATE = """
You are a log analysis assistant.
Given a single log line, you must detect whether it indicates an error or failure
with the system. You may ignore any formatting errors in the logs.

Analyze the following log entry and detect ANY errors or abnormal behavior,
keeping the answer short and simple.

Log entry:
```log
{log_text}
Respond ONLY with a valid JSON object. Do not repeat the prompt, do not add commentary.
Use exactly the following keys:
"has_error": true or false
"error_type": short string label (e.g., "network", "configuration", "none", etc.)
"explanation": short natural language explanation

Example output (error case):
{{
"has_error": true,
"error_type": "network",
"explanation": "Connection timed out while reaching the database server."
}}
Example output (no error case):
{{
"has_error": false,
"error_type": "none",
"explanation": "No abnormal behavior detected in the log."
}}
"""

def build_prompt(log_text: str) -> str:
    return PROMPT_TEMPLATE.format(log_text=log_text)


### 2. CSV → batched vLLM inference → JSON output

def process_csv_with_vllm(
    csv_path: str,
    model: str,
    has_header: bool = True,
    max_rows: int | None = None,
    batch_size: int = 64,
    output_path: str = "output.json",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []  # store row_index, row, log_text
    count_batch = 0
    if "gpt-oss-20b" == model:
        llm = LLM(
        model="openai/gpt-oss-20b",   # works if it's a causal LM on HF
        dtype="bfloat16",             # or float16
        )
    elif model == "qwen":
        llm = LLM(
        model="Qwen/Qwen2.5-4B-Instruct",   # ✔ correct HF ID (2507 = July 2025 update)
        dtype="bfloat16",
        )

    def run_batch():
        """Run vLLM on the current batch and extend results."""
        nonlocal results, batch_prompts, batch_meta

        if not batch_prompts:
            return

        # vLLM batched generation
        outputs = llm.generate(batch_prompts, sampling_params)

        for req_out, meta in zip(outputs, batch_meta):
            # Take first completion
            text = req_out.outputs[0].text.strip()

            # Try to parse JSON; if it fails, store raw text
            try:
                parsed = extract_json_from_text(text)
            except Exception as e:
                parsed = {
                    "has_error": None,
                    "error_type": "parse_error",
                    "explanation": "Model did not return valid JSON.",
                    "raw_output": text,
                }

            output_record = {
                "row_index": meta["row_index"],
                "original_row": meta["row"],
                "log_text": meta["log_text"],
                "llm_result": parsed,
            }
            results.append(output_record)

        # Clear batch buffers
        batch_prompts = []
        batch_meta = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        if has_header:
            header = next(reader, None)
            # print(f"# Skipping header: {header}", flush=True)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            # Use entire row EXCEPT first 5 columns (like your code)
            if len(row) > 4:
                log_text = ",".join(row[5:])
            else:
                log_text = ",".join(row)

            prompt = build_prompt(log_text)

            batch_prompts.append(prompt)
            batch_meta.append(
                {
                    "row_index": i,
                    "row": row,
                    "log_text": log_text,
                }
            )

            # When batch is full, run vLLM
            if len(batch_prompts) >= batch_size:
                run_batch()
                count_batch += batch_size

            # Periodic checkpoint: every 50 rows, dump accumulated results
            if i % 128 == 0 and i > 0:
                with open(output_path, "w", encoding="utf-8") as fp:
                    json.dump(results, fp, ensure_ascii=False, indent=2)
                    print(count_batch)

        # Flush any remaining prompts
        run_batch()

    # Final save
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    return results

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Loop through a CSV and feed each row (or a column) to an LLM for log error detection."
    )
    parser.add_argument("--csv_path", help="Path to input CSV file.")
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

    process_csv_with_vllm(
        csv_path=args.csv_path,
        model=args.model,
        has_header=args.has_header,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
