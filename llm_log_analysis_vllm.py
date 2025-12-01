from vllm import LLM, SamplingParams
import csv
import json
from typing import List, Dict, Any, Optional, Set
import argparse
import os
import hashlib
import re

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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


def compute_prompt_hash(prompt: str) -> str:
    """Compute a hash of the prompt for deduplication."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_existing_results(output_path: str) -> tuple[List[Dict[str, Any]], Set[str]]:
    """
    Load existing results from the output file and build a set of processed prompt hashes.
    
    Returns:
        Tuple of (existing_results_list, set_of_processed_prompt_hashes)
    """
    if not os.path.exists(output_path):
        return [], set()
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        
        # Build set of prompt hashes from existing results
        processed_hashes = set()
        for result in existing_results:
            log_text = result.get("log_text", "")
            prompt = build_prompt(log_text)
            prompt_hash = compute_prompt_hash(prompt)
            processed_hashes.add(prompt_hash)
        
        print(f"Loaded {len(existing_results)} existing results, skipping already processed prompts.")
        return existing_results, processed_hashes
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load existing results from {output_path}: {e}")
        return [], set()


def log_batch_to_wandb(
    batch_results: List[Dict[str, Any]],
    batch_num: int,
    total_processed: int,
) -> None:
    """Log batch results to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    # Count errors in this batch
    errors_in_batch = sum(
        1 for r in batch_results 
        if r.get("llm_result", {}).get("has_error") is True
    )
    parse_errors = sum(
        1 for r in batch_results
        if r.get("llm_result", {}).get("error_type") == "parse_error"
    )
    
    # Log metrics
    wandb.log({
        "batch_num": batch_num,
        "total_processed": total_processed,
        "batch_size": len(batch_results),
        "errors_detected": errors_in_batch,
        "parse_errors": parse_errors,
    })


### 2. CSV → batched vLLM inference → JSON output

def process_csv_with_vllm(
    csv_path: str,
    model: str,
    has_header: bool = True,
    max_rows: Optional[int] = None,
    batch_size: int = 64,
    output_path: str = "output.json",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_log_frequency: int = 1,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process CSV with vLLM for log analysis.
    
    Args:
        csv_path: Path to input CSV file
        model: Model name to use
        has_header: Whether CSV has a header row
        max_rows: Maximum number of rows to process
        batch_size: Number of prompts per batch
        output_path: Path to save results JSON
        wandb_project: W&B project name (if None, wandb logging is disabled)
        wandb_run_name: W&B run name (optional)
        wandb_log_frequency: Log to wandb every N batches (default: 1 = every batch)
        resume: If True, load existing results and skip already processed prompts
    
    Returns:
        List of result dictionaries
    """
    # Initialize wandb if project is specified
    if wandb_project and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "csv_path": csv_path,
                "model": model,
                "batch_size": batch_size,
                "max_rows": max_rows,
            },
            resume="allow",  # Allow resuming if run was interrupted
        )
        print(f"W&B logging enabled for project: {wandb_project}")
    elif wandb_project and not WANDB_AVAILABLE:
        print("Warning: wandb_project specified but wandb is not installed. Install with: pip install wandb")
    
    # Load existing results if resume is enabled
    processed_hashes: Set[str] = set()
    if resume:
        results, processed_hashes = load_existing_results(output_path)
    else:
        results: List[Dict[str, Any]] = []
    
    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []  # store row_index, row, log_text
    batch_num = 0
    skipped_count = 0
    
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
        nonlocal results, batch_prompts, batch_meta, batch_num

        if not batch_prompts:
            return

        batch_num += 1
        batch_start_idx = len(results)
        
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
        
        # Log to wandb after each batch (or per wandb_log_frequency)
        if batch_num % wandb_log_frequency == 0:
            batch_results = results[batch_start_idx:]
            log_batch_to_wandb(batch_results, batch_num, len(results))
        
        # Save checkpoint after each batch to minimize data loss on GPU suspension
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        
        print(f"Batch {batch_num}: processed {len(batch_prompts)} prompts, total: {len(results)}")

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
            
            # Check if this prompt was already processed (skip if exact match)
            prompt_hash = compute_prompt_hash(prompt)
            if prompt_hash in processed_hashes:
                skipped_count += 1
                continue
            
            # Mark as processed to avoid duplicates in current run
            processed_hashes.add(prompt_hash)

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

        # Flush any remaining prompts
        run_batch()

    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed prompts.")
    
    # Final save
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    
    # Log final summary to wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        total_errors = sum(
            1 for r in results 
            if r.get("llm_result", {}).get("has_error") is True
        )
        wandb.log({
            "final_total_processed": len(results),
            "final_total_errors": total_errors,
            "skipped_prompts": skipped_count,
        })
        wandb.finish()

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of prompts per batch (default: 64).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.json",
        help="Path to save results JSON (default: output.json).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name. If specified, enables wandb logging.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional).",
    )
    parser.add_argument(
        "--wandb-log-frequency",
        type=int,
        default=1,
        help="Log to wandb every N batches (default: 1 = every batch).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume functionality. By default, existing results are loaded and processed prompts are skipped.",
    )
    
    args = parser.parse_args()

    process_csv_with_vllm(
        csv_path=args.csv_path,
        model=args.model,
        has_header=args.has_header,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        output_path=args.output_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_log_frequency=args.wandb_log_frequency,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
