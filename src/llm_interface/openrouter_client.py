from __future__ import annotations

import csv
import datetime
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from openrouter import OpenRouter

INPUT_RATE_PER_TOKEN = 0.28 / 1_000_000  # USD per input token
OUTPUT_RATE_PER_TOKEN = 0.40 / 1_000_000  # USD per output token


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            elif isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
        return " ".join(parts)
    return str(content)


def extract_json_content(raw_content: str) -> Dict[str, Any]:
    text = raw_content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek/deepseek-v3.2",
        referer: str = "http://localhost",
        title: str = "llm-log-anomaly",
        log_dir: Path | str = "results/logs",
        timeout_ms: int = 180_000,
    ) -> None:
        self.model = model
        self.client = OpenRouter(api_key=api_key, timeout_ms=timeout_ms)
        self.http_headers: Dict[str, str] = {}
        if referer:
            self.http_headers["HTTP-Referer"] = referer
        if title:
            self.http_headers["X-Title"] = title
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log_path = self.log_dir / f"api_usage_{datetime.date.today().isoformat()}.csv"
        self._log_lock = threading.Lock()
        if not self.usage_log_path.exists():
            with self.usage_log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "dataset",
                        "sequence_index",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cost_input_usd",
                        "cost_output_usd",
                    ]
                )

    def send_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format_json: bool = True,
        temperature: float = 0.0,
        max_completion_tokens: int = 64,
        dataset: Optional[str] = None,
        sequence_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        response = self.client.chat.send(
            messages=messages,
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"} if response_format_json else None,
            max_completion_tokens=max_completion_tokens,
            stream=False,
            http_headers=self.http_headers or None,
        )

        usage = response.usage or {}
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        is_free_model = ":free" in (self.model or "")
        cost_input = 0.0 if is_free_model else prompt_tokens * INPUT_RATE_PER_TOKEN
        cost_output = 0.0 if is_free_model else completion_tokens * OUTPUT_RATE_PER_TOKEN

        if dataset is not None and sequence_index is not None:
            with self._log_lock:
                with self.usage_log_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            dataset,
                            sequence_index,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            f"{cost_input:.6f}",
                            f"{cost_output:.6f}",
                        ]
                    )

        choice = response.choices[0]
        raw_content = _content_to_text(choice.message.content)

        return {
            "response": response,
            "raw_content": raw_content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_input_usd": cost_input,
                "cost_output_usd": cost_output,
            },
        }
