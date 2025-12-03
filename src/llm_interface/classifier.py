from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from llm_interface.openrouter_client import OpenRouterClient, extract_json_content


BASE_SYSTEM_PROMPT = (
    "You are a system log anomaly detection assistant. "
    "You receive log event sequences extracted from the {dataset_name} system. "
    "Each sequence corresponds to one execution session and is represented as a list of integer event type IDs. "
    "Classify whether the sequence is NORMAL or ANOMALOUS. "
    'Return your answer strictly as a JSON object with keys: "label" (\"normal\" or \"anomaly\") '
    'and "anomaly" (boolean). Do not include explanations or any other keys.'
)

HADOOP_EXTRA = (
    "In this dataset, anomalous sequences correspond to faults such as disk full, machine down, "
    "or network disconnected. You do not need to predict the specific fault type, only whether the sequence is anomalous."
)

OPENSTACK_EXTRA = (
    "Anomalies indicate failures captured in OpenStack control and compute node logs. "
    "Label as anomaly if the sequence reflects abnormal behavior; otherwise label normal."
)

ADFA_EXTRA = (
    "Sequences come from host-based intrusion detection traces. Multiple attack types exist, "
    "but your task remains binary: anomaly for any attack, normal otherwise."
)


def build_messages(dataset_name: str, sequence: List[int]) -> List[Dict[str, str]]:
    dataset_name = dataset_name.lower()
    extras = {
        "hadoop": HADOOP_EXTRA,
        "openstack": OPENSTACK_EXTRA,
        "adfa": ADFA_EXTRA,
    }.get(dataset_name, "")

    system_content = BASE_SYSTEM_PROMPT.format(dataset_name=dataset_name)
    if extras:
        system_content = f"{system_content} {extras}"

    sequence_text = " ".join(str(token) for token in sequence)
    user_content = (
        "Here is a single log event sequence.\n\n"
        f"Event sequence (space-separated event type IDs):\n{sequence_text}\n\n"
        "Return the JSON object now."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def normalize_prediction(parsed: Dict[str, Any]) -> Tuple[str, bool]:
    label_raw = str(parsed.get("label", "")).strip().lower()
    anomaly_flag = parsed.get("anomaly")
    if anomaly_flag is None:
        anomaly_flag = label_raw == "anomaly"
    anomaly_bool = bool(anomaly_flag)
    label = "anomaly" if anomaly_bool else "normal"
    return label, anomaly_bool


def classify_sequence(
    client: OpenRouterClient,
    dataset_name: str,
    sequence: List[int],
    *,
    sequence_index: int,
    temperature: float = 0.0,
    skip_on_error: bool = False,
    max_completion_tokens: int = 64,
) -> Dict[str, Any]:
    messages = build_messages(dataset_name, sequence)
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = client.send_chat(
                messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                dataset=dataset_name,
                sequence_index=sequence_index,
            )
            parsed = extract_json_content(response["raw_content"])
            label, anomaly = normalize_prediction(parsed)
            return {
                "label": label,
                "anomaly": anomaly,
                "raw_content": response["raw_content"],
                "usage": response["usage"],
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if skip_on_error:
                return {
                    "error": str(exc),
                    "raw_content": f"error: {exc}",
                    "usage": {},
                }
            # tighten instructions and retry once
            messages[-1]["content"] = (
                messages[-1]["content"]
                + "\nYour last response was invalid. Return ONLY a JSON object with keys \"label\" and \"anomaly\"."
            )

    # If still failing, mark as anomaly to be conservative
    if skip_on_error:
        return {
            "error": f"parse_error: {last_error}",
            "raw_content": f"parse_error: {last_error}",
            "usage": {},
        }
    return {
        "label": "anomaly",
        "anomaly": True,
        "raw_content": f"parse_error: {last_error}",
        "usage": {},
    }
