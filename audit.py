from __future__ import annotations

import json
import os
from typing import Any


def _log_path() -> str:
    return os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")


def append_audit_log(entry: dict[str, Any]) -> None:
    log_path = _log_path()
    parent_dir = os.path.dirname(log_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(entry, ensure_ascii=False)}\n")


def get_audit_log_entries(limit: int = 50) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    try:
        with open(_log_path(), "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []

    entries: list[dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)

    return entries


def find_first_event(content_id: str, event: str) -> dict[str, Any] | None:
    try:
        with open(_log_path(), "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("content_id") == content_id and payload.get("event") == event:
                    return payload
    except FileNotFoundError:
        return None

    return None


def update_first_event(content_id: str, event: str, updates: dict[str, Any]) -> bool:
    log_path = _log_path()
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return False

    updated = False
    parsed: list[dict[str, Any] | None] = []

    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parsed.append(None)
            continue
        if not updated and isinstance(payload, dict) and payload.get("content_id") == content_id and payload.get("event") == event:
            payload.update(updates)
            updated = True
        parsed.append(payload if isinstance(payload, dict) else None)

    if not updated:
        return False

    with open(log_path, "w", encoding="utf-8") as handle:
        for original_line, payload in zip(lines, parsed, strict=False):
            if payload is None:
                handle.write(original_line)
            else:
                handle.write(f"{json.dumps(payload, ensure_ascii=False)}\n")

    return True
