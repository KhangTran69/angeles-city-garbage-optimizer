"""Persistence helpers for optimization experiment metadata."""
import json
import os
from typing import Any, Dict, List, Optional

from src.config import EXPERIMENTS_PATH, MAX_HISTORY_SIZE


def load_experiment_history(path: str = EXPERIMENTS_PATH) -> List[Dict[str, Any]]:
    """Load previously persisted experiment records."""
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []

    if not isinstance(raw_data, list):
        return []
    return [item for item in raw_data if isinstance(item, dict)]


def append_experiment_record(
    record: Dict[str, Any],
    path: str = EXPERIMENTS_PATH,
    max_items: int = MAX_HISTORY_SIZE * 5,
) -> List[Dict[str, Any]]:
    """Append an experiment record and persist capped history to disk."""
    history = load_experiment_history(path)
    history.append(record)
    trimmed_history = history[-max_items:]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(trimmed_history, file, indent=2, ensure_ascii=False)

    return trimmed_history


def latest_experiment(path: str = EXPERIMENTS_PATH) -> Optional[Dict[str, Any]]:
    """Return the most recent experiment record, if available."""
    history = load_experiment_history(path)
    return history[-1] if history else None
