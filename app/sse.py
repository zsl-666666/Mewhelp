import json
from typing import Any


def format_sse(data: dict[str, Any], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    payload = json.dumps(data, ensure_ascii=False)
    return f"{prefix}data: {payload}\n\n"
