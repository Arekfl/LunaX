import json
import os
from pathlib import Path


def _get_status_file_path() -> Path:
    configured_path = os.getenv("DETECTION_STATUS_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detection_statuses.json"


def read_detection_statuses() -> dict[str, str]:
    status_file = _get_status_file_path()
    if not status_file.exists():
        return {}

    with status_file.open("r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    if not isinstance(raw_data, dict):
        return {}

    return {
        str(detection_id): str(status)
        for detection_id, status in raw_data.items()
    }


def write_detection_statuses(statuses: dict[str, str]) -> None:
    status_file = _get_status_file_path()
    status_file.parent.mkdir(parents=True, exist_ok=True)

    with status_file.open("w", encoding="utf-8") as file_handle:
        json.dump(statuses, file_handle, indent=2, ensure_ascii=True)


def upsert_detection_status(detection_id: str, status: str) -> dict[str, str]:
    statuses = read_detection_statuses()
    statuses[detection_id] = status
    write_detection_statuses(statuses)
    return statuses
