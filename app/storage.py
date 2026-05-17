import json
import os
from pathlib import Path


def _get_status_file_path() -> Path:
    configured_path = os.getenv("DETECTION_STATUS_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detection_statuses.json"


def _get_comment_file_path() -> Path:
    configured_path = os.getenv("DETECTION_COMMENT_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detection_comments.json"


def _read_string_mapping(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    if not isinstance(raw_data, dict):
        return {}

    return {str(key): str(value) for key, value in raw_data.items()}


def _write_string_mapping(file_path: Path, values: dict[str, str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(values, file_handle, indent=2, ensure_ascii=True)


def read_detection_statuses() -> dict[str, str]:
    return _read_string_mapping(_get_status_file_path())


def write_detection_statuses(statuses: dict[str, str]) -> None:
    _write_string_mapping(_get_status_file_path(), statuses)


def upsert_detection_status(detection_id: str, status: str) -> dict[str, str]:
    statuses = read_detection_statuses()
    statuses[detection_id] = status
    write_detection_statuses(statuses)
    return statuses


def read_detection_comments() -> dict[str, str]:
    return _read_string_mapping(_get_comment_file_path())


def write_detection_comments(comments: dict[str, str]) -> None:
    _write_string_mapping(_get_comment_file_path(), comments)


def upsert_detection_comment(detection_id: str, comment: str) -> dict[str, str]:
    comments = read_detection_comments()
    comments[detection_id] = comment
    write_detection_comments(comments)
    return comments
