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


def _get_tags_file_path() -> Path:
    configured_path = os.getenv("DETECTION_TAG_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detection_tags.json"


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


def _read_tags_mapping(file_path: Path) -> dict[str, list[str]]:
    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    if not isinstance(raw_data, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, value in raw_data.items():
        if not isinstance(value, list):
            continue

        normalized[str(key)] = [str(tag) for tag in value if str(tag).strip()]

    return normalized


def _write_tags_mapping(file_path: Path, values: dict[str, list[str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_values: dict[str, list[str]] = {}
    for key, value in values.items():
        if not isinstance(value, list):
            continue

        normalized_values[str(key)] = [str(tag) for tag in value if str(tag).strip()]

    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(normalized_values, file_handle, indent=2, ensure_ascii=True)


def read_detection_statuses() -> dict[str, str]:
    return _read_string_mapping(_get_status_file_path())


def write_detection_statuses(statuses: dict[str, str]) -> None:
    _write_string_mapping(_get_status_file_path(), statuses)


def upsert_detection_status(detection_id: str, status: str) -> dict[str, str]:
    statuses = read_detection_statuses()
    statuses[detection_id] = status
    write_detection_statuses(statuses)
    return statuses


def delete_detection_status(detection_id: str) -> dict[str, str]:
    statuses = read_detection_statuses()
    statuses.pop(detection_id, None)
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


def delete_detection_comment(detection_id: str) -> dict[str, str]:
    comments = read_detection_comments()
    comments.pop(detection_id, None)
    write_detection_comments(comments)
    return comments


def read_detection_tags() -> dict[str, list[str]]:
    return _read_tags_mapping(_get_tags_file_path())


def write_detection_tags(tags: dict[str, list[str]]) -> None:
    _write_tags_mapping(_get_tags_file_path(), tags)


def upsert_detection_tags(detection_id: str, tags: list[str]) -> dict[str, list[str]]:
    normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    unique_tags = list(dict.fromkeys(normalized_tags))

    stored_tags = read_detection_tags()
    stored_tags[detection_id] = unique_tags
    write_detection_tags(stored_tags)
    return stored_tags


def upsert_detection_tags_bulk(detection_ids: list[str], tag: str) -> dict[str, list[str]]:
    normalized_tag = str(tag).strip()
    if not normalized_tag:
        return read_detection_tags()

    unique_detection_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_detection_id in detection_ids:
        detection_id = str(raw_detection_id).strip()
        if not detection_id or detection_id in seen_ids:
            continue

        seen_ids.add(detection_id)
        unique_detection_ids.append(detection_id)

    if not unique_detection_ids:
        return read_detection_tags()

    stored_tags = read_detection_tags()
    for detection_id in unique_detection_ids:
        current_tags = [
            str(item).strip()
            for item in stored_tags.get(detection_id, [])
            if str(item).strip()
        ]
        if normalized_tag not in current_tags:
            current_tags.append(normalized_tag)

        stored_tags[detection_id] = list(dict.fromkeys(current_tags))

    write_detection_tags(stored_tags)
    return stored_tags


def delete_detection_tags(detection_id: str) -> dict[str, list[str]]:
    stored_tags = read_detection_tags()
    stored_tags.pop(detection_id, None)
    write_detection_tags(stored_tags)
    return stored_tags
