import os
import hashlib
import math
from io import BytesIO
from pathlib import Path
from typing import Literal, Sequence
from datetime import datetime, timezone
from uuid import uuid4

import duckdb
import pandas as pd
from PIL import Image

from app.schemas import Detection
from app.storage import (
    read_detection_comments,
    read_detection_statuses,
    read_detection_tags,
)


def _get_detections_parquet_path() -> Path:
    configured_path = os.getenv("DETECTIONS_PARQUET_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detections.parquet"


def _get_no_detections_parquet_path() -> Path:
    # Backward compatibility shim: analysis image metadata now lives in detections.parquet.
    return _get_detections_parquet_path()


def _get_no_detections_image_dir() -> Path:
    configured_path = os.getenv("NO_DETECTIONS_IMAGE_DIR")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "images" / "no_detections"


def _normalize_analysis_image_status(status: str | None) -> str:
    normalized_status = str(status or "").strip().lower()
    if not normalized_status:
        return "no_detections"

    if normalized_status in {"no_detection", "no_detections"}:
        return "no_detections"

    # Backward compatibility for rows created before status-specific folders.
    if normalized_status == "detections":
        return "to_verify"

    return normalized_status


def _status_for_analysis_image_storage(status: str | None) -> str:
    normalized_status = _normalize_analysis_image_status(status)
    if normalized_status == "no_detections":
        return "no_detection"
    return normalized_status


def _filter_image_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    if "image_id" not in frame.columns or "path" not in frame.columns:
        return frame.iloc[0:0].copy()

    image_ids = frame["image_id"].fillna("").astype(str).str.strip()
    paths = frame["path"].fillna("").astype(str).str.strip()
    return frame[(image_ids != "") & (paths != "")]


def _get_analysis_image_dir(status: str) -> Path:
    normalized_status = _normalize_analysis_image_status(status)
    if normalized_status == "no_detections":
        return _get_no_detections_image_dir()

    return _get_no_detections_image_dir().parent / normalized_status


def _append_rows_to_parquet(parquet_file: Path, rows: list[dict]) -> Path:
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return parquet_file

    new_frame = pd.DataFrame(rows)

    if parquet_file.exists():
        existing_frame = pd.read_parquet(parquet_file)
        if existing_frame.empty:
            combined_frame = new_frame
        else:
            columns = list(dict.fromkeys([*existing_frame.columns, *new_frame.columns]))
            existing_aligned = existing_frame.reindex(columns=columns)
            new_aligned = new_frame.reindex(columns=columns)
            combined_frame = pd.concat([existing_aligned, new_aligned], ignore_index=True)
        combined_frame.to_parquet(parquet_file, index=False)
    else:
        new_frame.to_parquet(parquet_file, index=False)

    return parquet_file


def _compute_png_hash_and_bytes(image: Image.Image) -> tuple[str, bytes]:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    content_hash = hashlib.sha256(png_bytes).hexdigest()
    return content_hash, png_bytes


def _find_analysis_image_row_by_hash(
    parquet_file: Path, content_hash: str, *, status: str | None = None
) -> dict[str, str | float] | None:
    if not parquet_file.exists():
        return None

    frame = _filter_image_rows(pd.read_parquet(parquet_file))
    if frame.empty or "content_hash" not in frame.columns:
        return None

    matching = frame[frame["content_hash"].astype(str) == content_hash]
    if matching.empty:
        return None

    if status is not None:
        requested_status = _normalize_analysis_image_status(status)
        if "status" not in matching.columns:
            if requested_status != "no_detections":
                return None
        else:
            matching_statuses = (
                matching["status"]
                .fillna("no_detections")
                .astype(str)
                .map(_normalize_analysis_image_status)
            )
            matching = matching[matching_statuses == requested_status]
            if matching.empty:
                return None

    if "timestamp" in matching.columns:
        matching = matching.sort_values(by="timestamp", ascending=False, na_position="last")

    row = matching.iloc[0]
    return {
        "image_id": str(row.get("image_id") or ""),
        "analysis_id": str(row.get("analysis_id") or ""),
        "path": str(row.get("path") or ""),
        "status": _normalize_analysis_image_status(str(row.get("status") or "no_detections")),
        "lat": float(row["lat"]) if "lat" in matching.columns and pd.notna(row["lat"]) else 0.0,
        "lon": float(row["lon"]) if "lon" in matching.columns and pd.notna(row["lon"]) else 0.0,
        "resolution": str(row.get("resolution") or ""),
        "timestamp": str(row.get("timestamp") or ""),
        "content_hash": str(row.get("content_hash") or ""),
    }


def save_analysis_image_and_metadata(
    image: Image.Image,
    *,
    analysis_id: str | None = None,
    lon: float,
    lat: float,
    resolution: str,
    status: str = "no_detections",
    timestamp: str | None = None,
) -> dict[str, str | float]:
    analysis_timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    normalized_status = _normalize_analysis_image_status(status)
    stored_status = _status_for_analysis_image_storage(status)
    parquet_file = _get_detections_parquet_path()
    content_hash, png_bytes = _compute_png_hash_and_bytes(image)
    image_id = f"img-{uuid4().hex}"

    existing_row = _find_analysis_image_row_by_hash(
        parquet_file,
        content_hash,
        status=normalized_status,
    )
    image_path: Path | None = None
    if existing_row is not None:
        existing_path_raw = existing_row.get("path")
        if isinstance(existing_path_raw, str) and existing_path_raw:
            candidate_path = Path(existing_path_raw)
            if candidate_path.exists() and candidate_path.is_file():
                image_path = candidate_path

    if image_path is None:
        timestamp_token = analysis_timestamp.replace(":", "-")
        filename = (
            f"{timestamp_token}_lat-{lat:.6f}_lon-{lon:.6f}_{resolution}_{image_id}.png"
        )

        image_dir = _get_analysis_image_dir(normalized_status)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / filename
        image_path.write_bytes(png_bytes)

    metadata_row: dict[str, str | float] = {
        "detection_id": "",
        "image_id": image_id,
        "analysis_id": str(analysis_id) if analysis_id else "",
        "path": str(image_path),
        "status": stored_status,
        "class": None,
        "class_name": None,
        "confidence": None,
        "bbox": None,
        "bbox_x": None,
        "bbox_y": None,
        "bbox_width": None,
        "bbox_height": None,
        "lat": float(lat),
        "lon": float(lon),
        "resolution": resolution,
        "resolutionMode": resolution,
        "timestamp": analysis_timestamp,
        "content_hash": content_hash,
        "comment": "",
        "tags": None,
    }

    _append_rows_to_parquet(parquet_file, [metadata_row])

    return metadata_row


def save_no_detections_image_and_metadata(
    image: Image.Image,
    *,
    analysis_id: str | None = None,
    lon: float,
    lat: float,
    resolution: str,
    timestamp: str | None = None,
) -> dict[str, str | float]:
    return save_analysis_image_and_metadata(
        image,
        analysis_id=analysis_id,
        lon=lon,
        lat=lat,
        resolution=resolution,
        status="no_detections",
        timestamp=timestamp,
    )


def query_analysis_images(status: str | None = None) -> list[dict]:
    parquet_file = _get_detections_parquet_path()
    if not parquet_file.exists():
        return []

    analysis_images_frame = _filter_image_rows(pd.read_parquet(parquet_file))
    if analysis_images_frame.empty:
        return []

    stored_tags = read_detection_tags()

    expected_columns = [
        "image_id",
        "analysis_id",
        "path",
        "status",
        "lat",
        "lon",
        "resolution",
        "timestamp",
    ]

    if "resolution" not in analysis_images_frame.columns:
        if "resolutionMode" in analysis_images_frame.columns:
            analysis_images_frame["resolution"] = (
                analysis_images_frame["resolutionMode"].fillna("").astype(str)
            )
        else:
            analysis_images_frame["resolution"] = ""
    elif "resolutionMode" in analysis_images_frame.columns:
        empty_resolution_mask = (
            analysis_images_frame["resolution"].fillna("").astype(str).str.strip() == ""
        )
        if empty_resolution_mask.any():
            analysis_images_frame.loc[empty_resolution_mask, "resolution"] = (
                analysis_images_frame.loc[empty_resolution_mask, "resolutionMode"]
                .fillna("")
                .astype(str)
            )

    for column_name in expected_columns:
        if column_name not in analysis_images_frame.columns:
            analysis_images_frame[column_name] = "no_detection" if column_name == "status" else None

    analysis_images_frame["status"] = (
        analysis_images_frame["status"]
        .fillna("no_detection")
        .astype(str)
        .map(_normalize_analysis_image_status)
    )

    if status is not None:
        requested_status = _normalize_analysis_image_status(status)
        analysis_images_frame = analysis_images_frame[analysis_images_frame["status"] == requested_status]

    if analysis_images_frame.empty:
        return []

    sorted_frame = analysis_images_frame[expected_columns].sort_values(
        by="timestamp", ascending=False, na_position="last"
    )

    records: list[dict] = []
    for row in sorted_frame.to_dict(orient="records"):
        image_id = str(row["image_id"]).strip() if row["image_id"] is not None else ""
        records.append(
            {
                "image_id": image_id,
                "analysis_id": str(row["analysis_id"]) if row["analysis_id"] is not None else "",
                "path": str(row["path"]) if row["path"] is not None else "",
                "status": str(row["status"]) if row["status"] is not None else "no_detections",
                "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
                "resolution": str(row["resolution"]) if row["resolution"] is not None else "",
                "timestamp": str(row["timestamp"]) if row["timestamp"] is not None else "",
                "tags": [
                    str(tag)
                    for tag in stored_tags.get(image_id, [])
                    if str(tag).strip()
                ],
            }
        )

    return records


def query_no_detections() -> list[dict]:
    return query_analysis_images(status="no_detections")


def get_analysis_image_path(image_id: str, *, status: str | None = None) -> Path | None:
    parquet_file = _get_detections_parquet_path()
    if not parquet_file.exists():
        return None

    analysis_images_frame = _filter_image_rows(pd.read_parquet(parquet_file))
    if analysis_images_frame.empty or "image_id" not in analysis_images_frame.columns:
        return None

    requested_image_id = str(image_id).strip()
    if not requested_image_id:
        return None

    filtered_frame = analysis_images_frame[
        analysis_images_frame["image_id"].fillna("").astype(str).str.strip() == requested_image_id
    ]
    if filtered_frame.empty or "path" not in filtered_frame.columns:
        return None

    if status is not None:
        requested_status = _normalize_analysis_image_status(status)
        if "status" not in filtered_frame.columns:
            if requested_status != "no_detections":
                return None
        else:
            filtered_frame = filtered_frame[
                filtered_frame["status"]
                .fillna("no_detection")
                .astype(str)
                .map(_normalize_analysis_image_status)
                == requested_status
            ]
        if filtered_frame.empty:
            return None

    if "timestamp" in filtered_frame.columns:
        filtered_frame = filtered_frame.sort_values(
            by="timestamp", ascending=False, na_position="last"
        )

    path_value = filtered_frame.iloc[0]["path"]
    if path_value is None:
        return None

    image_path = Path(str(path_value)).expanduser().resolve()
    if status is None:
        allowed_root = _get_no_detections_image_dir().expanduser().resolve().parent
    else:
        allowed_root = _get_analysis_image_dir(status).expanduser().resolve()

    try:
        image_path.relative_to(allowed_root)
    except ValueError:
        return None

    if not image_path.exists() or not image_path.is_file():
        return None

    return image_path


def get_no_detection_image_path(image_id: str) -> Path | None:
    return get_analysis_image_path(image_id, status="no_detections")


def _build_empty_parquet_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy()


def _extract_detection_bbox_center(row: pd.Series) -> tuple[float, float] | None:
    required_columns = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
    if any(column_name not in row.index for column_name in required_columns):
        return None

    try:
        bbox_x = float(row["bbox_x"])
        bbox_y = float(row["bbox_y"])
        bbox_width = float(row["bbox_width"])
        bbox_height = float(row["bbox_height"])
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in [bbox_x, bbox_y, bbox_width, bbox_height]):
        return None

    if bbox_width <= 0 or bbox_height <= 0:
        return None

    center_lon = bbox_x + bbox_width / 2.0
    center_lat = bbox_y + bbox_height / 2.0
    if not (-180.0 <= center_lon <= 180.0 and -90.0 <= center_lat <= 90.0):
        return None

    return center_lon, center_lat


def _delete_related_analysis_image(
    *,
    analysis_id: str,
    detection_center: tuple[float, float] | None,
) -> dict[str, str] | None:
    if not analysis_id:
        return None

    metadata_file = _get_detections_parquet_path()
    if not metadata_file.exists():
        return None

    metadata_frame = pd.read_parquet(metadata_file)
    image_rows = _filter_image_rows(metadata_frame)
    if image_rows.empty or "analysis_id" not in image_rows.columns:
        return None

    analysis_rows = image_rows[image_rows["analysis_id"].astype(str) == analysis_id]
    if analysis_rows.empty:
        return None

    if "status" in analysis_rows.columns:
        normalized_statuses = (
            analysis_rows["status"]
            .fillna("no_detection")
            .astype(str)
            .map(_normalize_analysis_image_status)
        )
        with_detections = analysis_rows[normalized_statuses != "no_detections"]
        if not with_detections.empty:
            analysis_rows = with_detections

    selected_index = analysis_rows.index[0]

    if (
        detection_center is not None
        and "lat" in analysis_rows.columns
        and "lon" in analysis_rows.columns
    ):
        center_lon, center_lat = detection_center
        geo_rows = analysis_rows.dropna(subset=["lat", "lon"])
        if not geo_rows.empty:
            distances = (geo_rows["lon"].astype(float) - center_lon) ** 2 + (
                geo_rows["lat"].astype(float) - center_lat
            ) ** 2
            selected_index = distances.idxmin()
        elif "timestamp" in analysis_rows.columns:
            selected_index = analysis_rows.sort_values(
                by="timestamp", ascending=False, na_position="last"
            ).index[0]
    elif "timestamp" in analysis_rows.columns:
        selected_index = analysis_rows.sort_values(
            by="timestamp", ascending=False, na_position="last"
        ).index[0]

    selected_row = image_rows.loc[selected_index]
    removed_image_id = str(selected_row.get("image_id") or "")
    removed_path_raw = str(selected_row.get("path") or "")

    updated_frame = metadata_frame.drop(index=selected_index)
    if updated_frame.empty:
        updated_frame = _build_empty_parquet_like(metadata_frame)
    updated_frame.to_parquet(metadata_file, index=False)

    if removed_path_raw and "path" in updated_frame.columns:
        still_referenced = (updated_frame["path"].astype(str) == removed_path_raw).any()
    else:
        still_referenced = False

    if removed_path_raw and not still_referenced:
        image_path = Path(removed_path_raw).expanduser().resolve()
        allowed_root = _get_no_detections_image_dir().expanduser().resolve().parent

        try:
            image_path.relative_to(allowed_root)
        except ValueError:
            image_path = None

        if image_path is not None and image_path.exists() and image_path.is_file():
            image_path.unlink()

    return {
        "image_id": removed_image_id,
        "path": removed_path_raw,
    }


def delete_detection_and_related_assets(
    detection_id: str,
    *,
    delete_images: bool = False,
) -> dict[str, str | bool | None]:
    detections_file = _get_detections_parquet_path()
    if not detections_file.exists():
        return {
            "detection_deleted": False,
            "deleted_image_id": None,
            "deleted_image_path": None,
            "related_image_missing": False,
            "related_image_in_use": False,
        }

    detections_frame = pd.read_parquet(detections_file)
    if detections_frame.empty or "detection_id" not in detections_frame.columns:
        return {
            "detection_deleted": False,
            "deleted_image_id": None,
            "deleted_image_path": None,
            "related_image_missing": False,
            "related_image_in_use": False,
        }

    matched_rows = detections_frame[
        detections_frame["detection_id"].astype(str) == str(detection_id)
    ]
    if matched_rows.empty:
        return {
            "detection_deleted": False,
            "deleted_image_id": None,
            "deleted_image_path": None,
            "related_image_missing": False,
            "related_image_in_use": False,
        }

    if "timestamp" in matched_rows.columns:
        matched_rows = matched_rows.sort_values(by="timestamp", ascending=False, na_position="last")

    representative_row = matched_rows.iloc[0]
    analysis_id = str(representative_row.get("analysis_id") or "")
    detection_center = _extract_detection_bbox_center(representative_row)

    updated_detections_frame = detections_frame[
        detections_frame["detection_id"].astype(str) != str(detection_id)
    ]
    if updated_detections_frame.empty:
        updated_detections_frame = _build_empty_parquet_like(detections_frame)
    updated_detections_frame.to_parquet(detections_file, index=False)

    if not delete_images:
        return {
            "detection_deleted": True,
            "analysis_id": analysis_id,
            "deleted_image_id": None,
            "deleted_image_path": None,
            "related_image_missing": False,
            "related_image_in_use": False,
        }

    analysis_image_rows = _filter_image_rows(updated_detections_frame)
    has_related_analysis_images = False
    if analysis_id and "analysis_id" in analysis_image_rows.columns:
        has_related_analysis_images = not analysis_image_rows[
            analysis_image_rows["analysis_id"].fillna("").astype(str) == analysis_id
        ].empty

    if not has_related_analysis_images:
        return {
            "detection_deleted": True,
            "analysis_id": analysis_id,
            "deleted_image_id": None,
            "deleted_image_path": None,
            "related_image_missing": True,
            "related_image_in_use": False,
        }

    if analysis_id and "analysis_id" in updated_detections_frame.columns:
        same_analysis_rows = updated_detections_frame[
            updated_detections_frame["analysis_id"].fillna("").astype(str) == analysis_id
        ]
        if "detection_id" in same_analysis_rows.columns:
            same_analysis_rows = same_analysis_rows[
                same_analysis_rows["detection_id"].fillna("").astype(str).str.strip() != ""
            ]

        if not same_analysis_rows.empty:
            return {
                "detection_deleted": True,
                "analysis_id": analysis_id,
                "deleted_image_id": None,
                "deleted_image_path": None,
                "related_image_missing": False,
                "related_image_in_use": True,
            }

    removed_image = _delete_related_analysis_image(
        analysis_id=analysis_id,
        detection_center=detection_center,
    )
    related_image_missing = (
        removed_image is None
        or not bool(str(removed_image.get("image_id") or "").strip())
    )

    return {
        "detection_deleted": True,
        "analysis_id": analysis_id,
        "deleted_image_id": None if removed_image is None else removed_image.get("image_id") or None,
        "deleted_image_path": None if removed_image is None else removed_image.get("path") or None,
        "related_image_missing": related_image_missing,
        "related_image_in_use": False,
    }


def delete_detections_bulk_and_related_assets(
    detection_ids: Sequence[str],
    *,
    delete_images: bool = False,
) -> dict[str, int | bool | list[str]]:
    unique_detection_ids: list[str] = []
    seen_ids: set[str] = set()

    for raw_detection_id in detection_ids:
        detection_id = str(raw_detection_id).strip()
        if not detection_id or detection_id in seen_ids:
            continue

        seen_ids.add(detection_id)
        unique_detection_ids.append(detection_id)

    deleted_detection_ids: list[str] = []
    missing_detection_ids: list[str] = []
    related_image_missing = False
    related_image_in_use = False
    related_image_missing_count = 0
    related_image_in_use_count = 0
    touched_analysis_ids: set[str] = set()
    in_use_analysis_ids: set[str] = set()

    for detection_id in unique_detection_ids:
        delete_payload = delete_detection_and_related_assets(
            detection_id=detection_id,
            delete_images=delete_images,
        )
        if bool(delete_payload.get("detection_deleted")):
            deleted_detection_ids.append(detection_id)
            analysis_id = str(delete_payload.get("analysis_id") or "").strip()
            if analysis_id:
                touched_analysis_ids.add(analysis_id)
            related_image_missing = related_image_missing or bool(
                delete_payload.get("related_image_missing")
            )
            related_image_in_use = related_image_in_use or bool(
                delete_payload.get("related_image_in_use")
            )
            if bool(delete_payload.get("related_image_missing")):
                related_image_missing_count += 1
            if bool(delete_payload.get("related_image_in_use")):
                related_image_in_use_count += 1
                if analysis_id:
                    in_use_analysis_ids.add(analysis_id)
            continue

        missing_detection_ids.append(detection_id)

    if delete_images and touched_analysis_ids:
        metadata_file = _get_detections_parquet_path()
        if metadata_file.exists():
            metadata_frame = pd.read_parquet(metadata_file)
            has_analysis_id_column = "analysis_id" in metadata_frame.columns
            has_detection_id_column = "detection_id" in metadata_frame.columns
            for analysis_id in touched_analysis_ids:
                if not has_analysis_id_column:
                    continue

                analysis_rows = metadata_frame[
                    metadata_frame["analysis_id"].fillna("").astype(str) == analysis_id
                ]
                if analysis_rows.empty:
                    continue

                if has_detection_id_column:
                    remaining_detection_rows = analysis_rows[
                        analysis_rows["detection_id"].fillna("").astype(str).str.strip() != ""
                    ]
                else:
                    remaining_detection_rows = analysis_rows.iloc[0:0].copy()
                if not remaining_detection_rows.empty:
                    continue

                analysis_image_rows = _filter_image_rows(analysis_rows)
                if "image_id" not in analysis_image_rows.columns:
                    continue
                image_ids_to_delete = [
                    image_id
                    for image_id in analysis_image_rows["image_id"].fillna("").astype(str).str.strip().tolist()
                    if image_id
                ]
                if not image_ids_to_delete:
                    continue

                image_delete_summary = delete_analysis_images_by_ids(image_ids_to_delete)
                missing_image_ids = [
                    str(image_id)
                    for image_id in image_delete_summary.get("missing_image_ids", [])
                    if str(image_id).strip()
                ]
                if missing_image_ids:
                    related_image_missing = True
                    related_image_missing_count += len(missing_image_ids)

            if in_use_analysis_ids:
                refreshed_frame = pd.read_parquet(metadata_file)
                if "analysis_id" not in refreshed_frame.columns or "detection_id" not in refreshed_frame.columns:
                    unresolved_in_use_analyses = set()
                else:
                    unresolved_in_use_analyses = {
                        analysis_id
                        for analysis_id in in_use_analysis_ids
                        if not refreshed_frame[
                            (refreshed_frame["analysis_id"].fillna("").astype(str) == analysis_id)
                            & (
                                refreshed_frame["detection_id"].fillna("").astype(str).str.strip()
                                != ""
                            )
                        ].empty
                    }
                related_image_in_use = bool(unresolved_in_use_analyses)
                related_image_in_use_count = len(unresolved_in_use_analyses)

    return {
        "requested_count": len(unique_detection_ids),
        "deleted_count": len(deleted_detection_ids),
        "deleted_detection_ids": deleted_detection_ids,
        "missing_detection_ids": missing_detection_ids,
        "related_image_missing": related_image_missing,
        "related_image_in_use": related_image_in_use,
        "related_image_missing_count": related_image_missing_count,
        "related_image_in_use_count": related_image_in_use_count,
    }


def get_existing_detection_ids(detection_ids: Sequence[str]) -> set[str]:
    unique_detection_ids: list[str] = []
    seen_ids: set[str] = set()

    for raw_detection_id in detection_ids:
        detection_id = str(raw_detection_id).strip()
        if not detection_id or detection_id in seen_ids:
            continue

        seen_ids.add(detection_id)
        unique_detection_ids.append(detection_id)

    if not unique_detection_ids:
        return set()

    detections_file = _get_detections_parquet_path()
    if not detections_file.exists():
        return set()

    detections_frame = pd.read_parquet(detections_file)
    if detections_frame.empty or "detection_id" not in detections_frame.columns:
        return set()

    requested_ids = set(unique_detection_ids)
    detection_ids_series = (
        detections_frame["detection_id"].fillna("").astype(str).str.strip()
    )

    return {
        detection_id
        for detection_id in detection_ids_series.tolist()
        if detection_id and detection_id in requested_ids
    }


def get_existing_analysis_image_ids(
    image_ids: Sequence[str], *, status: str | None = None
) -> set[str]:
    unique_image_ids: list[str] = []
    seen_ids: set[str] = set()

    for raw_image_id in image_ids:
        image_id = str(raw_image_id).strip()
        if not image_id or image_id in seen_ids:
            continue

        seen_ids.add(image_id)
        unique_image_ids.append(image_id)

    if not unique_image_ids:
        return set()

    metadata_file = _get_detections_parquet_path()
    if not metadata_file.exists():
        return set()

    metadata_frame = _filter_image_rows(pd.read_parquet(metadata_file))
    if metadata_frame.empty or "image_id" not in metadata_frame.columns:
        return set()

    if status is not None:
        requested_status = _normalize_analysis_image_status(status)
        if "status" not in metadata_frame.columns:
            if requested_status != "no_detections":
                return set()
        else:
            metadata_frame = metadata_frame[
                metadata_frame["status"]
                .fillna("no_detection")
                .astype(str)
                .map(_normalize_analysis_image_status)
                == requested_status
            ]
            if metadata_frame.empty:
                return set()

    requested_ids = set(unique_image_ids)
    image_ids_series = metadata_frame["image_id"].fillna("").astype(str).str.strip()
    return {
        image_id
        for image_id in image_ids_series.tolist()
        if image_id and image_id in requested_ids
    }


def delete_analysis_images_by_ids(
    image_ids: Sequence[str],
    *,
    delete_files: bool = False,
) -> dict[str, int | list[str]]:
    unique_image_ids: list[str] = []
    seen_ids: set[str] = set()

    for raw_image_id in image_ids:
        image_id = str(raw_image_id).strip()
        if not image_id or image_id in seen_ids:
            continue

        seen_ids.add(image_id)
        unique_image_ids.append(image_id)

    metadata_file = _get_detections_parquet_path()
    if not metadata_file.exists():
        return {
            "requested_count": len(unique_image_ids),
            "deleted_count": 0,
            "deleted_image_ids": [],
            "missing_image_ids": unique_image_ids,
        }

    metadata_frame = pd.read_parquet(metadata_file)
    image_rows = _filter_image_rows(metadata_frame)
    if image_rows.empty or "image_id" not in image_rows.columns:
        return {
            "requested_count": len(unique_image_ids),
            "deleted_count": 0,
            "deleted_image_ids": [],
            "missing_image_ids": unique_image_ids,
        }

    requested_set = set(unique_image_ids)
    normalized_image_ids = image_rows["image_id"].fillna("").astype(str).str.strip()
    selected_rows = image_rows[normalized_image_ids.isin(requested_set)]

    if selected_rows.empty:
        return {
            "requested_count": len(unique_image_ids),
            "deleted_count": 0,
            "deleted_image_ids": [],
            "missing_image_ids": unique_image_ids,
        }

    deleted_image_ids_set = set(
        selected_rows["image_id"].fillna("").astype(str).str.strip().tolist()
    )
    deleted_image_ids = [
        image_id for image_id in unique_image_ids if image_id in deleted_image_ids_set
    ]
    missing_image_ids = [
        image_id for image_id in unique_image_ids if image_id not in deleted_image_ids_set
    ]

    updated_frame = metadata_frame.drop(index=selected_rows.index)
    if updated_frame.empty:
        updated_frame = _build_empty_parquet_like(metadata_frame)
    updated_frame.to_parquet(metadata_file, index=False)

    if not delete_files:
        return {
            "requested_count": len(unique_image_ids),
            "deleted_count": len(deleted_image_ids),
            "deleted_image_ids": deleted_image_ids,
            "missing_image_ids": missing_image_ids,
        }

    if "path" in selected_rows.columns:
        removed_paths = {
            str(path_value)
            for path_value in selected_rows["path"].dropna().astype(str).tolist()
            if str(path_value).strip()
        }
    else:
        removed_paths = set()

    if "path" in updated_frame.columns:
        remaining_paths = {
            str(path_value)
            for path_value in updated_frame["path"].dropna().astype(str).tolist()
            if str(path_value).strip()
        }
    else:
        remaining_paths = set()

    allowed_root = _get_no_detections_image_dir().expanduser().resolve().parent
    for removed_path_raw in removed_paths:
        if removed_path_raw in remaining_paths:
            continue

        image_path = Path(removed_path_raw).expanduser().resolve()
        try:
            image_path.relative_to(allowed_root)
        except ValueError:
            continue

        if image_path.exists() and image_path.is_file():
            image_path.unlink()

    return {
        "requested_count": len(unique_image_ids),
        "deleted_count": len(deleted_image_ids),
        "deleted_image_ids": deleted_image_ids,
        "missing_image_ids": missing_image_ids,
    }


def delete_analysis_image_by_id(image_id: str, *, delete_files: bool = False) -> bool:
    summary = delete_analysis_images_by_ids([image_id], delete_files=delete_files)
    return int(summary.get("deleted_count", 0)) > 0


def save_detections_to_parquet(
    detections: Sequence[Detection],
    default_status: str = "to_verify",
    resolution_mode: str = "detail",
    timestamp: str | None = None,
) -> Path:
    parquet_file = _get_detections_parquet_path()

    analysis_timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    rows = [
        {
            "detection_id": detection.detection_id,
            "image_id": None,
            "analysis_id": detection.analysis_id,
            "path": None,
            "class": detection.class_name,
            "class_name": detection.class_name,
            "confidence": float(detection.confidence),
            "bbox": {
                "x": float(detection.bbox.x),
                "y": float(detection.bbox.y),
                "width": float(detection.bbox.width),
                "height": float(detection.bbox.height),
            },
            "bbox_x": float(detection.bbox.x),
            "bbox_y": float(detection.bbox.y),
            "bbox_width": float(detection.bbox.width),
            "bbox_height": float(detection.bbox.height),
            "status": default_status,
            "lat": None,
            "lon": None,
            "resolution": resolution_mode,
            "resolutionMode": resolution_mode,
            "timestamp": analysis_timestamp,
            "content_hash": None,
            "comment": "",
            "tags": None,
        }
        for detection in detections
    ]

    if not rows:
        return parquet_file

    return _append_rows_to_parquet(parquet_file, rows)


def query_detections(
    status: str | None = None,
    class_name: str | None = None,
    min_confidence: float | None = None,
    resolution_mode: str | None = None,
    analysis_id: str | None = None,
    sort_by: Literal["confidence", "data"] = "confidence",
    sort_order: Literal["asc", "desc"] = "desc",
) -> list[dict]:
    parquet_file = _get_detections_parquet_path()
    if not parquet_file.exists():
        return []

    stored_statuses = read_detection_statuses()
    stored_comments = read_detection_comments()
    stored_tags = read_detection_tags()

    connection = duckdb.connect(database=":memory:")
    try:
        parquet_columns = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(parquet_file)],
            ).fetchall()
        }
        has_resolution_column = "resolution" in parquet_columns
        has_resolution_mode_column = "resolutionMode" in parquet_columns

        if has_resolution_column and has_resolution_mode_column:
            resolution_expr = "COALESCE(d.resolution, d.resolutionMode, '')"
        elif has_resolution_column:
            resolution_expr = "COALESCE(d.resolution, '')"
        elif has_resolution_mode_column:
            resolution_expr = "COALESCE(d.resolutionMode, '')"
        else:
            resolution_expr = "''"

        connection.execute(
            "CREATE TEMP TABLE status_overrides (detection_id VARCHAR, status VARCHAR)"
        )
        if stored_statuses:
            connection.executemany(
                "INSERT INTO status_overrides (detection_id, status) VALUES (?, ?)",
                list(stored_statuses.items()),
            )

        connection.execute(
            "CREATE TEMP TABLE comment_overrides (detection_id VARCHAR, comment VARCHAR)"
        )
        if stored_comments:
            connection.executemany(
                "INSERT INTO comment_overrides (detection_id, comment) VALUES (?, ?)",
                list(stored_comments.items()),
            )

        normalized_sort_order = "asc" if sort_order == "asc" else "desc"
        if sort_by == "data":
            timestamp_direction = "ASC" if normalized_sort_order == "asc" else "DESC"
            order_by_clause = (
                f"d.timestamp {timestamp_direction} NULLS LAST, "
                "d.confidence DESC NULLS LAST, d.detection_id ASC"
            )
        else:
            confidence_direction = "ASC" if normalized_sort_order == "asc" else "DESC"
            order_by_clause = (
                f"d.confidence {confidence_direction} NULLS LAST, "
                "d.timestamp DESC NULLS LAST, d.detection_id ASC"
            )

        query = f"""
            SELECT
                d.detection_id,
                d.analysis_id,
                d.class_name,
                d.confidence,
                d.bbox_x,
                d.bbox_y,
                d.bbox_width,
                d.bbox_height,
                COALESCE(s.status, d.status, 'to_verify') AS status,
                COALESCE(c.comment, '') AS comment,
                d.timestamp,
                {resolution_expr} AS resolution
            FROM read_parquet(?) AS d
            LEFT JOIN status_overrides AS s USING (detection_id)
            LEFT JOIN comment_overrides AS c USING (detection_id)
            WHERE (? IS NULL OR COALESCE(s.status, d.status, 'to_verify') = ?)
                            AND d.detection_id IS NOT NULL
                            AND TRIM(CAST(d.detection_id AS VARCHAR)) <> ''
                            AND d.confidence IS NOT NULL
                            AND d.bbox_x IS NOT NULL
                            AND d.bbox_y IS NOT NULL
                            AND d.bbox_width IS NOT NULL
                            AND d.bbox_height IS NOT NULL
              AND (? IS NULL OR d.class_name = ?)
              AND (? IS NULL OR d.confidence >= ?)
                            AND (? IS NULL OR {resolution_expr} = ?)
                            AND (? IS NULL OR d.analysis_id = ?)
            ORDER BY {order_by_clause}
        """

        rows = connection.execute(
            query,
            [
                str(parquet_file),
                status,
                status,
                class_name,
                class_name,
                min_confidence,
                min_confidence,
                resolution_mode,
                resolution_mode,
                analysis_id,
                analysis_id,
            ],
        ).fetchall()

        return [
            {
                "detection_id": row[0],
                "analysis_id": row[1],
                "class": row[2],
                "confidence": row[3],
                "bbox": {
                    "x": row[4],
                    "y": row[5],
                    "width": row[6],
                    "height": row[7],
                },
                "status": row[8],
                "comment": row[9],
                "timestamp": str(row[10]) if row[10] is not None else "",
                "resolution": str(row[11]) if row[11] is not None else "",
                "tags": [str(tag) for tag in stored_tags.get(str(row[0]), []) if str(tag).strip()],
            }
            for row in rows
        ]
    finally:
        connection.close()


def validate_detections_bulk(
    detection_ids: Sequence[str],
    *,
    target_status: Literal["confirmed", "rejected"],
) -> dict[str, int | list[str]]:
    """Update status for detections and move their related image files to the target folder.

    Returns a summary dict with counts and lists of updated / missing detection ids.
    """
    unique_detection_ids: list[str] = list(
        dict.fromkeys(str(d).strip() for d in detection_ids if str(d).strip())
    )

    if not unique_detection_ids:
        return {
            "requested_count": 0,
            "updated_count": 0,
            "updated_detection_ids": [],
            "missing_detection_ids": [],
            "files_moved": 0,
            "files_missing": 0,
            "files_in_use": 0,
        }

    parquet_file = _get_detections_parquet_path()
    if not parquet_file.exists():
        return {
            "requested_count": len(unique_detection_ids),
            "updated_count": 0,
            "updated_detection_ids": [],
            "missing_detection_ids": unique_detection_ids,
            "files_moved": 0,
            "files_missing": 0,
            "files_in_use": 0,
        }

    frame = pd.read_parquet(parquet_file)
    if frame.empty or "detection_id" not in frame.columns:
        return {
            "requested_count": len(unique_detection_ids),
            "updated_count": 0,
            "updated_detection_ids": [],
            "missing_detection_ids": unique_detection_ids,
            "files_moved": 0,
            "files_missing": 0,
            "files_in_use": 0,
        }

    detection_ids_series = frame["detection_id"].fillna("").astype(str).str.strip()
    requested_set = set(unique_detection_ids)
    found_ids = {d for d in detection_ids_series.tolist() if d in requested_set}
    missing_detection_ids = [d for d in unique_detection_ids if d not in found_ids]
    updated_detection_ids = [d for d in unique_detection_ids if d in found_ids]

    if not updated_detection_ids:
        return {
            "requested_count": len(unique_detection_ids),
            "updated_count": 0,
            "updated_detection_ids": [],
            "missing_detection_ids": missing_detection_ids,
            "files_moved": 0,
            "files_missing": 0,
            "files_in_use": 0,
        }

    # Update status column in parquet for matching detection rows.
    update_mask = detection_ids_series.isin(set(updated_detection_ids))
    if "status" not in frame.columns:
        frame["status"] = "to_verify"
    frame.loc[update_mask, "status"] = target_status

    # Collect analysis_ids for updated detections to find related image rows.
    updated_rows = frame[update_mask]
    analysis_ids_for_update: set[str] = set()
    if "analysis_id" in updated_rows.columns:
        analysis_ids_for_update = {
            str(a).strip()
            for a in updated_rows["analysis_id"].fillna("").tolist()
            if str(a).strip()
        }

    target_dir = _get_analysis_image_dir(target_status)
    target_dir.mkdir(parents=True, exist_ok=True)
    allowed_root = _get_no_detections_image_dir().expanduser().resolve().parent

    files_moved = 0
    files_missing = 0
    files_in_use = 0

    if analysis_ids_for_update and "image_id" in frame.columns and "path" in frame.columns:
        image_rows = _filter_image_rows(frame)

        # Check if each analysis still has remaining detections with other statuses
        # (to determine if image is "in use" by other detections of same analysis).
        for analysis_id in analysis_ids_for_update:
            if not analysis_id or "analysis_id" not in image_rows.columns:
                continue

            # Image rows for this analysis.
            img_mask = image_rows["analysis_id"].fillna("").astype(str).str.strip() == analysis_id
            analysis_img_rows = image_rows[img_mask]
            if analysis_img_rows.empty:
                continue

            # Check whether other (non-validated) detection rows reference this analysis.
            if "detection_id" in frame.columns:
                other_det_mask = (
                    (frame["detection_id"].fillna("").astype(str).str.strip() != "")
                    & (frame["analysis_id"].fillna("").astype(str).str.strip() == analysis_id)
                    & (~detection_ids_series.isin(set(updated_detection_ids)))
                )
                still_has_other_detections = frame[other_det_mask].shape[0] > 0
            else:
                still_has_other_detections = False

            if still_has_other_detections:
                files_in_use += len(analysis_img_rows)
                continue

            # Move image files for this analysis.
            for idx, img_row in analysis_img_rows.iterrows():
                old_path_raw = str(img_row.get("path") or "").strip()
                if not old_path_raw:
                    files_missing += 1
                    continue

                old_path = Path(old_path_raw).expanduser().resolve()

                try:
                    old_path.relative_to(allowed_root)
                except ValueError:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Image path outside allowed root, skipping: %s", old_path
                    )
                    files_missing += 1
                    continue

                if not old_path.exists() or not old_path.is_file():
                    files_missing += 1
                    continue

                new_path = target_dir / old_path.name
                # Avoid overwriting – append image_id suffix if collides.
                if new_path.exists() and new_path != old_path:
                    image_id_suffix = str(img_row.get("image_id") or "").strip() or "dup"
                    new_path = target_dir / f"{old_path.stem}_{image_id_suffix}{old_path.suffix}"

                try:
                    old_path.rename(new_path)
                    # Update path in dataframe.
                    frame.loc[idx, "path"] = str(new_path)
                    # Also update status for image row.
                    frame.loc[idx, "status"] = _status_for_analysis_image_storage(target_status)
                    files_moved += 1
                except OSError as exc:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Could not move image file %s → %s: %s", old_path, new_path, exc
                    )
                    files_missing += 1

    frame.to_parquet(parquet_file, index=False)

    return {
        "requested_count": len(unique_detection_ids),
        "updated_count": len(updated_detection_ids),
        "updated_detection_ids": updated_detection_ids,
        "missing_detection_ids": missing_detection_ids,
        "files_moved": files_moved,
        "files_missing": files_missing,
        "files_in_use": files_in_use,
    }
