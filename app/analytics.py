import os
from pathlib import Path
from typing import Sequence
from datetime import datetime, timezone
from uuid import uuid4

import duckdb
import pandas as pd
from PIL import Image

from app.schemas import Detection
from app.storage import read_detection_comments, read_detection_statuses


def _get_detections_parquet_path() -> Path:
    configured_path = os.getenv("DETECTIONS_PARQUET_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detections.parquet"


def _get_no_detections_parquet_path() -> Path:
    configured_path = os.getenv("NO_DETECTIONS_PARQUET_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "no_detections.parquet"


def _get_no_detections_image_dir() -> Path:
    configured_path = os.getenv("NO_DETECTIONS_IMAGE_DIR")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "images" / "no_detections"


def _append_rows_to_parquet(parquet_file: Path, rows: list[dict]) -> Path:
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return parquet_file

    new_frame = pd.DataFrame(rows)

    if parquet_file.exists():
        existing_frame = pd.read_parquet(parquet_file)
        combined_frame = pd.concat([existing_frame, new_frame], ignore_index=True)
        combined_frame.to_parquet(parquet_file, index=False)
    else:
        new_frame.to_parquet(parquet_file, index=False)

    return parquet_file


def save_no_detections_image_and_metadata(
    image: Image.Image,
    *,
    lon: float,
    lat: float,
    resolution: str,
    timestamp: str | None = None,
) -> dict[str, str | float]:
    analysis_timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    image_id = f"img-{uuid4().hex}"
    timestamp_token = analysis_timestamp.replace(":", "-")
    filename = (
        f"{timestamp_token}_lat-{lat:.6f}_lon-{lon:.6f}_{resolution}_{image_id}.png"
    )

    image_dir = _get_no_detections_image_dir()
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / filename
    image.save(image_path, format="PNG")

    metadata_row: dict[str, str | float] = {
        "image_id": image_id,
        "path": str(image_path),
        "status": "no_detections",
        "lat": float(lat),
        "lon": float(lon),
        "resolution": resolution,
        "timestamp": analysis_timestamp,
    }

    parquet_file = _get_no_detections_parquet_path()
    _append_rows_to_parquet(parquet_file, [metadata_row])

    return metadata_row


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
            "analysis_id": detection.analysis_id,
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
            "resolutionMode": resolution_mode,
            "timestamp": analysis_timestamp,
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
) -> list[dict]:
    parquet_file = _get_detections_parquet_path()
    if not parquet_file.exists():
        return []

    stored_statuses = read_detection_statuses()
    stored_comments = read_detection_comments()

    connection = duckdb.connect(database=":memory:")
    try:
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

        query = """
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
                COALESCE(c.comment, '') AS comment
            FROM read_parquet(?) AS d
            LEFT JOIN status_overrides AS s USING (detection_id)
            LEFT JOIN comment_overrides AS c USING (detection_id)
            WHERE (? IS NULL OR COALESCE(s.status, d.status, 'to_verify') = ?)
              AND (? IS NULL OR d.class_name = ?)
              AND (? IS NULL OR d.confidence >= ?)
                            AND (? IS NULL OR d.resolutionMode = ?)
                            AND (? IS NULL OR d.analysis_id = ?)
            ORDER BY d.confidence DESC
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
            }
            for row in rows
        ]
    finally:
        connection.close()
