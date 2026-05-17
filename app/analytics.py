import os
from pathlib import Path
from typing import Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from app.schemas import Detection
from app.storage import read_detection_statuses


def _get_detections_parquet_path() -> Path:
    configured_path = os.getenv("DETECTIONS_PARQUET_FILE")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "detections.parquet"


def save_detections_to_parquet(
    detections: Sequence[Detection], default_status: str = "to_verify"
) -> Path:
    parquet_file = _get_detections_parquet_path()
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "detection_id": detection.detection_id,
            "analysis_id": detection.analysis_id,
            "class_name": detection.class_name,
            "confidence": float(detection.confidence),
            "bbox_x": float(detection.bbox.x),
            "bbox_y": float(detection.bbox.y),
            "bbox_width": float(detection.bbox.width),
            "bbox_height": float(detection.bbox.height),
            "status": default_status,
        }
        for detection in detections
    ]

    if not rows:
        return parquet_file

    new_table = pa.Table.from_pylist(rows)

    if parquet_file.exists():
        existing_table = pq.read_table(parquet_file)
        combined_table = pa.concat_tables(
            [existing_table, new_table], promote_options="default"
        )
        pq.write_table(combined_table, parquet_file)
    else:
        pq.write_table(new_table, parquet_file)

    return parquet_file


def query_detections(
    status: str | None = None,
    class_name: str | None = None,
    min_confidence: float | None = None,
) -> list[dict]:
    parquet_file = _get_detections_parquet_path()
    if not parquet_file.exists():
        return []

    stored_statuses = read_detection_statuses()

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
                COALESCE(s.status, d.status, 'to_verify') AS status
            FROM read_parquet(?) AS d
            LEFT JOIN status_overrides AS s USING (detection_id)
            WHERE (? IS NULL OR COALESCE(s.status, d.status, 'to_verify') = ?)
              AND (? IS NULL OR d.class_name = ?)
              AND (? IS NULL OR d.confidence >= ?)
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
            }
            for row in rows
        ]
    finally:
        connection.close()
