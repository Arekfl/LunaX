import logging
from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.adapter import run_inference
from data.downloader import download_tile
from app.schemas import (
    AnalysisRunRequest,
    AnalysisRunResponse,
    BBox,
    Detection,
    DetectionCommentUpdateRequest,
    DetectionCommentUpdateResponse,
    DetectionsQueryParams,
    DetectionStatusUpdateRequest,
    DetectionStatusUpdateResponse,
    HealthResponse,
)
from app.analytics import query_detections, save_detections_to_parquet
from app.storage import (
    read_detection_statuses,
    upsert_detection_comment,
    upsert_detection_status,
)

app = FastAPI(title="LunaX API", version="0.1.0")
logger = logging.getLogger(__name__)

IMAGE_WIDTH = 2048.0
IMAGE_HEIGHT = 1024.0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _pixel_bbox_to_geo_bbox(pixel_bbox: list[float]) -> list[float]:
    # Converts [x_min, y_min, x_max, y_max] pixels to [lon_min, lat_min, lon_max, lat_max].
    x_min, y_min, x_max, y_max = pixel_bbox

    lon_min = (x_min / IMAGE_WIDTH) * 360.0 - 180.0
    lon_max = (x_max / IMAGE_WIDTH) * 360.0 - 180.0
    lat_a = 90.0 - (y_min / IMAGE_HEIGHT) * 180.0
    lat_b = 90.0 - (y_max / IMAGE_HEIGHT) * 180.0

    return [
        _clamp(min(lon_min, lon_max), -180.0, 180.0),
        _clamp(min(lat_a, lat_b), -90.0, 90.0),
        _clamp(max(lon_min, lon_max), -180.0, 180.0),
        _clamp(max(lat_a, lat_b), -90.0, 90.0),
    ]


def _is_geo_bbox(bbox: list[float]) -> bool:
    x_min, y_min, x_max, y_max = bbox
    return (
        -180.0 <= x_min <= 180.0
        and -180.0 <= x_max <= 180.0
        and -90.0 <= y_min <= 90.0
        and -90.0 <= y_max <= 90.0
    )


def _normalize_analysis_bbox_to_geo(bbox: list[float]) -> list[float]:
    # Backward compatible: accepts either pixel bbox or lon/lat bbox.
    if _is_geo_bbox(bbox):
        return bbox
    return _pixel_bbox_to_geo_bbox(bbox)


@app.post("/analysis/run", response_model=AnalysisRunResponse)
def run_analysis(payload: AnalysisRunRequest) -> AnalysisRunResponse:
    analysis_id = str(uuid4())
    analysis_timestamp = datetime.now(timezone.utc).isoformat()
    geo_bbox = _normalize_analysis_bbox_to_geo(payload.bbox)

    adapter_detections = []
    for _ in range(payload.num_samples):
        tile_image = download_tile(payload.resolution_mode, geo_bbox)
        sample_detections = run_inference(image=tile_image)
        adapter_detections.extend(sample_detections)

    mock_detections = [
        Detection(
            detection_id=detection["detection_id"],
            analysis_id=analysis_id,
            confidence=detection["confidence"],
            **{"class": detection["class"]},
            bbox=BBox(**detection["bbox"]),
        )
        for detection in adapter_detections
    ]

    filtered_detections = [
        detection
        for detection in mock_detections
        if detection.confidence >= payload.confidence_threshold
    ]

    try:
        save_detections_to_parquet(
            filtered_detections,
            resolution_mode=payload.resolution_mode,
            timestamp=analysis_timestamp,
        )
    except Exception as exc:  # pragma: no cover - defensive logging for IO layer
        logger.warning("Could not persist detections to parquet: %s", exc)

    return AnalysisRunResponse(
        analysis_id=analysis_id,
        source="mock",
        detections=filtered_detections,
    )


@app.patch("/detections/{id}/status", response_model=DetectionStatusUpdateResponse)
def update_detection_status(
    id: str, payload: DetectionStatusUpdateRequest
) -> DetectionStatusUpdateResponse:
    upsert_detection_status(detection_id=id, status=payload.status)
    return DetectionStatusUpdateResponse(detection_id=id, status=payload.status)


@app.patch("/detections/{id}/comment", response_model=DetectionCommentUpdateResponse)
def update_detection_comment(
    id: str, payload: DetectionCommentUpdateRequest
) -> DetectionCommentUpdateResponse:
    upsert_detection_comment(detection_id=id, comment=payload.comment)
    return DetectionCommentUpdateResponse(detection_id=id, comment=payload.comment)


@app.get("/detections/statuses", response_model=dict[str, str])
def get_detection_statuses() -> dict[str, str]:
    return read_detection_statuses()


@app.get("/detections/query", response_model=list[dict])
def get_detections_query(
    params: Annotated[DetectionsQueryParams, Depends()]
) -> list[dict]:
    return query_detections(
        status=params.status,
        class_name=params.class_name,
        min_confidence=params.confidence,
        resolution_mode=params.resolution_mode,
        analysis_id=params.analysis_id,
    )
