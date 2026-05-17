import logging
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai.adapter import run_inference
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


@app.post("/analysis/run", response_model=AnalysisRunResponse)
def run_analysis(payload: AnalysisRunRequest) -> AnalysisRunResponse:
    analysis_id = str(uuid4())

    adapter_detections = run_inference(image=payload.region_id)

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
        save_detections_to_parquet(filtered_detections)
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
    )
