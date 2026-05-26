import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ai.adapter import run_inference
from data.downloader import download_tile
from app.schemas import (
    AnalysisRunRequest,
    AnalysisRunResponse,
    BBox,
    DetectionDeleteResponse,
    Detection,
    DetectionCommentUpdateRequest,
    DetectionCommentUpdateResponse,
    DetectionsQueryParams,
    DetectionStatusUpdateRequest,
    DetectionStatusUpdateResponse,
    HealthResponse,
    LocalValidationRunRequest,
)
from app.analytics import (
    delete_detection_and_related_assets,
    get_analysis_image_path,
    get_no_detection_image_path,
    query_analysis_images,
    query_detections,
    query_no_detections,
    save_analysis_image_and_metadata,
    save_detections_to_parquet,
)
from app.storage import (
    delete_detection_comment,
    delete_detection_status,
    read_detection_statuses,
    upsert_detection_comment,
    upsert_detection_status,
)

app = FastAPI(title="LunaX API", version="0.1.0")
logger = logging.getLogger(__name__)

IMAGE_WIDTH = 2048.0
IMAGE_HEIGHT = 1024.0
VALIDATION_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

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


def _build_sample_bbox(
    base_bbox: list[float], sample_index: int, total_samples: int
) -> list[float]:
    if total_samples <= 1:
        return list(base_bbox)

    x_min, y_min, x_max, y_max = base_bbox
    cols = math.ceil(math.sqrt(total_samples))
    rows = math.ceil(total_samples / cols)

    cell_width = (x_max - x_min) / cols
    cell_height = (y_max - y_min) / rows

    col = sample_index % cols
    row = sample_index // cols

    sample_x_min = x_min + col * cell_width
    sample_y_min = y_min + row * cell_height
    sample_x_max = x_max if col == cols - 1 else x_min + (col + 1) * cell_width
    sample_y_max = y_max if row == rows - 1 else y_min + (row + 1) * cell_height

    return [sample_x_min, sample_y_min, sample_x_max, sample_y_max]


def _get_validation_image_dir() -> Path:
    configured_path = os.getenv("VALIDATION_IMAGE_DIR")
    if configured_path:
        return Path(configured_path)

    return Path(__file__).resolve().parents[1] / "data" / "images" / "validation"


def _get_validation_image_paths(validation_dir: Path) -> list[Path]:
    if not validation_dir.exists() or not validation_dir.is_dir():
        return []

    return sorted(
        [
            path
            for path in validation_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALIDATION_IMAGE_EXTENSIONS
        ]
    )


def _pixel_bbox_to_display_bbox(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    x1 = _clamp(x, 0.0, float(image_width))
    y1 = _clamp(y, 0.0, float(image_height))
    x2 = _clamp(x + width, 0.0, float(image_width))
    y2 = _clamp(y + height, 0.0, float(image_height))

    x_min = (min(x1, x2) / float(image_width)) * 180.0
    y_min = (min(y1, y2) / float(image_height)) * 90.0
    x_max = (max(x1, x2) / float(image_width)) * 180.0
    y_max = (max(y1, y2) / float(image_height)) * 90.0

    return {
        "x": x_min,
        "y": y_min,
        "width": max(0.0, x_max - x_min),
        "height": max(0.0, y_max - y_min),
    }


@app.post("/analysis/run", response_model=AnalysisRunResponse)
def run_analysis(payload: AnalysisRunRequest) -> AnalysisRunResponse:
    analysis_id = str(uuid4())
    analysis_timestamp = datetime.now(timezone.utc).isoformat()
    geo_bbox = _normalize_analysis_bbox_to_geo(payload.bbox)

    filtered_detections: list[Detection] = []
    sample_download_errors: list[str] = []
    for sample_index in range(payload.num_samples):
        sample_bbox = _build_sample_bbox(geo_bbox, sample_index, payload.num_samples)
        center_lon = (sample_bbox[0] + sample_bbox[2]) / 2.0
        center_lat = (sample_bbox[1] + sample_bbox[3]) / 2.0

        try:
            tile_image = download_tile(
                payload.resolution_mode,
                sample_bbox,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            sample_download_errors.append(str(exc))
            logger.warning(
                "Skipping sample %s/%s due to WMS error: %s",
                sample_index + 1,
                payload.num_samples,
                exc,
            )
            continue
        sample_detections = run_inference(
            image=tile_image,
            confidence_threshold=payload.confidence_threshold,
        )

        sample_model_detections = [
            Detection(
                detection_id=detection["detection_id"],
                analysis_id=analysis_id,
                confidence=detection["confidence"],
                **{"class": detection["class"]},
                bbox=BBox(**detection["bbox"]),
            )
            for detection in sample_detections
        ]

        sample_filtered_detections = [
            detection
            for detection in sample_model_detections
            if detection.confidence >= payload.confidence_threshold
        ]

        sample_status = "to_verify" if sample_filtered_detections else "no_detections"

        if sample_filtered_detections:
            filtered_detections.extend(sample_filtered_detections)

        try:
            save_analysis_image_and_metadata(
                tile_image,
                analysis_id=analysis_id,
                lon=center_lon,
                lat=center_lat,
                resolution=payload.resolution_mode,
                status=sample_status,
                timestamp=analysis_timestamp,
            )
        except Exception as exc:  # pragma: no cover - defensive logging for IO layer
            logger.warning("Could not persist analysis image metadata: %s", exc)

    if sample_download_errors and len(sample_download_errors) == payload.num_samples:
        raise HTTPException(
            status_code=502,
            detail=(
                "No valid imagery for selected WMS source/layer and area: "
                f"{sample_download_errors[0]}"
            ),
        )

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


@app.post("/analysis/local-run", response_model=AnalysisRunResponse)
def run_local_validation_analysis(payload: LocalValidationRunRequest) -> AnalysisRunResponse:
    analysis_id = str(uuid4())
    analysis_timestamp = datetime.now(timezone.utc).isoformat()

    validation_dir = _get_validation_image_dir()
    validation_images = _get_validation_image_paths(validation_dir)
    if not validation_images:
        raise HTTPException(
            status_code=404,
            detail=f"No validation images found in {validation_dir}",
        )

    filtered_detections: list[Detection] = []
    for image_path in validation_images:
        try:
            with Image.open(image_path) as opened_image:
                local_image = opened_image.convert("RGB")
        except OSError as exc:
            logger.warning("Could not open validation image %s: %s", image_path, exc)
            continue

        image_width, image_height = local_image.size
        if image_width <= 0 or image_height <= 0:
            logger.warning("Skipping validation image with invalid dimensions: %s", image_path)
            continue

        sample_detections = run_inference(
            image=local_image,
            confidence_threshold=payload.confidence_threshold,
            image_size=max(local_image.size),
        )

        sample_model_detections: list[Detection] = []
        for detection in sample_detections:
            display_bbox = _pixel_bbox_to_display_bbox(
                x=float(detection["bbox"]["x"]),
                y=float(detection["bbox"]["y"]),
                width=float(detection["bbox"]["width"]),
                height=float(detection["bbox"]["height"]),
                image_width=image_width,
                image_height=image_height,
            )

            sample_model_detections.append(
                Detection(
                    detection_id=detection["detection_id"],
                    analysis_id=analysis_id,
                    confidence=detection["confidence"],
                    **{"class": detection["class"]},
                    bbox=BBox(**display_bbox),
                )
            )

        sample_filtered_detections = [
            detection
            for detection in sample_model_detections
            if detection.confidence >= payload.confidence_threshold
        ]
        filtered_detections.extend(sample_filtered_detections)

    try:
        save_detections_to_parquet(
            filtered_detections,
            resolution_mode="detail",
            timestamp=analysis_timestamp,
        )
    except Exception as exc:  # pragma: no cover - defensive logging for IO layer
        logger.warning("Could not persist local validation detections to parquet: %s", exc)

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


@app.delete("/detections/{id}", response_model=DetectionDeleteResponse)
def delete_detection(id: str) -> DetectionDeleteResponse:
    deleted_payload = delete_detection_and_related_assets(detection_id=id)
    if not bool(deleted_payload.get("detection_deleted")):
        raise HTTPException(status_code=404, detail="Detection not found")

    delete_detection_status(detection_id=id)
    delete_detection_comment(detection_id=id)

    return DetectionDeleteResponse(
        detection_id=id,
        detection_deleted=True,
        deleted_image_id=(
            None
            if deleted_payload.get("deleted_image_id") is None
            else str(deleted_payload["deleted_image_id"])
        ),
        deleted_image_path=(
            None
            if deleted_payload.get("deleted_image_path") is None
            else str(deleted_payload["deleted_image_path"])
        ),
    )


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


@app.get("/no-detections/query", response_model=list[dict])
def get_no_detections_query() -> list[dict]:
    return query_no_detections()


@app.get("/analysis-images/query", response_model=list[dict])
def get_analysis_images_query() -> list[dict]:
    return query_analysis_images()


@app.get("/no-detections/image/{image_id}")
def get_no_detection_image(image_id: str) -> FileResponse:
    image_path = get_no_detection_image_path(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="No-detection image not found")

    return FileResponse(path=image_path, media_type="image/png", filename=image_path.name)


@app.get("/analysis-images/image/{image_id}")
def get_analysis_image(image_id: str) -> FileResponse:
    image_path = get_analysis_image_path(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Analysis image not found")

    return FileResponse(path=image_path, media_type="image/png", filename=image_path.name)
