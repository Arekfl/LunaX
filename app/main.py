import logging
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ai.adapter import run_inference
from data.downloader import download_tile
from app.schemas import (
    AnalysisImageBulkDeleteRequest,
    AnalysisImageBulkDeleteResponse,
    AnalysisImageBulkTagsUpdateRequest,
    AnalysisImageBulkTagsUpdateResponse,
    AnalysisReanalysisRequest,
    AnalysisReanalysisResponse,
    AnalysisImageDeleteResponse,
    AnalysisRunRequest,
    AnalysisRunResponse,
    BBox,
    DetectionBulkDeleteRequest,
    DetectionBulkDeleteResponse,
    DetectionBulkTagsUpdateRequest,
    DetectionBulkTagsUpdateResponse,
    DetectionBulkValidateRequest,
    DetectionBulkValidateResponse,
    DetectionDeleteResponse,
    Detection,
    DetectionCommentUpdateRequest,
    DetectionCommentUpdateResponse,
    DetectionTagsUpdateRequest,
    DetectionTagsUpdateResponse,
    DetectionsQueryParams,
    DetectionStatusUpdateRequest,
    DetectionStatusUpdateResponse,
    HealthResponse,
    LocalValidationRunRequest,
)
from app.analytics import (
    delete_analysis_image_by_id,
    delete_analysis_images_by_ids,
    delete_detections_bulk_and_related_assets,
    delete_detection_and_related_assets,
    get_existing_analysis_image_ids,
    get_existing_detection_ids,
    get_analysis_image_path,
    get_no_detection_image_path,
    query_analysis_images,
    query_detections,
    query_no_detections,
    save_analysis_image_and_metadata,
    save_detections_to_parquet,
    validate_detections_bulk,
)
from app.storage import (
    delete_detection_comment,
    delete_detection_status,
    delete_detection_tags,
    read_detection_statuses,
    upsert_detection_comment,
    upsert_detection_status,
    upsert_detection_tags,
    upsert_detection_tags_bulk,
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


def _build_bbox_from_center(
    base_bbox: list[float],
    *,
    center_x: float,
    center_y: float,
    window_width: float,
    window_height: float,
) -> list[float]:
    x_min, y_min, x_max, y_max = base_bbox
    base_width = x_max - x_min
    base_height = y_max - y_min

    width = max(0.0, min(window_width, base_width))
    height = max(0.0, min(window_height, base_height))

    if width == 0.0 or height == 0.0:
        return list(base_bbox)

    sample_x_min = max(x_min, min(center_x - width / 2.0, x_max - width))
    sample_y_min = max(y_min, min(center_y - height / 2.0, y_max - height))
    return [sample_x_min, sample_y_min, sample_x_min + width, sample_y_min + height]


def _uniform_anchor_points(total_samples: int) -> tuple[list[tuple[float, float]], int, int]:
    if total_samples <= 1:
        return [(0.5, 0.5)], 3, 3
    if total_samples == 2:
        return [(0.5, 0.5), (0.0, 0.5)], 3, 3
    if total_samples == 3:
        return [(0.5, 0.5), (0.0, 0.5), (1.0, 0.5)], 3, 3
    if total_samples == 4:
        return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)], 3, 3
    if total_samples == 5:
        return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)], 3, 3

    cols = math.ceil(math.sqrt(total_samples))
    rows = math.ceil(total_samples / cols)
    anchors: list[tuple[float, float]] = []
    for row in range(rows):
        col_indexes = list(range(cols))
        if row % 2 == 1:
            col_indexes.reverse()

        for col in col_indexes:
            anchor_x = (col + 0.5) / cols
            anchor_y = (row + 0.5) / rows
            anchors.append((anchor_x, anchor_y))
            if len(anchors) >= total_samples:
                return anchors, cols, rows

    return anchors[:total_samples], cols, rows


def _build_uniform_sample_bboxes(base_bbox: list[float], total_samples: int) -> list[list[float]]:
    x_min, y_min, x_max, y_max = base_bbox
    width = x_max - x_min
    height = y_max - y_min

    anchors, cols, rows = _uniform_anchor_points(total_samples)
    if total_samples <= 5:
        sample_width = width / 3.0
        sample_height = height / 3.0
    else:
        sample_width = width / cols
        sample_height = height / rows

    sample_bboxes: list[list[float]] = []
    for anchor_x, anchor_y in anchors:
        center_x = x_min + anchor_x * width
        center_y = y_min + anchor_y * height
        sample_bboxes.append(
            _build_bbox_from_center(
                base_bbox,
                center_x=center_x,
                center_y=center_y,
                window_width=sample_width,
                window_height=sample_height,
            )
        )

    return sample_bboxes


def _build_random_sample_bboxes(base_bbox: list[float], total_samples: int) -> list[list[float]]:
    x_min, y_min, x_max, y_max = base_bbox
    width = x_max - x_min
    height = y_max - y_min

    cols = math.ceil(math.sqrt(total_samples))
    rows = math.ceil(total_samples / cols)
    sample_width = width / cols
    sample_height = height / rows

    min_center_x = x_min + sample_width / 2.0
    max_center_x = x_max - sample_width / 2.0
    min_center_y = y_min + sample_height / 2.0
    max_center_y = y_max - sample_height / 2.0

    sample_bboxes: list[list[float]] = []
    for _ in range(total_samples):
        center_x = random.uniform(min_center_x, max_center_x)
        center_y = random.uniform(min_center_y, max_center_y)
        sample_bboxes.append(
            _build_bbox_from_center(
                base_bbox,
                center_x=center_x,
                center_y=center_y,
                window_width=sample_width,
                window_height=sample_height,
            )
        )

    return sample_bboxes


def _build_sample_bboxes(
    base_bbox: list[float],
    total_samples: int,
    sampling_mode: str,
) -> list[list[float]]:
    if total_samples <= 0:
        return []

    if sampling_mode == "random":
        return _build_random_sample_bboxes(base_bbox, total_samples)

    return _build_uniform_sample_bboxes(base_bbox, total_samples)


def _extract_lat_lon_from_path(path_value: str | None) -> tuple[float, float] | None:
    normalized_path = str(path_value or "").strip()
    if not normalized_path:
        return None

    matched = re.search(r"lat-(-?\d+(?:\.\d+)?)_lon-(-?\d+(?:\.\d+)?)", normalized_path)
    if matched is None:
        return None

    lat = float(matched.group(1))
    lon = float(matched.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    return lat, lon


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
    geo_bbox = _normalize_analysis_bbox_to_geo(payload.bbox)
    sample_bboxes = _build_sample_bboxes(geo_bbox, payload.num_samples, payload.sampling_mode)
    requested_samples = len(sample_bboxes)
    analyzed_samples = 0

    filtered_detections: list[Detection] = []
    sample_download_errors: list[str] = []
    for sample_index, sample_bbox in enumerate(sample_bboxes):
        sample_timestamp = datetime.now(timezone.utc).isoformat()
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
        analyzed_samples += 1
        sample_detections = run_inference(
            image=tile_image,
            confidence_threshold=payload.confidence_threshold,
            model_name=payload.model_name,
        )

        sample_model_detections = [
            Detection(
                detection_id=detection["detection_id"],
                analysis_id=analysis_id,
                confidence=detection["confidence"],
                **{"class": detection["class"]},
                class_id=detection["class_id"],
                bbox=BBox(**detection["bbox"]),
            )
            for detection in sample_detections
        ]

        sample_filtered_detections = [
            detection
            for detection in sample_model_detections
            if detection.confidence >= payload.confidence_threshold
        ]

        sample_status = "to_verify" if sample_filtered_detections else "no_detection"

        if sample_filtered_detections:
            filtered_detections.extend(sample_filtered_detections)
            try:
                save_detections_to_parquet(
                    sample_filtered_detections,
                    resolution_mode=payload.resolution_mode,
                    timestamp=sample_timestamp,
                )
            except Exception as exc:  # pragma: no cover - defensive logging for IO layer
                logger.warning("Could not persist detections to parquet: %s", exc)

        try:
            save_analysis_image_and_metadata(
                tile_image,
                analysis_id=analysis_id,
                lon=center_lon,
                lat=center_lat,
                resolution=payload.resolution_mode,
                status=sample_status,
                timestamp=sample_timestamp,
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

    return AnalysisRunResponse(
        analysis_id=analysis_id,
        source="mock",
        detections=filtered_detections,
        requested_samples=requested_samples,
        analyzed_samples=analyzed_samples,
        skipped_samples=max(0, requested_samples - analyzed_samples),
    )


@app.post("/analysis/reanalyze", response_model=AnalysisReanalysisResponse)
def reanalyze_images(payload: AnalysisReanalysisRequest) -> AnalysisReanalysisResponse:
    normalized_image_ids: list[str] = []
    seen_image_ids: set[str] = set()
    for raw_image_id in payload.image_ids:
        image_id = str(raw_image_id).strip()
        if not image_id or image_id in seen_image_ids:
            continue

        seen_image_ids.add(image_id)
        normalized_image_ids.append(image_id)

    if not normalized_image_ids:
        raise HTTPException(status_code=422, detail="At least one image ID is required")

    analysis_id = str(uuid4())
    image_rows = query_analysis_images()
    image_lookup = {
        str(item.get("image_id") or "").strip(): item
        for item in image_rows
        if str(item.get("image_id") or "").strip()
    }

    detections: list[Detection] = []
    missing_image_ids: list[str] = []
    failed_image_ids: list[str] = []
    reanalyzed_count = 0

    for image_id in normalized_image_ids:
        image_row = image_lookup.get(image_id)
        if image_row is None:
            missing_image_ids.append(image_id)
            continue

        image_path_raw = str(image_row.get("path") or "").strip()
        if not image_path_raw:
            failed_image_ids.append(image_id)
            continue

        image_path = Path(image_path_raw)
        if not image_path.exists() or not image_path.is_file():
            failed_image_ids.append(image_id)
            continue

        try:
            with Image.open(image_path) as opened_image:
                analysis_image = opened_image.convert("RGB")
        except OSError:
            failed_image_ids.append(image_id)
            continue

        sample_timestamp = datetime.now(timezone.utc).isoformat()
        model_detections = run_inference(
            image=analysis_image,
            confidence_threshold=payload.settings.confidence_threshold,
            image_size=max(analysis_image.size),
            model_name=payload.model_name,
        )

        filtered_sample_detections: list[Detection] = []
        for detection in model_detections:
            parsed_detection = Detection(
                detection_id=detection["detection_id"],
                analysis_id=analysis_id,
                confidence=detection["confidence"],
                **{"class": detection["class"]},
                class_id=detection["class_id"],
                bbox=BBox(**detection["bbox"]),
            )
            if parsed_detection.confidence >= payload.settings.confidence_threshold:
                filtered_sample_detections.append(parsed_detection)

        if filtered_sample_detections:
            detections.extend(filtered_sample_detections)
            try:
                save_detections_to_parquet(
                    filtered_sample_detections,
                    resolution_mode=payload.settings.resolution_mode,
                    timestamp=sample_timestamp,
                )
            except Exception as exc:  # pragma: no cover - defensive logging for IO layer
                logger.warning("Could not persist reanalysis detections to parquet: %s", exc)

        parsed_lat_lon = _extract_lat_lon_from_path(image_path_raw)
        fallback_lat = float(image_row.get("lat")) if image_row.get("lat") is not None else 0.0
        fallback_lon = float(image_row.get("lon")) if image_row.get("lon") is not None else 0.0
        center_lat = parsed_lat_lon[0] if parsed_lat_lon is not None else fallback_lat
        center_lon = parsed_lat_lon[1] if parsed_lat_lon is not None else fallback_lon

        try:
            save_analysis_image_and_metadata(
                analysis_image,
                analysis_id=analysis_id,
                lon=center_lon,
                lat=center_lat,
                resolution=payload.settings.resolution_mode,
                status="to_verify",
                timestamp=sample_timestamp,
            )
        except Exception as exc:  # pragma: no cover - defensive logging for IO layer
            logger.warning("Could not persist reanalysis image metadata: %s", exc)

        reanalyzed_count += 1

    return AnalysisReanalysisResponse(
        analysis_id=analysis_id,
        source="mock",
        requested_count=len(normalized_image_ids),
        reanalyzed_count=reanalyzed_count,
        missing_image_ids=missing_image_ids,
        failed_image_ids=failed_image_ids,
        detections=detections,
    )


@app.post("/analysis/local-run", response_model=AnalysisRunResponse)
def run_local_validation_analysis(payload: LocalValidationRunRequest) -> AnalysisRunResponse:
    analysis_id = str(uuid4())

    validation_dir = _get_validation_image_dir()
    validation_images = _get_validation_image_paths(validation_dir)
    if not validation_images:
        raise HTTPException(
            status_code=404,
            detail=f"No validation images found in {validation_dir}",
        )

    filtered_detections: list[Detection] = []
    for image_path in validation_images:
        sample_timestamp = datetime.now(timezone.utc).isoformat()
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
            model_name=payload.model_name,
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
                    class_id=detection["class_id"],
                    bbox=BBox(**display_bbox),
                )
            )

        sample_filtered_detections = [
            detection
            for detection in sample_model_detections
            if detection.confidence >= payload.confidence_threshold
        ]
        filtered_detections.extend(sample_filtered_detections)

        if sample_filtered_detections:
            try:
                save_detections_to_parquet(
                    sample_filtered_detections,
                    resolution_mode="detail",
                    timestamp=sample_timestamp,
                )
            except Exception as exc:  # pragma: no cover - defensive logging for IO layer
                logger.warning(
                    "Could not persist local validation detections to parquet for %s: %s",
                    image_path,
                    exc,
                )

        sample_status = "to_verify" if sample_filtered_detections else "no_detection"
        try:
            save_analysis_image_and_metadata(
                local_image,
                analysis_id=analysis_id,
                # Local validation images do not have geo coordinates.
                lon=0.0,
                lat=0.0,
                resolution="detail",
                status=sample_status,
                timestamp=sample_timestamp,
            )
        except Exception as exc:  # pragma: no cover - defensive logging for IO layer
            logger.warning(
                "Could not persist local validation analysis image metadata for %s: %s",
                image_path,
                exc,
            )

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


@app.patch("/detections/bulk/validate", response_model=DetectionBulkValidateResponse)
def validate_detections_bulk_endpoint(
    payload: DetectionBulkValidateRequest,
) -> DetectionBulkValidateResponse:
    summary = validate_detections_bulk(
        detection_ids=payload.detection_ids,
        target_status=payload.target_status,
    )

    for detection_id in summary.get("updated_detection_ids", []):
        upsert_detection_status(
            detection_id=str(detection_id),
            status=payload.target_status,
        )

    return DetectionBulkValidateResponse(
        requested_count=int(summary.get("requested_count", 0)),
        updated_count=int(summary.get("updated_count", 0)),
        updated_detection_ids=[str(d) for d in summary.get("updated_detection_ids", [])],
        missing_detection_ids=[str(d) for d in summary.get("missing_detection_ids", [])],
        target_status=payload.target_status,
        files_moved=int(summary.get("files_moved", 0)),
        files_missing=int(summary.get("files_missing", 0)),
        files_in_use=int(summary.get("files_in_use", 0)),
    )


@app.patch("/detections/bulk/tags", response_model=DetectionBulkTagsUpdateResponse)
def update_detections_bulk_tags(
    payload: DetectionBulkTagsUpdateRequest,
) -> DetectionBulkTagsUpdateResponse:
    normalized_tag = str(payload.tag).strip()
    if not normalized_tag:
        raise HTTPException(status_code=422, detail="Tag cannot be empty")

    normalized_detection_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_detection_id in payload.detection_ids:
        detection_id = str(raw_detection_id).strip()
        if not detection_id or detection_id in seen_ids:
            continue

        seen_ids.add(detection_id)
        normalized_detection_ids.append(detection_id)

    if not normalized_detection_ids:
        raise HTTPException(status_code=422, detail="At least one detection ID is required")

    existing_detection_ids = get_existing_detection_ids(normalized_detection_ids)
    updated_detection_ids = [
        detection_id
        for detection_id in normalized_detection_ids
        if detection_id in existing_detection_ids
    ]
    missing_detection_ids = [
        detection_id
        for detection_id in normalized_detection_ids
        if detection_id not in existing_detection_ids
    ]

    if updated_detection_ids:
        upsert_detection_tags_bulk(detection_ids=updated_detection_ids, tag=normalized_tag)

    return DetectionBulkTagsUpdateResponse(
        requested_count=len(normalized_detection_ids),
        updated_count=len(updated_detection_ids),
        updated_detection_ids=updated_detection_ids,
        missing_detection_ids=missing_detection_ids,
        tag=normalized_tag,
    )


@app.patch("/detections/{id}/tags", response_model=DetectionTagsUpdateResponse)
def update_detection_tags(
    id: str, payload: DetectionTagsUpdateRequest
) -> DetectionTagsUpdateResponse:
    upsert_detection_tags(detection_id=id, tags=payload.tags)
    normalized_tags = list(dict.fromkeys([str(tag).strip() for tag in payload.tags if str(tag).strip()]))
    return DetectionTagsUpdateResponse(detection_id=id, tags=normalized_tags)


@app.patch("/analysis-images/bulk/tags", response_model=AnalysisImageBulkTagsUpdateResponse)
def update_analysis_images_bulk_tags(
    payload: AnalysisImageBulkTagsUpdateRequest,
) -> AnalysisImageBulkTagsUpdateResponse:
    normalized_tag = str(payload.tag).strip()
    if not normalized_tag:
        raise HTTPException(status_code=422, detail="Tag cannot be empty")

    normalized_image_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_image_id in payload.image_ids:
        image_id = str(raw_image_id).strip()
        if not image_id or image_id in seen_ids:
            continue

        seen_ids.add(image_id)
        normalized_image_ids.append(image_id)

    if not normalized_image_ids:
        raise HTTPException(status_code=422, detail="At least one image ID is required")

    existing_image_ids = get_existing_analysis_image_ids(
        normalized_image_ids,
        status="no_detections",
    )
    updated_image_ids = [
        image_id for image_id in normalized_image_ids if image_id in existing_image_ids
    ]
    missing_image_ids = [
        image_id for image_id in normalized_image_ids if image_id not in existing_image_ids
    ]

    if updated_image_ids:
        upsert_detection_tags_bulk(detection_ids=updated_image_ids, tag=normalized_tag)

    return AnalysisImageBulkTagsUpdateResponse(
        requested_count=len(normalized_image_ids),
        updated_count=len(updated_image_ids),
        updated_image_ids=updated_image_ids,
        missing_image_ids=missing_image_ids,
        tag=normalized_tag,
    )


@app.delete("/detections/bulk", response_model=DetectionBulkDeleteResponse)
def delete_detections_bulk(payload: DetectionBulkDeleteRequest) -> DetectionBulkDeleteResponse:
    delete_summary = delete_detections_bulk_and_related_assets(
        detection_ids=payload.detection_ids,
        delete_images=payload.delete_images,
    )

    deleted_detection_ids = [
        str(detection_id)
        for detection_id in delete_summary.get("deleted_detection_ids", [])
    ]

    for detection_id in deleted_detection_ids:
        delete_detection_status(detection_id=detection_id)
        delete_detection_comment(detection_id=detection_id)
        delete_detection_tags(detection_id=detection_id)

    return DetectionBulkDeleteResponse(
        requested_count=int(delete_summary.get("requested_count", 0)),
        deleted_count=int(delete_summary.get("deleted_count", 0)),
        deleted_detection_ids=deleted_detection_ids,
        missing_detection_ids=[
            str(detection_id)
            for detection_id in delete_summary.get("missing_detection_ids", [])
        ],
        related_image_missing=bool(delete_summary.get("related_image_missing")),
        related_image_in_use=bool(delete_summary.get("related_image_in_use")),
        related_image_missing_count=int(delete_summary.get("related_image_missing_count", 0)),
        related_image_in_use_count=int(delete_summary.get("related_image_in_use_count", 0)),
    )


@app.delete("/detections/{id}", response_model=DetectionDeleteResponse)
def delete_detection(
    id: str,
    delete_images: Annotated[bool, Query(alias="deleteImages")] = False,
) -> DetectionDeleteResponse:
    deleted_payload = delete_detection_and_related_assets(
        detection_id=id,
        delete_images=delete_images,
    )
    if not bool(deleted_payload.get("detection_deleted")):
        raise HTTPException(status_code=404, detail="Detection not found")

    delete_detection_status(detection_id=id)
    delete_detection_comment(detection_id=id)
    delete_detection_tags(detection_id=id)

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
        related_image_missing=bool(deleted_payload.get("related_image_missing")),
        related_image_in_use=bool(deleted_payload.get("related_image_in_use")),
    )


@app.delete("/analysis-images/bulk", response_model=AnalysisImageBulkDeleteResponse)
def delete_analysis_images_bulk(
    payload: AnalysisImageBulkDeleteRequest,
) -> AnalysisImageBulkDeleteResponse:
    delete_summary = delete_analysis_images_by_ids(
        payload.image_ids,
        delete_files=payload.delete_files,
    )

    return AnalysisImageBulkDeleteResponse(
        requested_count=int(delete_summary.get("requested_count", 0)),
        deleted_count=int(delete_summary.get("deleted_count", 0)),
        deleted_image_ids=[
            str(image_id) for image_id in delete_summary.get("deleted_image_ids", [])
        ],
        missing_image_ids=[
            str(image_id) for image_id in delete_summary.get("missing_image_ids", [])
        ],
    )


@app.delete("/analysis-images/{image_id}", response_model=AnalysisImageDeleteResponse)
def delete_analysis_image(
    image_id: str,
    delete_files: Annotated[bool, Query(alias="deleteFiles")] = False,
) -> AnalysisImageDeleteResponse:
    image_deleted = delete_analysis_image_by_id(
        image_id=image_id,
        delete_files=delete_files,
    )
    if not image_deleted:
        raise HTTPException(status_code=404, detail="Analysis image not found")

    return AnalysisImageDeleteResponse(image_id=image_id, image_deleted=True)


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
        sort_by=params.sort_by,
        sort_order=params.sort_order,
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
