from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"]


class BBox(BaseModel):
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detection_id: str
    analysis_id: str
    bbox: BBox
    confidence: float = Field(..., ge=0, le=1)
    class_name: str = Field(alias="class")


AnalysisResolutionMode = Literal["preview", "detail", "ultra"]


class AnalysisRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    region_id: str | None = None
    resolution_mode: AnalysisResolutionMode = Field(default="detail", alias="resolutionMode")
    num_samples: int = Field(default=1, ge=1, le=20, alias="numSamples")
    confidence_threshold: float = Field(0.5, ge=0, le=1, alias="confidenceThreshold")
    bbox: list[float] = Field(
        default_factory=lambda: [-180.0, -90.0, 180.0, 90.0],
        min_length=4,
        max_length=4,
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox_order(cls, value: list[float]) -> list[float]:
        xmin, ymin, xmax, ymax = value

        if xmax <= xmin:
            raise ValueError("bbox must satisfy xmax > xmin")
        if ymax <= ymin:
            raise ValueError("bbox must satisfy ymax > ymin")

        return value


class AnalysisRunResponse(BaseModel):
    analysis_id: str
    source: Literal["mock"]
    detections: list[Detection]


class LocalValidationRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    confidence_threshold: float = Field(0.5, ge=0, le=1, alias="confidenceThreshold")


DetectionStatus = Literal["confirmed", "to_verify", "rejected"]


class DetectionStatusUpdateRequest(BaseModel):
    status: DetectionStatus


class DetectionStatusUpdateResponse(BaseModel):
    detection_id: str
    status: DetectionStatus


class DetectionCommentUpdateRequest(BaseModel):
    comment: str = Field(default="", max_length=2000)


class DetectionCommentUpdateResponse(BaseModel):
    detection_id: str
    comment: str


class DetectionDeleteResponse(BaseModel):
    detection_id: str
    detection_deleted: bool
    deleted_image_id: str | None = None
    deleted_image_path: str | None = None
    related_image_missing: bool = False


class DetectionsQueryParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: DetectionStatus | None = None
    class_name: str | None = Field(default=None, alias="class")
    confidence: float | None = Field(default=None, ge=0, le=1)
    resolution_mode: AnalysisResolutionMode | None = Field(
        default=None, alias="resolutionMode"
    )
    analysis_id: str | None = None
