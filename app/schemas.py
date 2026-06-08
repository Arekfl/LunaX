from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelName = Literal["best.pt", "best_kratery.pt"]


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
    class_id: int = Field(default=0)


AnalysisResolutionMode = Literal["preview", "detail", "ultra"]


class AnalysisRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    region_id: str | None = None
    resolution_mode: AnalysisResolutionMode = Field(default="detail", alias="resolutionMode")
    num_samples: int = Field(default=1, ge=1, le=20, alias="numSamples")
    confidence_threshold: float = Field(0.5, ge=0, le=1, alias="confidenceThreshold")
    model_name: ModelName = Field(default="best.pt", alias="modelName")
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
    model_name: ModelName = Field(default="best.pt", alias="modelName")


DetectionStatus = Literal["confirmed", "to_verify", "rejected"]
DetectionSortBy = Literal["confidence", "data"]
DetectionSortOrder = Literal["asc", "desc"]


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


class DetectionTagsUpdateRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


class DetectionTagsUpdateResponse(BaseModel):
    detection_id: str
    tags: list[str]


class DetectionDeleteResponse(BaseModel):
    detection_id: str
    detection_deleted: bool
    deleted_image_id: str | None = None
    deleted_image_path: str | None = None
    related_image_missing: bool = False
    related_image_in_use: bool = False


class DetectionBulkDeleteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detection_ids: list[str] = Field(..., min_length=1, alias="detectionIds")
    delete_images: bool = Field(default=False, alias="deleteImages")


class DetectionBulkDeleteResponse(BaseModel):
    requested_count: int
    deleted_count: int
    deleted_detection_ids: list[str]
    missing_detection_ids: list[str]
    related_image_missing: bool = False
    related_image_in_use: bool = False
    related_image_missing_count: int = 0
    related_image_in_use_count: int = 0


class DetectionBulkTagsUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detection_ids: list[str] = Field(..., min_length=1, alias="detectionIds")
    tag: str = Field(..., min_length=1)


class DetectionBulkTagsUpdateResponse(BaseModel):
    requested_count: int
    updated_count: int
    updated_detection_ids: list[str]
    missing_detection_ids: list[str]
    tag: str


class AnalysisImageDeleteResponse(BaseModel):
    image_id: str
    image_deleted: bool


class AnalysisImageBulkDeleteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_ids: list[str] = Field(..., min_length=1, alias="imageIds")
    delete_files: bool = Field(default=False, alias="deleteFiles")


class AnalysisImageBulkDeleteResponse(BaseModel):
    requested_count: int
    deleted_count: int
    deleted_image_ids: list[str]
    missing_image_ids: list[str]


class AnalysisImageBulkTagsUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_ids: list[str] = Field(..., min_length=1, alias="imageIds")
    tag: str = Field(..., min_length=1)


class AnalysisImageBulkTagsUpdateResponse(BaseModel):
    requested_count: int
    updated_count: int
    updated_image_ids: list[str]
    missing_image_ids: list[str]
    tag: str


class DetectionsQueryParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: DetectionStatus | None = None
    class_name: str | None = Field(default=None, alias="class")
    confidence: float | None = Field(default=None, ge=0, le=1)
    resolution_mode: AnalysisResolutionMode | None = Field(
        default=None, alias="resolutionMode"
    )
    analysis_id: str | None = None
    sort_by: DetectionSortBy = Field(default="confidence", alias="sortBy")
    sort_order: DetectionSortOrder = Field(default="desc", alias="sortOrder")
