from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class AnalysisRunRequest(BaseModel):
    region_id: str | None = None
    confidence_threshold: float = Field(0.5, ge=0, le=1)


class AnalysisRunResponse(BaseModel):
    analysis_id: str
    source: Literal["mock"]
    detections: list[Detection]


DetectionStatus = Literal["confirmed", "to_verify", "rejected"]


class DetectionStatusUpdateRequest(BaseModel):
    status: DetectionStatus


class DetectionStatusUpdateResponse(BaseModel):
    detection_id: str
    status: DetectionStatus


class DetectionsQueryParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: DetectionStatus | None = None
    class_name: str | None = Field(default=None, alias="class")
    confidence: float | None = Field(default=None, ge=0, le=1)
