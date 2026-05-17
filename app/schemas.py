from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class BBox(BaseModel):
    x_min: float = Field(..., ge=0)
    y_min: float = Field(..., ge=0)
    x_max: float = Field(..., ge=0)
    y_max: float = Field(..., ge=0)


class Detection(BaseModel):
    id: str
    label: str
    score: float = Field(..., ge=0, le=1)
    bbox: BBox


class AnalysisRunRequest(BaseModel):
    region_id: str | None = None
    confidence_threshold: float = Field(0.5, ge=0, le=1)


class AnalysisRunResponse(BaseModel):
    analysis_id: str
    source: Literal["mock"]
    detections: list[Detection]
