from uuid import uuid4

from fastapi import FastAPI

from app.schemas import AnalysisRunRequest, AnalysisRunResponse, BBox, Detection, HealthResponse

app = FastAPI(title="LunaX API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/analysis/run", response_model=AnalysisRunResponse)
def run_analysis(payload: AnalysisRunRequest) -> AnalysisRunResponse:
    mock_detections = [
        Detection(
            id="det-1",
            label="cave_candidate",
            score=0.93,
            bbox=BBox(x_min=124.5, y_min=210.0, x_max=178.2, y_max=263.7),
        ),
        Detection(
            id="det-2",
            label="cave_candidate",
            score=0.78,
            bbox=BBox(x_min=342.1, y_min=115.4, x_max=391.6, y_max=164.0),
        ),
        Detection(
            id="det-3",
            label="cave_candidate",
            score=0.56,
            bbox=BBox(x_min=58.0, y_min=302.3, x_max=102.4, y_max=346.8),
        ),
    ]

    filtered_detections = [
        detection
        for detection in mock_detections
        if detection.score >= payload.confidence_threshold
    ]

    return AnalysisRunResponse(
        analysis_id=str(uuid4()),
        source="mock",
        detections=filtered_detections,
    )
