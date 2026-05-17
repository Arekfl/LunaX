from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalysisRunRequest, AnalysisRunResponse, BBox, Detection, HealthResponse

app = FastAPI(title="LunaX API", version="0.1.0")

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

    mock_detections = [
        Detection(
            detection_id="det-1",
            analysis_id=analysis_id,
            confidence=0.93,
            class_name="cave_candidate",
            bbox=BBox(x=124.5, y=210.0, width=53.7, height=53.7),
        ),
        Detection(
            detection_id="det-2",
            analysis_id=analysis_id,
            confidence=0.78,
            class_name="cave_candidate",
            bbox=BBox(x=342.1, y=115.4, width=49.5, height=48.6),
        ),
        Detection(
            detection_id="det-3",
            analysis_id=analysis_id,
            confidence=0.56,
            class_name="cave_candidate",
            bbox=BBox(x=58.0, y=302.3, width=44.4, height=44.5),
        ),
    ]

    filtered_detections = [
        detection
        for detection in mock_detections
        if detection.confidence >= payload.confidence_threshold
    ]

    return AnalysisRunResponse(
        analysis_id=analysis_id,
        source="mock",
        detections=filtered_detections,
    )
