from app.analytics import query_detections, save_detections_to_parquet
from app.schemas import BBox, Detection
from app.storage import upsert_detection_comment, upsert_detection_status
import pandas as pd


def _sample_detections() -> list[Detection]:
    return [
        Detection(
            detection_id="det-a",
            analysis_id="analysis-1",
            confidence=0.92,
            **{"class": "cave_candidate"},
            bbox=BBox(x=10, y=20, width=30, height=40),
        ),
        Detection(
            detection_id="det-b",
            analysis_id="analysis-1",
            confidence=0.61,
            **{"class": "crater"},
            bbox=BBox(x=50, y=60, width=35, height=45),
        ),
        Detection(
            detection_id="det-c",
            analysis_id="analysis-2",
            confidence=0.77,
            **{"class": "cave_candidate"},
            bbox=BBox(x=70, y=80, width=20, height=30),
        ),
    ]


def test_query_detections_filters_by_status_class_and_confidence(tmp_path, monkeypatch) -> None:
    parquet_file = tmp_path / "detections.parquet"
    status_file = tmp_path / "detection_statuses.json"
    comment_file = tmp_path / "detection_comments.json"
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(parquet_file))
    monkeypatch.setenv("DETECTION_STATUS_FILE", str(status_file))
    monkeypatch.setenv("DETECTION_COMMENT_FILE", str(comment_file))

    save_detections_to_parquet(_sample_detections())

    all_rows = query_detections()
    assert len(all_rows) == 3
    assert all(row["resolution"] == "detail" for row in all_rows)

    cave_rows = query_detections(class_name="cave_candidate")
    assert len(cave_rows) == 2
    assert all(row["class"] == "cave_candidate" for row in cave_rows)

    high_conf_rows = query_detections(min_confidence=0.8)
    assert len(high_conf_rows) == 1
    assert high_conf_rows[0]["detection_id"] == "det-a"

    upsert_detection_status("det-b", "rejected")
    upsert_detection_comment("det-b", "Review required")
    rejected_rows = query_detections(status="rejected")
    assert len(rejected_rows) == 1
    assert rejected_rows[0]["detection_id"] == "det-b"
    assert rejected_rows[0]["status"] == "rejected"
    assert rejected_rows[0]["comment"] == "Review required"


def test_save_detections_to_parquet_persists_requested_fields(tmp_path, monkeypatch) -> None:
    parquet_file = tmp_path / "detections.parquet"
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(parquet_file))

    save_detections_to_parquet(
        _sample_detections()[:1],
        resolution_mode="ultra",
        timestamp="2026-05-24T12:00:00+00:00",
    )

    frame = pd.read_parquet(parquet_file)
    row = frame.iloc[0]

    assert row["detection_id"] == "det-a"
    assert isinstance(row["bbox"], dict)
    assert set(row["bbox"].keys()) == {"x", "y", "width", "height"}
    assert row["confidence"] == 0.92
    assert row["class"] == "cave_candidate"
    assert row["resolution"] == "ultra"
    assert row["resolutionMode"] == "ultra"
    assert row["timestamp"] == "2026-05-24T12:00:00+00:00"
