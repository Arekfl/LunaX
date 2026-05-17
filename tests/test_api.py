import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok_status_and_json_structure() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_run_returns_mock_detections_with_expected_json_structure() -> None:
    response = client.post("/analysis/run", json={})

    assert response.status_code == 200

    payload = response.json()
    assert set(payload.keys()) == {"analysis_id", "source", "detections"}
    assert isinstance(payload["analysis_id"], str)
    assert payload["analysis_id"]
    assert payload["source"] == "mock"

    detections = payload["detections"]
    assert isinstance(detections, list)
    assert 2 <= len(detections) <= 3

    for detection in detections:
        assert set(detection.keys()) == {
            "detection_id",
            "analysis_id",
            "bbox",
            "confidence",
            "class",
        }
        assert isinstance(detection["detection_id"], str)
        assert detection["detection_id"]
        assert detection["analysis_id"] == payload["analysis_id"]
        assert isinstance(detection["confidence"], (int, float))
        assert 0 <= detection["confidence"] <= 1
        assert isinstance(detection["class"], str)
        assert detection["class"]

        bbox = detection["bbox"]
        assert set(bbox.keys()) == {"x", "y", "width", "height"}
        assert bbox["x"] >= 0
        assert bbox["y"] >= 0
        assert bbox["width"] > 0
        assert bbox["height"] > 0


def test_patch_detection_status_persists_status_by_detection_id(tmp_path, monkeypatch) -> None:
    status_file = tmp_path / "detection_statuses.json"
    monkeypatch.setenv("DETECTION_STATUS_FILE", str(status_file))

    response = client.patch("/detections/det-42/status", json={"status": "confirmed"})

    assert response.status_code == 200
    assert response.json() == {
        "detection_id": "det-42",
        "status": "confirmed",
    }

    with status_file.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    assert payload["det-42"] == "confirmed"


def test_get_detection_statuses_returns_status_mapping(tmp_path, monkeypatch) -> None:
    status_file = tmp_path / "detection_statuses.json"
    monkeypatch.setenv("DETECTION_STATUS_FILE", str(status_file))

    client.patch("/detections/det-1/status", json={"status": "to_verify"})
    client.patch("/detections/det-2/status", json={"status": "rejected"})

    response = client.get("/detections/statuses")

    assert response.status_code == 200
    assert response.json() == {
        "det-1": "to_verify",
        "det-2": "rejected",
    }


def test_get_detections_query_filters_with_query_params(tmp_path, monkeypatch) -> None:
    parquet_file = tmp_path / "detections.parquet"
    status_file = tmp_path / "detection_statuses.json"
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(parquet_file))
    monkeypatch.setenv("DETECTION_STATUS_FILE", str(status_file))

    run_response = client.post("/analysis/run", json={"confidence_threshold": 0.0})
    assert run_response.status_code == 200

    patch_response = client.patch(
        "/detections/det-2/status", json={"status": "rejected"}
    )
    assert patch_response.status_code == 200

    response = client.get(
        "/detections/query",
        params={
            "status": "rejected",
            "class": "cave_candidate",
            "confidence": 0.7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["detection_id"] == "det-2"
    assert payload[0]["status"] == "rejected"
