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
