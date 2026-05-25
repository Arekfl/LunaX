import json
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pandas as pd
from PIL import Image

from app.main import app

client = TestClient(app)


def _mock_tile(*_args, **_kwargs):
    return Image.new("L", (64, 64), color=128)


def _mock_inference(*_args, **_kwargs):
    return [
        {
            "detection_id": "det-mock-1",
            "bbox": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
            "confidence": 0.93,
            "class": "cave_candidate",
        },
        {
            "detection_id": "det-mock-2",
            "bbox": {"x": 50.0, "y": 60.0, "width": 35.0, "height": 45.0},
            "confidence": 0.78,
            "class": "cave_candidate",
        },
        {
            "detection_id": "det-mock-3",
            "bbox": {"x": 70.0, "y": 80.0, "width": 20.0, "height": 30.0},
            "confidence": 0.56,
            "class": "crater",
        },
    ]


def test_health_returns_ok_status_and_json_structure() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_run_returns_mock_detections_with_expected_json_structure(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

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


def test_analysis_run_aggregates_results_from_num_samples(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 2,
            "confidenceThreshold": 0.0,
            "bbox": [-22.2, 4.1, -21.7, 4.6],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["detections"]) >= 6


def test_analysis_run_uses_distinct_sample_bboxes_for_multiple_samples(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)
    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 4,
            "confidenceThreshold": 1.0,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200
    assert mocked_download.call_count == 4

    bboxes = [call.args[1] for call in mocked_download.call_args_list]
    assert len({tuple(round(value, 8) for value in bbox) for bbox in bboxes}) == 4


def test_analysis_run_passes_confidence_threshold_to_inference(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    mocked_inference = Mock(return_value=[])
    monkeypatch.setattr("app.main.run_inference", mocked_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.1,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200
    mocked_inference.assert_called_once()
    _, kwargs = mocked_inference.call_args
    assert kwargs["confidence_threshold"] == 0.1


def test_analysis_run_uses_fixed_wms_configuration(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)
    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "preview",
            "numSamples": 1,
            "confidenceThreshold": 0.1,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200
    mocked_download.assert_called_once()
    args, kwargs = mocked_download.call_args
    assert args[0] == "preview"
    assert args[1] == [-10.0, -5.0, 10.0, 5.0]
    assert kwargs == {}


def test_analysis_run_accepts_legacy_wms_fields_without_effect(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "preview",
            "wmsSource": "invalid_source",
            "wmsLayer": "invalid_layer",
            "numSamples": 1,
            "confidenceThreshold": 0.1,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200


def test_analysis_run_returns_502_when_wms_download_fails(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", Mock(side_effect=RuntimeError("WMS response is not an image")))
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "preview",
            "numSamples": 1,
            "confidenceThreshold": 0.01,
            "bbox": [-53.027344, 15.688477, -52.792969, 15.864257],
        },
    )

    assert response.status_code == 502
    assert "No valid imagery" in response.json()["detail"]


def test_local_analysis_runs_on_validation_images(tmp_path, monkeypatch) -> None:
    validation_dir = tmp_path / "images" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 160), color=(128, 128, 128)).save(validation_dir / "sample_01.png")
    Image.new("RGB", (640, 320), color=(64, 64, 64)).save(validation_dir / "sample_02.jpg")

    detections_parquet_file = tmp_path / "detections.parquet"
    monkeypatch.setenv("VALIDATION_IMAGE_DIR", str(validation_dir))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    mocked_inference = Mock(
        return_value=[
            {
                "detection_id": "det-local-1",
                "bbox": {"x": 80.0, "y": 40.0, "width": 80.0, "height": 40.0},
                "confidence": 0.82,
                "class": "cave_candidate",
            }
        ]
    )
    monkeypatch.setattr("app.main.run_inference", mocked_inference)

    response = client.post(
        "/analysis/local-run",
        json={
            "confidenceThreshold": 0.1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_id"]
    assert payload["source"] == "mock"
    assert len(payload["detections"]) == 2

    assert mocked_inference.call_count == 2
    image_sizes: list[int] = []
    for call in mocked_inference.call_args_list:
        _, kwargs = call
        assert kwargs["confidence_threshold"] == 0.1
        image_sizes.append(int(kwargs["image_size"]))

    assert sorted(image_sizes) == [320, 640]

    for detection in payload["detections"]:
        bbox = detection["bbox"]
        assert 0 <= bbox["x"] <= 180
        assert 0 <= bbox["y"] <= 90
        assert bbox["width"] > 0
        assert bbox["height"] > 0
        assert bbox["x"] + bbox["width"] <= 180
        assert bbox["y"] + bbox["height"] <= 90

    stored = pd.read_parquet(detections_parquet_file)
    assert len(stored) == 2
    assert set(stored["analysis_id"]) == {payload["analysis_id"]}


def test_local_analysis_returns_404_when_validation_folder_has_no_images(tmp_path, monkeypatch) -> None:
    validation_dir = tmp_path / "images" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VALIDATION_IMAGE_DIR", str(validation_dir))

    response = client.post(
        "/analysis/local-run",
        json={
            "confidenceThreshold": 0.1,
        },
    )

    assert response.status_code == 404
    assert "No validation images found" in response.json()["detail"]


def test_analysis_run_generates_new_analysis_id_for_each_run(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    first_response = client.post("/analysis/run", json={"confidenceThreshold": 0.0})
    second_response = client.post("/analysis/run", json={"confidenceThreshold": 0.0})

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_payload = first_response.json()
    second_payload = second_response.json()

    assert first_payload["analysis_id"]
    assert second_payload["analysis_id"]
    assert first_payload["analysis_id"] != second_payload["analysis_id"]

    for detection in first_payload["detections"]:
        assert detection["analysis_id"] == first_payload["analysis_id"]

    for detection in second_payload["detections"]:
        assert detection["analysis_id"] == second_payload["analysis_id"]


def test_analysis_run_filters_out_detections_below_confidence_threshold(monkeypatch) -> None:
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.9,
            "bbox": [-22.2, 4.1, -21.7, 4.6],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    detections = payload["detections"]
    assert len(detections) == 1
    assert detections[0]["confidence"] >= 0.9


def test_analysis_run_converts_pixel_bbox_to_geo_before_download(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.0,
            "bbox": [512.0, 256.0, 1536.0, 768.0],
        },
    )

    assert response.status_code == 200
    mocked_download.assert_called_once()

    mode, geo_bbox = mocked_download.call_args.args
    assert mode == "detail"
    assert geo_bbox == [-90.0, -45.0, 90.0, 45.0]


def test_analysis_run_keeps_geo_bbox_before_download(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.0,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200
    mocked_download.assert_called_once()

    mode, geo_bbox = mocked_download.call_args.args
    assert mode == "detail"
    assert geo_bbox == [-10.0, -5.0, 10.0, 5.0]


def test_analysis_run_saves_no_detection_images_and_metadata_per_sample(
    tmp_path, monkeypatch
) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)
    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 3,
            "confidenceThreshold": 0.5,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detections"] == []

    saved_images = sorted(no_detections_image_dir.glob("*.png"))
    assert len(saved_images) == 1
    assert all("detail" in saved_image.name for saved_image in saved_images)
    assert all("lat-" in saved_image.name for saved_image in saved_images)
    assert all("lon-" in saved_image.name for saved_image in saved_images)

    metadata = pd.read_parquet(no_detections_parquet_file)
    assert len(metadata) == 3
    assert {
        "image_id",
        "analysis_id",
        "path",
        "status",
        "lat",
        "lon",
        "resolution",
        "timestamp",
        "content_hash",
    }.issubset(metadata.columns)
    assert set(metadata["status"]) == {"no_detections"}
    assert metadata["analysis_id"].nunique() == 1
    assert metadata["analysis_id"].iloc[0] == payload["analysis_id"]
    assert set(metadata["resolution"]) == {"detail"}
    assert metadata["lat"].between(-5.0, 5.0).all()
    assert metadata["lon"].between(-10.0, 10.0).all()
    assert metadata[["lat", "lon"]].drop_duplicates().shape[0] == 3
    assert all(Path(path_value).exists() for path_value in metadata["path"])
    assert metadata["content_hash"].nunique() == 1


def test_analysis_run_saves_images_and_metadata_per_sample_when_detections_exist(
    tmp_path, monkeypatch
) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    to_verify_image_dir = tmp_path / "images" / "to_verify"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)
    monkeypatch.setattr(
        "app.main.run_inference",
        Mock(
            return_value=[
                {
                    "detection_id": "det-saved-1",
                    "bbox": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
                    "confidence": 0.95,
                    "class": "cave_candidate",
                }
            ]
        ),
    )

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 3,
            "confidenceThreshold": 0.5,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["detections"]) == 3

    saved_images = sorted(to_verify_image_dir.glob("*.png"))
    assert len(saved_images) == 1
    assert len(sorted(no_detections_image_dir.glob("*.png"))) == 0

    metadata = pd.read_parquet(no_detections_parquet_file)
    assert len(metadata) == 3
    assert set(metadata["status"]) == {"to_verify"}
    assert set(metadata["analysis_id"]) == {payload["analysis_id"]}
    assert set(metadata["resolution"]) == {"detail"}
    assert metadata[["lat", "lon"]].drop_duplicates().shape[0] == 3
    assert all(Path(path_value).exists() for path_value in metadata["path"])


def test_get_no_detections_query_excludes_samples_with_detections(
    tmp_path, monkeypatch
) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    monkeypatch.setattr("app.main.download_tile", Mock(return_value=Image.new("L", (64, 64), color=128)))
    monkeypatch.setattr(
        "app.main.run_inference",
        Mock(
            return_value=[
                {
                    "detection_id": "det-filter-1",
                    "bbox": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
                    "confidence": 0.95,
                    "class": "cave_candidate",
                }
            ]
        ),
    )

    run_response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "preview",
            "numSamples": 2,
            "confidenceThreshold": 0.5,
            "bbox": [-20.0, -10.0, 20.0, 10.0],
        },
    )
    assert run_response.status_code == 200

    response = client.get("/no-detections/query")

    assert response.status_code == 200
    assert response.json() == []


def test_get_no_detections_query_returns_saved_images(tmp_path, monkeypatch) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    monkeypatch.setattr("app.main.download_tile", Mock(return_value=Image.new("L", (64, 64), color=128)))
    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))

    run_response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "preview",
            "numSamples": 2,
            "confidenceThreshold": 0.5,
            "bbox": [-20.0, -10.0, 20.0, 10.0],
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()

    response = client.get("/no-detections/query")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert {
        "image_id",
        "analysis_id",
        "path",
        "status",
        "lat",
        "lon",
        "resolution",
        "timestamp",
    }.issubset(payload[0].keys())
    assert all(item["status"] == "no_detections" for item in payload)
    assert all(item["analysis_id"] == run_payload["analysis_id"] for item in payload)
    assert all(item["resolution"] == "preview" for item in payload)
    assert all(Path(item["path"]).exists() for item in payload)


def test_no_detections_reuses_file_but_keeps_entries_per_sample(tmp_path, monkeypatch) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    monkeypatch.setattr("app.main.download_tile", Mock(return_value=Image.new("L", (64, 64), color=128)))
    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "ultra",
            "numSamples": 10,
            "confidenceThreshold": 0.5,
            "bbox": [-56.25, 28.125, -50.625, 30.9375],
        },
    )
    assert response.status_code == 200
    run_payload = response.json()

    files = sorted(no_detections_image_dir.glob("*.png"))
    assert len(files) == 1

    metadata = pd.read_parquet(no_detections_parquet_file)
    assert len(metadata) == 10
    assert metadata["content_hash"].nunique() == 1
    assert set(metadata["analysis_id"]) == {run_payload["analysis_id"]}

    list_response = client.get("/no-detections/query")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload) == 10
    assert all(item["analysis_id"] == run_payload["analysis_id"] for item in payload)


def test_get_no_detections_image_returns_png(tmp_path, monkeypatch) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    monkeypatch.setattr("app.main.download_tile", Mock(return_value=Image.new("L", (64, 64), color=128)))
    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))

    run_response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.5,
            "bbox": [-20.0, -10.0, 20.0, 10.0],
        },
    )
    assert run_response.status_code == 200

    list_response = client.get("/no-detections/query")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload) == 1

    image_id = payload[0]["image_id"]
    image_response = client.get(f"/no-detections/image/{image_id}")

    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")
    assert len(image_response.content) > 0


def test_get_analysis_images_query_returns_all_saved_images(tmp_path, monkeypatch) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    monkeypatch.setattr("app.main.download_tile", Mock(return_value=Image.new("L", (64, 64), color=128)))

    monkeypatch.setattr("app.main.run_inference", Mock(return_value=[]))
    no_detections_response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "preview",
            "numSamples": 2,
            "confidenceThreshold": 0.5,
            "bbox": [-20.0, -10.0, 20.0, 10.0],
        },
    )
    assert no_detections_response.status_code == 200
    no_detections_analysis_id = no_detections_response.json()["analysis_id"]

    monkeypatch.setattr(
        "app.main.run_inference",
        Mock(
            return_value=[
                {
                    "detection_id": "det-all-1",
                    "bbox": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
                    "confidence": 0.95,
                    "class": "cave_candidate",
                }
            ]
        ),
    )
    detections_response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 3,
            "confidenceThreshold": 0.5,
            "bbox": [-10.0, -5.0, 10.0, 5.0],
        },
    )
    assert detections_response.status_code == 200
    detections_analysis_id = detections_response.json()["analysis_id"]

    response = client.get("/analysis-images/query")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 5
    assert {
        "image_id",
        "analysis_id",
        "path",
        "status",
        "lat",
        "lon",
        "resolution",
        "timestamp",
    }.issubset(payload[0].keys())
    assert {item["status"] for item in payload} == {"to_verify", "no_detections"}
    assert {item["analysis_id"] for item in payload} == {
        no_detections_analysis_id,
        detections_analysis_id,
    }
    assert all(Path(item["path"]).exists() for item in payload)


def test_get_analysis_image_returns_png_for_detection_sample(tmp_path, monkeypatch) -> None:
    no_detections_image_dir = tmp_path / "images" / "no_detections"
    no_detections_parquet_file = tmp_path / "no_detections.parquet"
    detections_parquet_file = tmp_path / "detections.parquet"

    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(no_detections_image_dir))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(no_detections_parquet_file))
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(detections_parquet_file))

    monkeypatch.setattr("app.main.download_tile", Mock(return_value=Image.new("L", (64, 64), color=128)))
    monkeypatch.setattr(
        "app.main.run_inference",
        Mock(
            return_value=[
                {
                    "detection_id": "det-image-1",
                    "bbox": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
                    "confidence": 0.95,
                    "class": "cave_candidate",
                }
            ]
        ),
    )

    run_response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.5,
            "bbox": [-20.0, -10.0, 20.0, 10.0],
        },
    )
    assert run_response.status_code == 200

    list_response = client.get("/analysis-images/query")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload) == 1
    assert payload[0]["status"] == "to_verify"

    image_id = payload[0]["image_id"]
    image_response = client.get(f"/analysis-images/image/{image_id}")

    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")
    assert len(image_response.content) > 0


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


def test_patch_detection_comment_persists_comment_by_detection_id(tmp_path, monkeypatch) -> None:
    comment_file = tmp_path / "detection_comments.json"
    monkeypatch.setenv("DETECTION_COMMENT_FILE", str(comment_file))

    response = client.patch(
        "/detections/det-42/comment", json={"comment": "To verify manually"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "detection_id": "det-42",
        "comment": "To verify manually",
    }

    with comment_file.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    assert payload["det-42"] == "To verify manually"


def test_get_detections_query_filters_with_query_params(tmp_path, monkeypatch) -> None:
    parquet_file = tmp_path / "detections.parquet"
    status_file = tmp_path / "detection_statuses.json"
    comment_file = tmp_path / "detection_comments.json"
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(parquet_file))
    monkeypatch.setenv("DETECTION_STATUS_FILE", str(status_file))
    monkeypatch.setenv("DETECTION_COMMENT_FILE", str(comment_file))
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    first_run_response = client.post(
        "/analysis/run",
        json={"confidenceThreshold": 0.0, "resolutionMode": "detail"},
    )
    second_run_response = client.post(
        "/analysis/run",
        json={"confidenceThreshold": 0.0, "resolutionMode": "preview"},
    )
    assert first_run_response.status_code == 200
    assert second_run_response.status_code == 200

    run_payload = first_run_response.json()
    first_analysis_id = run_payload["analysis_id"]
    target_detection = next(
        detection
        for detection in run_payload["detections"]
        if detection["class"] == "cave_candidate" and detection["confidence"] < 0.9
    )
    target_detection_id = target_detection["detection_id"]

    patch_response = client.patch(
        f"/detections/{target_detection_id}/status", json={"status": "rejected"}
    )
    assert patch_response.status_code == 200

    comment_response = client.patch(
        f"/detections/{target_detection_id}/comment",
        json={"comment": "Potential false positive"},
    )
    assert comment_response.status_code == 200

    response = client.get(
        "/detections/query",
        params={
            "status": "rejected",
            "class": "cave_candidate",
            "confidence": 0.7,
            "resolutionMode": "detail",
            "analysis_id": first_analysis_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["detection_id"] == target_detection_id
    assert payload[0]["analysis_id"] == first_analysis_id
    assert payload[0]["status"] == "rejected"
    assert payload[0]["comment"] == "Potential false positive"


def test_get_detections_query_returns_rows_without_filters(tmp_path, monkeypatch) -> None:
    parquet_file = tmp_path / "detections.parquet"
    status_file = tmp_path / "detection_statuses.json"
    comment_file = tmp_path / "detection_comments.json"
    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(parquet_file))
    monkeypatch.setenv("DETECTION_STATUS_FILE", str(status_file))
    monkeypatch.setenv("DETECTION_COMMENT_FILE", str(comment_file))
    monkeypatch.setattr("app.main.download_tile", _mock_tile)
    monkeypatch.setattr("app.main.run_inference", _mock_inference)

    run_response = client.post("/analysis/run", json={"confidenceThreshold": 0.0})
    assert run_response.status_code == 200

    response = client.get("/detections/query")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) >= 2
    assert {
        "detection_id",
        "analysis_id",
        "class",
        "confidence",
        "bbox",
        "status",
        "comment",
    }.issubset(payload[0].keys())


def test_patch_detection_status_rejects_invalid_status_value() -> None:
    response = client.patch("/detections/det-42/status", json={"status": "invalid"})

    assert response.status_code == 422


def test_get_detections_query_rejects_invalid_confidence_param() -> None:
    response = client.get("/detections/query", params={"confidence": 1.5})

    assert response.status_code == 422


def test_analysis_run_rejects_bbox_with_invalid_length(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.5,
            "bbox": [-22.2, 4.1, -21.7],
        },
    )

    assert response.status_code == 422
    assert "bbox" in str(response.json()["detail"]).lower()
    mocked_download.assert_not_called()


def test_analysis_run_rejects_bbox_when_xmax_not_greater_than_xmin(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.5,
            "bbox": [-22.2, 4.1, -22.2, 4.6],
        },
    )

    assert response.status_code == 422
    assert "xmax > xmin" in str(response.json()["detail"])
    mocked_download.assert_not_called()


def test_analysis_run_rejects_bbox_when_ymax_not_greater_than_ymin(monkeypatch) -> None:
    mocked_download = Mock(return_value=Image.new("L", (64, 64), color=128))
    monkeypatch.setattr("app.main.download_tile", mocked_download)

    response = client.post(
        "/analysis/run",
        json={
            "resolutionMode": "detail",
            "numSamples": 1,
            "confidenceThreshold": 0.5,
            "bbox": [-22.2, 4.6, -21.7, 4.6],
        },
    )

    assert response.status_code == 422
    assert "ymax > ymin" in str(response.json()["detail"])
    mocked_download.assert_not_called()
