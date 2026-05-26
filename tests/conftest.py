import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(autouse=True)
def isolate_data_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"

    monkeypatch.setenv("DETECTIONS_PARQUET_FILE", str(data_dir / "detections.parquet"))
    monkeypatch.setenv("NO_DETECTIONS_PARQUET_FILE", str(data_dir / "no_detections.parquet"))
    monkeypatch.setenv("NO_DETECTIONS_IMAGE_DIR", str(images_dir / "no_detections"))

    monkeypatch.setenv("DETECTION_STATUS_FILE", str(data_dir / "detection_statuses.json"))
    monkeypatch.setenv("DETECTION_COMMENT_FILE", str(data_dir / "detection_comments.json"))
    monkeypatch.setenv("DETECTION_TAG_FILE", str(data_dir / "detection_tags.json"))
