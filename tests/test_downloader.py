from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

from data import downloader


def _png_bytes(size: int = 32, pixel_value: int = 128) -> bytes:
    image = Image.new("L", (size, size), color=pixel_value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_bbox_validation_accepts_valid_bbox() -> None:
    bbox = [10.0, 5.0, 11.0, 6.0]

    normalized = downloader._normalize_bbox(bbox)

    assert normalized == (10.0, 5.0, 11.0, 6.0)


def test_bbox_validation_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError, match="xmax > xmin and ymax > ymin"):
        downloader._normalize_bbox([10.0, 5.0, 9.0, 6.0])

    with pytest.raises(ValueError, match="4 values"):
        downloader._normalize_bbox([10.0, 5.0, 11.0])


def test_download_tile_returns_pil_image_with_mocked_wms(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked_response = Mock()
    mocked_response.headers = {"Content-Type": "image/png"}
    mocked_response.content = _png_bytes()
    mocked_response.raise_for_status = Mock()

    mocked_get = Mock(return_value=mocked_response)

    monkeypatch.setattr(downloader, "_get_selected_layer", lambda: "KaguyaTC_Ortho")
    monkeypatch.setattr(downloader.requests, "get", mocked_get)

    image = downloader.download_tile("detail", [10.0, 5.0, 11.0, 6.0])

    assert isinstance(image, Image.Image)
    assert image.mode == "L"

    assert mocked_get.call_count == 1
    sent_params = mocked_get.call_args.kwargs["params"]
    assert sent_params["request"] == "GetMap"
    assert sent_params["layers"] == "KaguyaTC_Ortho"
    assert sent_params["width"] == 1536
    assert sent_params["height"] == 1536


def test_download_tile_retries_after_first_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked_response = Mock()
    mocked_response.headers = {"Content-Type": "image/png"}
    mocked_response.content = _png_bytes()
    mocked_response.raise_for_status = Mock()

    mocked_get = Mock(
        side_effect=[
            downloader.requests.RequestException("temporary network error"),
            mocked_response,
        ]
    )

    monkeypatch.setattr(downloader, "_get_selected_layer", lambda: "KaguyaTC_Ortho")
    monkeypatch.setattr(downloader.requests, "get", mocked_get)
    monkeypatch.setattr(downloader.time, "sleep", lambda _seconds: None)

    image = downloader.download_tile("detail", [10.0, 5.0, 11.0, 6.0])

    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert mocked_get.call_count == 2
