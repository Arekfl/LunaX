from __future__ import annotations

import math
import time
from io import BytesIO
from typing import Any, Sequence, cast

import requests
from PIL import Image

MOON_RADIUS_M = 1737400.0
METERS_PER_DEG = (2.0 * math.pi * MOON_RADIUS_M) / 360.0

WMS_VERSION = "1.1.1"
WMS_SRS = "EPSG:4326"
WMS_URL = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
WMS_MAP_PATH = "/maps/earth/moon_simp_cyl.map"
WMS_LAYER_NAME = "KaguyaTC_Ortho"
REQUEST_TIMEOUT = 25

MAX_RETRIES_PER_TILE = 6
RETRY_SLEEP_SECONDS = 0.5

STRIPE_FILTER_ENABLED = True
STRIPE_WHITE_THRESHOLD = 245
STRIPE_COLUMN_RATIO_THRESHOLD = 0.25
STRIPE_MIN_RUN_WIDTH = 8

UserBBox = tuple[float, float, float, float]
RequestBBox = tuple[float, float, float, float]


def _delta_from_target_mpp(target_mpp: float, image_size: int) -> float:
    return (target_mpp * image_size) / METERS_PER_DEG


def _expected_mpp_equator(delta_deg: float, image_size: int) -> float:
    return (delta_deg * METERS_PER_DEG) / image_size


ULTRA_TARGET_MPP = 0.87
ULTRA_IMAGE_SIZE = 2048
ULTRA_DELTA = _delta_from_target_mpp(ULTRA_TARGET_MPP, ULTRA_IMAGE_SIZE)

MODE_CONFIG: dict[str, dict[str, float | int | str]] = {
    "preview": {
        "image_size": 1024,
        "delta": 2.0,
        "mpp": _expected_mpp_equator(2.0, 1024),
        "wms_format": "image/png",
    },
    "detail": {
        "image_size": 1536,
        "delta": 0.80,
        "mpp": _expected_mpp_equator(0.80, 1536),
        "wms_format": "image/png",
    },
    "ultra": {
        "image_size": ULTRA_IMAGE_SIZE,
        "delta": ULTRA_DELTA,
        "mpp": ULTRA_TARGET_MPP,
        "wms_format": "image/png",
    },
}


def _normalize_bbox(bbox: Sequence[float]) -> UserBBox:
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 values: [xmin, ymin, xmax, ymax]")

    x_min, y_min, x_max, y_max = map(float, bbox)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox must satisfy xmax > xmin and ymax > ymin")

    return x_min, y_min, x_max, y_max


def _build_request_bbox(source_bbox: UserBBox, delta_deg: float) -> RequestBBox:
    x_min, y_min, x_max, y_max = source_bbox
    width = x_max - x_min
    height = y_max - y_min

    # If the input area is smaller than mode window, keep caller bbox unchanged.
    if width < delta_deg or height < delta_deg:
        return source_bbox

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    req_x_min = max(x_min, min(center_x - delta_deg / 2.0, x_max - delta_deg))
    req_y_min = max(y_min, min(center_y - delta_deg / 2.0, y_max - delta_deg))
    req_x_max = req_x_min + delta_deg
    req_y_max = req_y_min + delta_deg

    return req_x_min, req_y_min, req_x_max, req_y_max


def _has_vertical_white_stripes(gray_image: Image.Image) -> bool:
    width, height = gray_image.size
    pixels = gray_image.load()
    if pixels is None:
        return False

    run_start: int | None = None
    runs: list[tuple[int, int]] = []

    for x in range(width):
        white_count = 0
        for y in range(height):
            pixel_value = pixels[x, y]
            if isinstance(pixel_value, tuple):
                luminance = max(pixel_value)
            else:
                luminance = cast(float, pixel_value)

            if luminance >= STRIPE_WHITE_THRESHOLD:
                white_count += 1

        bad_column = (white_count / height) >= STRIPE_COLUMN_RATIO_THRESHOLD
        if bad_column and run_start is None:
            run_start = x
        elif not bad_column and run_start is not None:
            runs.append((run_start, x - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, width - 1))

    wide_runs = [run for run in runs if (run[1] - run[0] + 1) >= STRIPE_MIN_RUN_WIDTH]
    return len(wide_runs) > 0


def _is_empty_wms_tile(image: Image.Image) -> tuple[bool, str]:
    if "A" in image.getbands():
        alpha = image.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        if alpha_max == 0:
            return True, "WMS tile is fully transparent (no coverage)"

    gray_image = image.convert("L")
    gray_min, gray_max = gray_image.getextrema()
    if gray_min == gray_max and gray_min in (0, 255):
        return True, "WMS tile is flat (single-value image)"

    return False, ""


def download_tile(
    mode: str,
    bbox: Sequence[float],
) -> Image.Image:
    """Download a single lunar tile from fixed USGS/KaguyaTC_Ortho config.

    Args:
        mode: One of "preview", "detail", "ultra".
        bbox: Bounding box in EPSG:4326 as [xmin, ymin, xmax, ymax].

    Returns:
        PIL.Image.Image in grayscale ("L"). No files are written to disk.
    """

    if mode not in MODE_CONFIG:
        supported_modes = ", ".join(sorted(MODE_CONFIG.keys()))
        raise ValueError(f"Unknown mode '{mode}'. Supported modes: {supported_modes}")

    config = MODE_CONFIG[mode]
    image_size = int(config["image_size"])
    delta_deg = float(config["delta"])
    wms_image_format = str(config["wms_format"])

    normalized_bbox = _normalize_bbox(bbox)
    request_bbox = _build_request_bbox(normalized_bbox, delta_deg)

    params: dict[str, Any] = {
        "request": "GetMap",
        "service": "WMS",
        "version": WMS_VERSION,
        "layers": WMS_LAYER_NAME,
        "styles": "",
        "srs": WMS_SRS,
        "bbox": f"{request_bbox[0]},{request_bbox[1]},{request_bbox[2]},{request_bbox[3]}",
        "width": image_size,
        "height": image_size,
        "format": wms_image_format,
        "map": WMS_MAP_PATH,
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES_PER_TILE + 1):
        try:
            response = requests.get(
                WMS_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                raise RuntimeError("WMS response is not an image")

            raw_image = Image.open(BytesIO(response.content))
            is_empty_tile, empty_reason = _is_empty_wms_tile(raw_image)
            if is_empty_tile:
                raise RuntimeError(empty_reason)

            image = raw_image.convert("L")

            if STRIPE_FILTER_ENABLED and _has_vertical_white_stripes(image):
                raise RuntimeError("Detected vertical white seam artifact")

            return image
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES_PER_TILE:
                break
            time.sleep(RETRY_SLEEP_SECONDS)

    if last_error is None:
        raise RuntimeError("Failed to download tile after retries")

    raise RuntimeError(f"Failed to download tile after retries: {last_error}") from last_error


__all__ = ["MODE_CONFIG", "download_tile"]
