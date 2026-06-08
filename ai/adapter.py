from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from PIL import Image
from ultralytics import YOLO


class AdapterBBox(TypedDict):
    x: float
    y: float
    width: float
    height: float


AdapterDetection = TypedDict(
    "AdapterDetection",
    {
        "detection_id": str,
        "bbox": AdapterBBox,
        "confidence": float,
        "class": str,
        "class_id": int,
    },
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_NAME = "best.pt"
# Backward-compatible path reference.
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Models loaded on first use, keyed by file name.
_model_cache: dict[str, YOLO] = {}


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> YOLO:
    if model_name not in _model_cache:
        model_path = MODELS_DIR / model_name
        _model_cache[model_name] = YOLO(str(model_path))
    return _model_cache[model_name]


# Pre-load default model at import time (preserves existing behaviour).
_get_model(DEFAULT_MODEL_NAME)
# Backward-compatible alias.
model = _model_cache[DEFAULT_MODEL_NAME]


def _next_detection_id() -> str:
    return f"det-{uuid4().hex}"


def _resolve_class_name(names: object, class_index: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_index, class_index))
    if isinstance(names, list) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


def run_inference(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    image_size: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[AdapterDetection]:
    """Run YOLO inference on a single PIL image and return adapter detections."""

    yolo_model = _get_model(model_name)
    model_input = image if image.mode == "RGB" else image.convert("RGB")
    predict_kwargs: dict[str, object] = {
        "source": model_input,
        "verbose": False,
        "conf": confidence_threshold,
    }
    if image_size is not None:
        predict_kwargs["imgsz"] = int(image_size)

    results = yolo_model.predict(**predict_kwargs)
    detections: list[AdapterDetection] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            class_index = int(box.cls[0].item())
            class_name = _resolve_class_name(result.names, class_index)

            detections.append(
                {
                    "detection_id": _next_detection_id(),
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "width": max(0.0, x2 - x1),
                        "height": max(0.0, y2 - y1),
                    },
                    "confidence": confidence,
                    "class": class_name,
                    "class_id": class_index,
                }
            )

    return detections
