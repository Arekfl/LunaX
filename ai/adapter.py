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
    },
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "best.pt"
# Loaded once at import time.
model = YOLO(str(MODEL_PATH))

def _next_detection_id() -> str:
    return f"det-{uuid4().hex}"


def _resolve_class_name(names: object, class_index: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_index, class_index))
    if isinstance(names, list) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


def run_inference(image: Image.Image, confidence_threshold: float = 0.25) -> list[AdapterDetection]:
    """Run YOLO inference on a single PIL image and return adapter detections."""

    results = model(image, verbose=False, conf=confidence_threshold)
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
                }
            )

    return detections
