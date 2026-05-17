from typing import Any, TypedDict
from uuid import uuid4


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

def _next_detection_id() -> str:
    return f"det-{uuid4().hex}"


def run_inference(image: Any) -> list[AdapterDetection]:
    """Run mock inference.

    The `image` argument is a placeholder for future model inputs
    (image array, tile path, or selected area metadata).
    """

    _ = image

    return [
        {
            "detection_id": _next_detection_id(),
            "bbox": {"x": 124.5, "y": 210.0, "width": 53.7, "height": 53.7},
            "confidence": 0.93,
            "class": "cave_candidate",
        },
        {
            "detection_id": _next_detection_id(),
            "bbox": {"x": 342.1, "y": 115.4, "width": 49.5, "height": 48.6},
            "confidence": 0.78,
            "class": "cave_candidate",
        },
        {
            "detection_id": _next_detection_id(),
            "bbox": {"x": 580.0, "y": 302.3, "width": 44.4, "height": 44.5},
            "confidence": 0.56,
            "class": "crater",
        },
    ]
