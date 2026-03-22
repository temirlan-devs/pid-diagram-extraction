from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image


def draw_bounding_boxes(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """
    Draw bounding boxes and labels for object and text detections.
    All detections use the unified (x1, y1, x2, y2) coordinate format.
    """
    image_np = np.array(image)

    for detection in detections:
        x1, y1, x2, y2 = detection["coordinates"]
        color = detection["color"]
        detection_type = detection.get("DetectionType")

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        if detection_type == "Object":
            label = f"{detection['Predicted Class']} {detection['Score']:.2f}"
        else:
            label = detection.get("Predicted Class", "")

        cv2.putText(
            image_np,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return Image.fromarray(image_np)