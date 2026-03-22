from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image


def draw_bounding_boxes(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """
    Draw bounding boxes and labels for object/text detections on an image.
    """
    image_np = np.array(image)

    for detection in detections:
        coords = detection["coordinates"]
        color = detection["color"]
        detection_type = detection.get("DetectionType")

        # Case 1: box coordinates like (x1, y1, x2, y2)
        if isinstance(coords, tuple):
            x1, y1, x2, y2 = coords
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

        # Case 2: polygon coordinates like [(x, y), ...]
        else:
            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image_np, [pts], isClosed=True, color=color, thickness=2)

            label = detection.get("text") or detection.get("Predicted Class", "")
            cv2.putText(
                image_np,
                label,
                (coords[0][0], coords[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return Image.fromarray(image_np)