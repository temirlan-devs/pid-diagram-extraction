from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = "models/best.pt" // TODO: replace with actual model path
model = YOLO(MODEL_PATH)


def detect_objects(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Run YOLO object detection on the full image and return structured detections.
    """
    detections: List[Dict[str, Any]] = []

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=image_bgr,
        imgsz=[image_np.shape[1], image_np.shape[0]],
    )

    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results[0].names[int(box.cls[0])]
        score = float(box.conf[0])

        detections.append(
            {
                "Predicted Class": label,
                "ItemNumber": idx + 1,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "Score": score,
                "coordinates": (x1, y1, x2, y2),
                "color": (0, 255, 0),
                "DetectionType": "Object",
            }
        )

    return detections