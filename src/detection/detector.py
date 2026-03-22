from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection = inter_width * inter_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union == 0:
        return 0.0

    return intersection / union


def deduplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate detections by keeping the higher-score box
    when overlap is above the threshold.
    """
    sorted_detections = sorted(detections, key=lambda d: d["Score"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for candidate in sorted_detections:
        candidate_box = candidate["coordinates"]
        is_duplicate = False

        for existing in kept:
            existing_box = existing["coordinates"]
            iou = compute_iou(candidate_box, existing_box)

            if iou >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


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
        conf=0.4,
        iou=0.4,
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

    deduplicated = deduplicate_detections(detections, iou_threshold=0.5)

    for idx, detection in enumerate(deduplicated, start=1):
        detection["ItemNumber"] = idx

    return deduplicated