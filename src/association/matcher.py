from typing import Any, Dict, List, Optional, Tuple

import math


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Return the center point of a bounding box in (x1, y1, x2, y2) format.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_polygon_center(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Return the approximate center point of a 4-point OCR polygon.
    """
    x_values = [pt[0] for pt in points]
    y_values = [pt[1] for pt in points]
    return (sum(x_values) / len(x_values), sum(y_values) / len(y_values))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.dist(p1, p2)


def associate_text_to_objects(
    object_detections: List[Dict[str, Any]],
    text_detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each detected object, find the nearest detected text and attach it.
    """
    enriched_objects: List[Dict[str, Any]] = []

    for obj in object_detections:
        obj_center = get_box_center(obj["coordinates"])

        nearest_text: Optional[str] = None
        nearest_distance: Optional[float] = None

        for text_det in text_detections:
            coords = text_det["coordinates"]

            if isinstance(coords, tuple):
                text_center = get_box_center(coords)
            else:
                text_center = get_polygon_center(coords)

            dist = distance(obj_center, text_center)

            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist
                nearest_text = text_det.get("Predicted Class")

        enriched = dict(obj)
        enriched["NearestText"] = nearest_text
        enriched["NearestTextDistance"] = round(nearest_distance, 2) if nearest_distance is not None else None
        enriched_objects.append(enriched)

    return enriched_objects