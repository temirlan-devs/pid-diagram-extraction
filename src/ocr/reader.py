from typing import Any, Dict, List, Tuple

import easyocr
import numpy as np


reader = easyocr.Reader(["en"])


def polygon_to_box(points: np.ndarray) -> Tuple[int, int, int, int]:
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return (
        int(np.min(x_coords)),
        int(np.min(y_coords)),
        int(np.max(x_coords)),
        int(np.max(y_coords)),
    )


def detect_text(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run OCR on a full image and return structured text detections
    using a unified bounding-box format.
    """
    results = reader.readtext(image)
    detections: List[Dict[str, Any]] = []

    for idx, (bbox, text, score) in enumerate(results):
        bbox_np = np.array(bbox).astype(int)
        x1, y1, x2, y2 = polygon_to_box(bbox_np)

        detections.append(
            {
                "Predicted Class": text,
                "ItemNumber": idx + 1,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "Score": float(score),
                "coordinates": (x1, y1, x2, y2),
                "DetectionType": "Text",
                "color": (255, 0, 0),
            }
        )

    return detections


def detect_text_in_tiles(
    tiles: List[Tuple[np.ndarray, int, int]]
) -> List[Dict[str, Any]]:
    """
    Run OCR on image tiles and remap coordinates back to the original image
    using a unified bounding-box format.
    """
    detections: List[Dict[str, Any]] = []

    for tile, x_offset, y_offset in tiles:
        results = reader.readtext(tile)

        for bbox, text, score in results:
            bbox_np = np.array(bbox).astype(int)

            adjusted_points = np.array(
                [[int(pt[0] + x_offset), int(pt[1] + y_offset)] for pt in bbox_np]
            )

            x1, y1, x2, y2 = polygon_to_box(adjusted_points)

            detections.append(
                {
                    "Predicted Class": text,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "coordinates": (x1, y1, x2, y2),
                    "Score": float(score),
                    "DetectionType": "Text",
                    "color": (255, 0, 0),
                }
            )

    return detections