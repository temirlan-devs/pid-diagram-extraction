from typing import Any, Dict, List, Tuple

import easyocr
import numpy as np


reader = easyocr.Reader(["en"])


def detect_text(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run OCR on a full image and return structured text detections.
    """
    results = reader.readtext(image)
    detections: List[Dict[str, Any]] = []

    for idx, (bbox, text, score) in enumerate(results):
        bbox = np.array(bbox).astype(int)
        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])

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
                "color": (255, 0, 0),
                "DetectionType": "Text",
            }
        )

    return detections


def detect_text_in_tiles(
    tiles: List[Tuple[np.ndarray, int, int]]
) -> List[Dict[str, Any]]:
    """
    Run OCR on image tiles and remap coordinates back to the original image.
    """
    detections: List[Dict[str, Any]] = []

    for tile, x_offset, y_offset in tiles:
        results = reader.readtext(tile)

        for bbox, text, score in results:
            bbox = np.array(bbox).astype(int)
            adjusted_bbox = [(int(pt[0] + x_offset), int(pt[1] + y_offset)) for pt in bbox]

            x1, y1 = adjusted_bbox[0]
            x2, y2 = adjusted_bbox[2]

            detections.append(
                {
                    "Predicted Class": text,
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "coordinates": adjusted_bbox,
                    "Score": float(score),
                    "DetectionType": "Text",
                    "color": (255, 0, 0),
                }
            )

    return detections