from typing import Any, Dict, List, Tuple

import easyocr
import numpy as np


reader = easyocr.Reader(["en"])


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

            detections.append(
                {
                    "Predicted Class": text,
                    "x": adjusted_bbox[0][0],
                    "y": adjusted_bbox[0][1],
                    "coordinates": adjusted_bbox,
                    "Score": float(score),
                    "DetectionType": "Text",
                }
            )

    return detections