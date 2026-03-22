from typing import Any, Dict

import numpy as np
from PIL import Image

from src.association.matcher import associate_text_to_objects
from src.detection.detector import detect_objects
from src.ocr.reader import detect_text
from src.rendering.annotate import draw_bounding_boxes


def process_diagram(image: Image.Image) -> Dict[str, Any]:
    """
    Run the full diagram-processing pipeline:
    object detection, text detection, and object-text association.
    """
    image_np = np.array(image)

    object_detections = detect_objects(image)
    text_detections = detect_text(image_np)
    all_detections = object_detections + text_detections
    matched_objects = associate_text_to_objects(object_detections, text_detections)
    annotated_image = draw_bounding_boxes(image, all_detections)

    return {
        "object_detections": object_detections,
        "text_detections": text_detections,
        "all_detections": all_detections,
        "matched_objects": matched_objects,
        "annotated_image": annotated_image,
    }