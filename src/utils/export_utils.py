import base64
import io
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pandas as pd


def build_objects_dataframe(object_detections: List[Dict[str, Any]]) -> pd.DataFrame:
    object_fields = [
        "Predicted Class",
        "ItemNumber",
        "x",
        "y",
        "width",
        "height",
        "Score",
    ]
    return pd.DataFrame(
        [{field: detection.get(field) for field in object_fields} for detection in object_detections]
    )


def build_matched_objects_dataframe(matched_objects: List[Dict[str, Any]]) -> pd.DataFrame:
    matched_fields = [
        "Predicted Class",
        "ItemNumber",
        "NearestText",
        "NearestTextDistance",
        "x",
        "y",
        "width",
        "height",
        "Score",
    ]
    return pd.DataFrame(
        [{field: detection.get(field) for field in matched_fields} for detection in matched_objects]
    )


def build_text_detections_dataframe(text_detections: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "DetectedText": detection.get("Predicted Class"),
            "ItemNumber": detection.get("ItemNumber"),
            "x": detection.get("x"),
            "y": detection.get("y"),
            "width": detection.get("width"),
            "height": detection.get("height"),
            "Score": detection.get("Score"),
        }
        for detection in text_detections
    ]
    return pd.DataFrame(rows)


def build_readme_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Field": "Pipeline", "Description": "Symbol detection -> OCR text extraction -> spatial matching."},
            {"Field": "Units", "Description": "All coordinates and distances are in pixels."},
            {"Field": "Predicted Class", "Description": "Detected symbol class label."},
            {"Field": "DetectedText", "Description": "OCR text detected in the diagram."},
            {"Field": "ItemNumber", "Description": "Unique identifier assigned to each detected symbol."},
            {"Field": "NearestText", "Description": "Closest OCR text matched to a detected symbol."},
            {"Field": "NearestTextDistance", "Description": "Distance between detected symbol and matched text."},
            {"Field": "x", "Description": "Top-left x coordinate of the bounding box."},
            {"Field": "y", "Description": "Top-left y coordinate of the bounding box."},
            {"Field": "width", "Description": "Bounding box width in pixels."},
            {"Field": "height", "Description": "Bounding box height in pixels."},
            {"Field": "Score", "Description": "Model confidence score."},
        ]
    )


def build_csv_base64(objects_df: pd.DataFrame) -> str:
    output = io.StringIO()
    objects_df.to_csv(output, index=False)
    csv_string = output.getvalue()
    output.close()
    return base64.b64encode(csv_string.encode()).decode()


def build_excel_base64(
    objects_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    text_df: pd.DataFrame,
    readme_df: pd.DataFrame,
) -> str:
    excel_output = BytesIO()

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        objects_df.to_excel(writer, index=False, sheet_name="Detected Symbols")
        text_df.to_excel(writer, index=False, sheet_name="Detected Text")
        matched_df.to_excel(writer, index=False, sheet_name="Symbol-Text Matches")
        readme_df.to_excel(writer, index=False, sheet_name="README")

    excel_output.seek(0)
    return base64.b64encode(excel_output.getvalue()).decode("utf-8")


def build_export_files(
    object_detections: List[Dict[str, Any]],
    matched_objects: List[Dict[str, Any]],
    text_detections: List[Dict[str, Any]],
) -> Tuple[str, str]:
    objects_df = build_objects_dataframe(object_detections)
    matched_df = build_matched_objects_dataframe(matched_objects)
    text_df = build_text_detections_dataframe(text_detections)
    readme_df = build_readme_dataframe()

    csv_base64 = build_csv_base64(objects_df)
    excel_base64 = build_excel_base64(objects_df, matched_df, text_df, readme_df)

    return csv_base64, excel_base64