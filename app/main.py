"""
Main Flask app entrypoint for P&ID diagram extraction.
"""

import base64
import io
import logging

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from src.pipeline.process_diagram import process_diagram
from src.utils.export_utils import build_export_files

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder="../templates")
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    logging.info("Received image for processing")

    file = request.files["image"]
    image = Image.open(file.stream)
    image_np = np.array(image)
    logging.debug(f"Original image size: {image_np.shape}")

    logging.info("Running diagram processing pipeline")
    pipeline_result = process_diagram(image)

    object_detections = pipeline_result["object_detections"]
    text_detections = pipeline_result["text_detections"]
    all_detections = pipeline_result["all_detections"]
    matched_objects = pipeline_result["matched_objects"]
    annotated_image = pipeline_result["annotated_image"]

    logging.info(
        f"Detected {len(object_detections)} objects and {len(text_detections)} text elements"
    )

    csv_base64, excel_base64 = build_export_files(
        object_detections=object_detections,
        matched_objects=matched_objects,
        text_detections=text_detections,
    )

    logging.info("Processed exports generated")

    img_byte_arr = io.BytesIO()
    annotated_image = annotated_image.convert("RGB")
    annotated_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    annotated_image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    logging.info("Processed image sent back to client")

    return jsonify(
        {
            "csv_objects": csv_base64,
            "excel_file": excel_base64,
            "detections_all": all_detections,
            "object_detections": object_detections,
            "text_detections": text_detections,
            "matched_objects": matched_objects,
            "annotated_image": annotated_image_base64,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)