"""
Main Flask app entrypoint for P&ID diagram extraction.
"""

import base64
import io
import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from src.ocr.tiling import split_image
from src.pipeline.process_diagram import process_diagram

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    logging.info("Received image for processing")
    file = request.files['image']
    image = Image.open(file.stream)
    image_np = np.array(image)
    logging.debug(f"Original image size: {image_np.shape}")

    logging.info("Running diagram processing pipeline")
    pipeline_result = process_diagram(image)

    object_detections = pipeline_result["object_detections"]
    text_detections = pipeline_result["text_detections"]
    all_detections = pipeline_result["all_detections"]
    matched_objects = pipeline_result["matched_objects"]

    logging.info(f"Detected {len(object_detections)} objects and {len(text_detections)} text elements")
    
    # Convert detections to DataFrame
    selected_fields = ['Predicted Class', 'ItemNumber', 'x', 'y', 'width', 'height', 'Score']
    filtered_detections = [{field: detection.get(field) for field in selected_fields} for detection in object_detections]

    df = pd.DataFrame(filtered_detections)
    
    # Save DataFrame to a CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_string = output.getvalue()
    output.close()

    # Encode CSV string as base64 to send as JSON
    csv_base64 = base64.b64encode(csv_string.encode()).decode()
    #csv
    
    logging.info("Processed image sent back to client")
    logging.info("Processed csv sent back to client")
    return jsonify({'csv_objects': csv_base64, 'detections_all': all_detections, 'object_detections': object_detections, 'text_detections': text_detections, 'matched_objects': matched_objects})

if __name__ == '__main__':
    app.run(debug=True)
