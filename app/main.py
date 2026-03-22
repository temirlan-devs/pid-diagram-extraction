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

from src.detection.detector import detect_objects
from src.ocr.reader import detect_text
from src.ocr.tiling import split_image
from src.association.matcher import associate_text_to_objects

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

    # Perform object detection on the entire image
    object_detections = detect_objects(image)
    
    # Perform text detection on full image
    text_detections = detect_text(image_np)
    
    # Combine all detections
    logging.info("Combining object and text detections")
    all_detections = object_detections + text_detections

    logging.info("Associating text with detected objects")
    matched_objects = associate_text_to_objects(object_detections, text_detections)
    
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
