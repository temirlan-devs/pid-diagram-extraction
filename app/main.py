"""
Main Flask app entrypoint for P&ID diagram extraction.
"""

import base64
import io
import logging

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from src.detection.detector import detect_objects
from src.rendering.annotate import draw_bounding_boxes
from src.ocr.reader import detect_text, detect_text_in_tiles
from src.ocr.tiling import split_image

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the size of each sub-image
sub_image_size = 1024

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
    
    # Perform text detection on sub-images
    text_detections = detect_text(image_np)
    
    # Combine all detections
    logging.info("Before combining")
    all_detections = object_detections + text_detections
    logging.info("After combining")
    
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
    return jsonify({'csv_objects': csv_base64, 'detections_all': all_detections, 'object_detections': object_detections, 'text_detections': text_detections})

if __name__ == '__main__':
    app.run(debug=True)
