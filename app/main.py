"""
Main Flask app entrypoint for P&ID diagram extraction.
"""

import base64
import io
import logging

import cv2
import easyocr
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from src.detection.detector import detect_objects

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
logging.info("EasyOCR reader initialized")

# Define the size of each sub-image
sub_image_size = 1024

def split_image(image, sub_image_size):
    height, width, _ = image.shape
    sub_images = []
    for y in range(0, height, sub_image_size):
        for x in range(0, width, sub_image_size):
            sub_image = image[y:y + sub_image_size, x:x + sub_image_size]
            sub_images.append((sub_image, x, y))
    logging.debug(f"Image split into {len(sub_images)} sub-images")
    return sub_images

def detect_text_subimages(sub_images):
    detections = []
    total_sub_images = len(sub_images)
    for i, (sub_image, x_offset, y_offset) in enumerate(sub_images):
        result = reader.readtext(sub_image)
        for (bbox, text, _) in result:
            bbox = np.array(bbox).astype(int)
            adjusted_bbox = [(pt[0] + x_offset, pt[1] + y_offset) for pt in bbox]
            detections.append({'coordinates': adjusted_bbox, 'text': text, 'color': (255, 0, 0)}) # Blue color
        logging.info(f"Processed sub-image {i + 1}/{total_sub_images} for text detection")
    return detections

def detect_text(image):
    logging.info("Text detection started on the entire image")
    image_np = np.array(image)
    result = reader.readtext(image_np)
    detections = []
    for idx, (bbox, text, score) in enumerate(result):
        bbox = np.array(bbox).astype(int)
        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])
        width = x2 - x1
        height = y2 - y1
        #detections.append({'coordinates': bbox, 'text': text, 'color': (255, 0, 0)}) # Blue color
        detections.append({'Predicted Class': text, 'ItemNumber': idx + 1, 'x': x1, 'y': y1, 'width': width, 'height': height, 'Score': score, 'coordinates': (x1, y1, x2, y2), 'color': (255, 0, 0), 'DetectionType': "Text"})
    logging.info("Text detection completed on the entire image")
    return detections

def draw_bounding_boxes(image, detections):
    image_np = np.array(image)
    for detection in detections:
        coords = detection['coordinates']
        color = detection['color']
        if 'Predicted Class' in detection:
            x1, y1, x2, y2 = coords
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, f"{detection['Predicted Class']} {detection['Score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image_np, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(image_np, detection['text'], (coords[0][0], coords[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return Image.fromarray(image_np)

def draw_certain_bounding_boxes(image, detections, item_number):
    image_np = np.array(image)
    for detection in detections:
        if detection['ItemNumber'] == item_number:
            coords = detection['coordinates']
            color = detection['color']
            if 'Predicted Class' in detection:
                x1, y1, x2, y2 = coords
                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_np, f"{detection['Predicted Class']} {detection['Score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image_np, [pts], isClosed=True, color=color, thickness=2)
                cv2.putText(image_np, detection['text'], (coords[0][0], coords[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return Image.fromarray(image_np)

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
    text_detections = detect_text(image)
    
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
