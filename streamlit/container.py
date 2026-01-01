# --- In utils.py ---

import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
# Ensure models are imported if needed directly, though likely not needed here
# from models import *
import time

# (Keep get_model_paths, YOLODetector, load_yolo_model, load_classification_model as they are)
# ...

# Preprocess image for classification (remains the same)
def preprocess_for_classification(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

# *** MODIFIED: Accept classification_threshold ***
# Perform classification
def classify_person(image, model, device, classification_threshold=0.5): # Add threshold param
    input_tensor = preprocess_for_classification(image)
    input_tensor = input_tensor.to(device)

    class_labels = {0: 'swimming', 1: 'tread water', 2: 'drowning'}

    pred_label = "Unknown" # Default label
    confidence = 0.0
    pred_class = -1 # Default class

    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

        # --- Performance Metrics (will be refined in Feature 2) ---
        # Store classification time if needed later, maybe return it?
        classification_time_ms = (end_time - start_time) * 1000
        print(f"Classification ({type(model).__name__}) took: {classification_time_ms:.2f} ms")
        # --- End Performance Metrics ---

        probabilities = torch.softmax(output, dim=1)[0]
        max_confidence, predicted_class_tensor = torch.max(probabilities, dim=0)
        max_confidence = max_confidence.item()
        predicted_class_idx = predicted_class_tensor.item()

        # *** ADDED: Check against classification threshold ***
        if max_confidence >= classification_threshold:
            pred_class = predicted_class_idx
            confidence = max_confidence
            pred_label = class_labels.get(pred_class, "Unknown")
        else:
            # If below threshold, consider it uncertain or default to a safe class if applicable
            pred_label = "Uncertain" # Or maybe default to 'swimming' depending on safety bias
            confidence = max_confidence # Still report the highest confidence found
            pred_class = -1 # Indicate uncertainty

    # Return classification time along with other results if needed for FPS calculation
    return pred_label, confidence, pred_class #, classification_time_ms

# *** MODIFIED: Accept yolo_threshold and classification_threshold ***
# Process image with YOLO and classification
def process_image(image, yolo_detector, classification_model, device,
                  yolo_threshold=0.4, classification_threshold=0.5): # Add thresholds
    # Convert PIL image to numpy array for YOLO
    img_array = np.array(image)

    # Run YOLO detection
    yolo_start_time = time.time()
    results = yolo_detector.detect(img_array)
    yolo_end_time = time.time()
    yolo_time_ms = (yolo_end_time - yolo_start_time) * 1000
    print(f"YOLO Detection took: {yolo_time_ms:.2f} ms")

    if results is None:
        # Return total processing time if needed later
        total_time_ms = yolo_time_ms
        return img_array, [], total_time_ms # Return time

    img_with_boxes = img_array.copy()
    detections = []
    total_classification_time_ms = 0

    boxes = results[0].boxes
    for box in boxes:
        cls = int(box.cls.item())
        conf = box.conf.item()

        # *** MODIFIED: Use yolo_threshold ***
        if cls == 0 and conf >= yolo_threshold:  # Person class detected above threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_roi = img_array[y1:y2, x1:x2] # Use original img_array for ROI

            if person_roi.size > 0:
                person_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)) # Ensure ROI is RGB for PIL

                # *** MODIFIED: Pass classification_threshold ***
                # Consider capturing classification time here if needed per detection
                pred_label, confidence, pred_class = classify_person(
                    person_pil, classification_model, device, classification_threshold
                )
                # total_classification_time_ms += class_time # Accumulate if returned

                # Only draw and add to detections if classification is not "Uncertain"
                # OR decide if you want to show uncertain detections differently
                if pred_class != -1: # Check if classification was certain enough
                    # Determine color based on classification
                    if pred_class == 0: color = (0, 255, 0)  # Green (swimming)
                    elif pred_class == 1: color = (0, 255, 255) # Yellow (tread water) - Changed BGR
                    else: color = (0, 0, 255)  # Red (drowning) - Changed BGR

                    bbox_height = y2 - y1
                    font_scale = max(bbox_height / 200.0, 0.5)

                    # Draw bounding box on the copy
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

                    label = f"{pred_label}: {confidence:.2f}"
                    cv2.putText(img_with_boxes, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": pred_label,
                        "confidence": confidence,
                        "class": pred_class
                    })
                # --- Optional: Handle uncertain detections visually ---
                # else:
                #     # Maybe draw a grey box for uncertain detections?
                #     color = (128, 128, 128) # Grey
                #     cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 1)
                #     label = f"Uncertain: {confidence:.2f}"
                #     cv2.putText(img_with_boxes, label, (x1, y1-10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                #     # Do not add uncertain detections to the list used for alerts?
                #     # Or add them with a special flag if needed elsewhere.

    total_time_ms = yolo_time_ms # + total_classification_time_ms # Add class time if needed
    # Return total processing time for this frame
    return img_with_boxes, detections #, total_time_ms


# *** MODIFIED: Accept and pass thresholds ***
# Process video frame function for WebRTC
def process_video_frame_for_webrtc(frame: np.ndarray, yolo_detector, classification_model, device,
                                   yolo_threshold=0.4, classification_threshold=0.5): # Add thresholds
    overall_start_time = time.time() # Start timer for overall processing

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
    except Exception as e:
        # Return original frame if conversion fails
        print(f"Error converting frame: {e}")
        return frame, [], 0 # Return 0 processing time

    # *** MODIFIED: Pass thresholds ***
    # Consider capturing total_processing_time from process_image if returned
    processed_frame_rgb_np, detections = process_image(
        pil_image, yolo_detector, classification_model, device,
        yolo_threshold, classification_threshold
    )

    # Convert final processed frame back to BGR
    processed_frame_bgr_np = cv2.cvtColor(processed_frame_rgb_np, cv2.COLOR_RGB2BGR)

    overall_end_time = time.time() # End timer
    total_processing_time_ms = (overall_end_time - overall_start_time) * 1000
    print(f"Total frame processing time (WebRTC): {total_processing_time_ms:.2f} ms")

    # Return processing time along with results
    return processed_frame_bgr_np, detections, total_processing_time_ms