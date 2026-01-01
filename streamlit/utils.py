import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from ultralytics import YOLO
from models import *
import time

# Model path configuration
def get_model_paths():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    model_dir = os.path.join(base_dir, "models")

    if not os.path.exists(model_dir):
        model_dir = "models" # Assume 'models' in the current working directory

    # Define paths using the provided directory
    kan_model_path = os.path.join(model_dir, "kan_rgb_model.pth")
    cnn_model_path = os.path.join(model_dir, "cnn_rgb_model.pth")
    yolo_model_path = os.path.join(model_dir, "best.pt")
    
    return kan_model_path, cnn_model_path, yolo_model_path

# YOLO for Object Detection
class YOLODetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        try:
            # Use forward slashes for paths for better cross-platform compatibility
            model_path_norm = model_path.replace("\\", "/")
            if not os.path.exists(model_path_norm):
                st.sidebar.error(f"YOLO model file not found: {model_path_norm}")
                self.model = None
            else:
                self.model = YOLO(model_path_norm) # Load model
                if self.model:
                    st.sidebar.success("YOLO model loaded successfully!")
                else:
                    st.sidebar.error(f"Failed to load YOLO model from: {model_path_norm}")
        except ImportError:
            st.sidebar.error("Ultralytics package not found.")
            self.model = None
        except Exception as e:
            st.sidebar.error(f"Error loading YOLO model: {e}")
            self.model = None

    def detect(self, img_array):
        if self.model is None:
            print("YOLO model not loaded.")
            return None

        results = None # Initialize results to None

        try:
            # --- Frame Validation ---
            if not isinstance(img_array, np.ndarray):
                 print(f"Error: Input is not a numpy array, type: {type(img_array)}")
                 return None
            print(f"Tracking frame with shape: {img_array.shape}, dtype: {img_array.dtype}") # Print frame info
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                 print(f"Warning: Unexpected frame shape {img_array.shape}. Converting if possible.")
                 # Attempt conversion or return error? For now, let YOLO handle it or fail.
                 pass
            if img_array.dtype != np.uint8:
                 print(f"Warning: Frame dtype is {img_array.dtype}, converting to uint8.")
                 img_array = img_array.astype(np.uint8)
            # --- End Frame Validation ---

            # Convert to RGB for YOLO
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # --- Explicitly Specify Tracker & Wrap track call ---
            try:
                start_time = time.time()

                # Specify tracker: 'bytetrack.yaml' or 'botsort.yaml'
                results = self.model.track(
                    img_rgb,
                    persist=True,
                    verbose=False,
                    tracker='bytetrack.yaml' # Explicitly specify tracker
                )

                end_time = time.time()
                detection_time_ms = (end_time - start_time) * 1000
                print(f"Human Detection (YOLO) took: {detection_time_ms:.2f} ms")
            except cv2.error as cv_err:
                 # Catch the specific OpenCV error
                 if "prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size()" in str(cv_err):
                     st.error(f"OpenCV Assertion Error during tracking (likely inconsistent frame properties). Frame shape: {img_array.shape}. Skipping frame.")
                     print(f"OpenCV Assertion Error details: {cv_err}")
                     return None
                 else:
                     # Re-raise other unexpected OpenCV errors
                     st.error(f"Unexpected OpenCV error during tracking: {cv_err}")
                     raise cv_err
            except Exception as track_err:
                 # Catch other potential errors during tracking
                 st.error(f"Error during model.track call: {track_err}")
                 raise track_err

            return results # Return results if tracking was successful

        except Exception as e:
            # Catch errors in the outer try block (e.g., color conversion)
            st.error(f"Error processing frame for YOLO tracking: {e}")
            return None

# Load YOLO model
@st.cache_resource
def load_yolo_model(model_path):
    # The constructor now handles the checks and messages
    detector = YOLODetector(model_path)
    if detector.model is None:
         # Error message is already shown by the constructor
         return None
    return detector

# Load KAN classification model
@st.cache_resource
def load_classification_model(model_type, kan_path, cnn_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = None
        model_path = ""

        if model_type == "KAN":
            model = KAN_RGB()
            model_path = kan_path
        else:
            model = CNN_RGB()
            model_path = cnn_path

        if not os.path.exists(model_path):
            st.sidebar.error(f"Classification model file not found: {model_path}")
            return None, device # Return None if file doesn't exist

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        st.sidebar.success(f"{model_type} model loaded successfully!")
        return model, device
    except FileNotFoundError:
         st.sidebar.error(f"Model file not found at path: {model_path}. Please check the path and ensure file is in deployment.")
         return None, torch.device("cpu")
    except Exception as e:
        st.sidebar.error(f"Error loading {model_type} model: {e}")
        return None, torch.device("cpu")
    
# Preprocess image for classification
def preprocess_for_classification(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

# Perform classification
def classify_person(image, model, device, classification_threshold=0.5):
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

        classification_time_ms = (end_time - start_time) * 1000
        print(f"Classification ({type(model).__name__}) took: {classification_time_ms:.2f} ms")

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
            pred_label = "Uncertain"
            confidence = max_confidence # Still report the highest confidence found
            pred_class = -1 # Indicate uncertainty

    return pred_label, confidence, pred_class

# Process image with YOLO and classification
def process_image(image, yolo_detector, classification_model, device,
                  yolo_threshold=0.4, classification_threshold=0.5):
    img_array = np.array(image)

    # --- Call the tracking method ---
    results = yolo_detector.detect(img_array)

    if results is None:
         print("Tracking failed or skipped for this frame.")
         return img_array, []

    if results[0].boxes is None or results[0].boxes.id is None:
        # Check if tracking produced results and track IDs
        return img_array, [] # Return empty detections

    img_with_boxes = img_array.copy()
    detections = []

    # --- Iterate through tracked boxes ---
    boxes = results[0].boxes
    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist() # Get track IDs as a list
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        xys = boxes.xyxy.int().cpu().tolist()

        for track_id, conf, cls, xyxy in zip(track_ids, confs, clss, xys):
            if cls == 0 and conf >= yolo_threshold:  # Person class detected above threshold
                x1, y1, x2, y2 = xyxy
                person_roi = img_array[y1:y2, x1:x2]

                if person_roi.size > 0:
                    person_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                    pred_label, confidence, pred_class = classify_person(
                        person_pil, classification_model, device, classification_threshold
                    )

                    if pred_class != -1: # Check if classification was certain
                        # Determine color (RGB)
                        if pred_class == 0: color = (0, 255, 0) # Green
                        elif pred_class == 1: color = (255, 255, 0) # Yellow
                        else: color = (255, 0, 0) # Red

                        bbox_height = y2 - y1
                        font_scale = max(bbox_height / 200.0, 0.5)

                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

                        # --- Add Track ID to label ---
                        label = f"ID {track_id}: {pred_label} ({confidence:.2f})"
                        cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

                        # --- Include track_id in detection dict ---
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "label": pred_label,
                            "confidence": confidence,
                            "class": pred_class,
                            "track_id": track_id # Add track ID
                        })

    return img_with_boxes, detections

# Process video frame function for WebRTC
def process_video_frame_for_webrtc(frame: np.ndarray, yolo_detector, classification_model, device,
                                   yolo_threshold=0.4, classification_threshold=0.5):
    overall_start_time = time.time()
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
    except Exception as e:
        print(f"Error converting frame: {e}")
        return frame, []

    processed_frame_rgb_np, detections = process_image(
        pil_image, yolo_detector, classification_model, device,
        yolo_threshold, classification_threshold
    )

    # Convert final processed frame back to BGR
    processed_frame_bgr_np = cv2.cvtColor(processed_frame_rgb_np, cv2.COLOR_RGB2BGR)

    overall_end_time = time.time()
    total_processing_time_ms = (overall_end_time - overall_start_time) * 1000
    print(f"Total frame processing time (WebRTC callback): {total_processing_time_ms:.2f} ms")

    # Return processing time along with results
    return processed_frame_bgr_np, detections