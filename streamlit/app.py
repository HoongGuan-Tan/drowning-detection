import streamlit as st
from PIL import Image
import cv2
import os, uuid
import tempfile
import io
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import time
from collections import defaultdict
from playsound3 import playsound
from datetime import datetime
import threading
import queue

from models import *
from utils import *
from alert_overlay_manager import AlertOverlayManager
from alert_incident_manager import AlertIncidentManager

import warnings
warnings.filterwarnings("ignore")


# Set page configuration
st.set_page_config(
    page_title="Drowning Detection System",
    page_icon="🏊",
    layout="wide"
)

result_queue: queue.Queue[list[dict]] = queue.Queue()

# Threading for beep sound so it doesn't block frame capturing
def play_beep_sound():
    playsound("beep.mp3", block=True)


# Main app function
def main():
    st.title("🏊 Drowning Detection System")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    kan_model_path, cnn_model_path, yolo_model_path = get_model_paths()
    
    model_type = st.sidebar.radio(
        "Choose Classification Model",
        ["KAN", "CNN"],
        key='model_type_selection'
    )

    # Confidence Threshold Sliders
    st.sidebar.header("Detection Settings")
    yolo_conf_threshold = st.sidebar.slider(
        "Human Detection Sensitivity",
        min_value=0.1, max_value=1.0, value=0.4, step=0.05,
        key='yolo_conf'
    )
    classification_conf_threshold = st.sidebar.slider(
        "Classification Sensitivity",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        key='class_conf'
    )
    
    # Load models
    with st.spinner("Loading models... Please wait."):
        # Load Classification model
        classification_model, device = load_classification_model(model_type, kan_model_path, cnn_model_path)
    
    # Input method selection
    st.sidebar.header("Input Selection")
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Upload Image", "Upload Video", "Live Camera"],
        key='input_method_selection'
    )

    # --- Create alert reports folder ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_alerts_folder = os.path.join(base_dir, "video_alert_reports")
    os.makedirs(video_alerts_folder, exist_ok=True)
    live_alerts_folder = os.path.join(base_dir, "live_alert_reports")
    os.makedirs(live_alerts_folder, exist_ok=True)

    # --- Initialise Session State variables ---
    # Use dictionaries keyed by track_id
    if 'drowning_start_times' not in st.session_state:
         # Stores {track_id: start_time}
        st.session_state.drowning_start_times = defaultdict(lambda: None)
    if 'alert_triggered_ids' not in st.session_state:
         # Stores {track_id} for which alert is active
        st.session_state.alert_triggered_ids = set()
    if 'drowning_detected_ids' not in st.session_state:
         # Stores {track_id} detected in the current frame
        st.session_state.drowning_detected_ids = set()
    if 'session_id' not in st.session_state:
        # Unique session ID for this run
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'alert_frame_count' not in st.session_state:
        # Counter for alert frames
        st.session_state.alert_frame_count = 0
    if 'alert_frames_captured' not in st.session_state:
        # List to store paths of captured frames
        st.session_state.alert_frames_captured = []

    # --- Reset capturing timers at the start of session ---
    st.session_state.pop('alert_capture_start_time', None)
    st.session_state.pop('pdf_timer_started', None)

    # FPS state
    if 'last_process_time' not in st.session_state:
        st.session_state.last_process_time = time.time() # Initialise with current time
    if 'delta_time' not in st.session_state:
         st.session_state.delta_time = 0 # Time difference between frames
    if 'fps' not in st.session_state:
        st.session_state.fps = 0 # Smoothed FPS

    # --- Main Area ---
    st.header(f"Input Method: {input_method}")
    
    # Handle Different Input Methods

    # === Handle Image Upload ===
    if input_method == "Upload Image":
        # Load YOLO specifically for image upload
        yolo_detector_image = load_yolo_model(yolo_model_path)
        if yolo_detector_image is None or classification_model is None:
            st.error("Error loading models for image processing.")
            return
        
        col1, col2 = st.columns(2)
        col1.subheader("Original Image")
        col2.subheader("Processed Image")

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            sample_image_path = os.path.join(base_dir, "sample.jpg")
            if not os.path.exists(sample_image_path): sample_image_path = "sample.jpg" # Fallback
        except NameError:
            sample_image_path = "sample.jpg"

        sample_image = None
        processed_sample_image = None
        sample_detections = []
        user_detections = []

        try:
            if os.path.exists(sample_image_path):
                sample_image = Image.open(sample_image_path)
                with st.spinner("Processing default image..."):
                    processed_sample_image, sample_detections = process_image(
                        sample_image, yolo_detector_image, classification_model, device,
                        yolo_conf_threshold, classification_conf_threshold
                    )
            else:
                st.warning(f"Default sample image not found at {sample_image_path}.")
        except Exception as e:
            st.error(f"Error loading/processing default image: {e}")

        # --- File Uploader ---
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "jfif", "png"])
        if uploaded_file is None:
            if sample_image: 
                col1.image(sample_image, caption="Default Sample Image", use_container_width=True)
            else: 
                col1.info("No default image found/loaded.")
            if processed_sample_image is not None:
                col2.image(processed_sample_image, caption="Processed Default Image", use_container_width=True)
                if sample_detections:
                    col2.write("Detections (Default):")
                    drowning_detected_default = False
                    for det in sample_detections:
                        col2.write(f"- {det['label']} (Confidence: {det['confidence']:.2f})")
                        if det['class'] == 2: drowning_detected_default = True
                    # Alert for default image
                    if drowning_detected_default: st.error("🚨 ALERT: Potential Drowning Detected in Default Image!")
                else:
                    col2.info("No persons detected meeting the thresholds in the default image.")
            elif sample_image:
                col2.warning("Could not process default image.")
            else: 
                col2.info("Upload an image to see results.")
        else:
            # User uploaded
            try:
                image_bytes = uploaded_file.read(); user_image = Image.open(io.BytesIO(image_bytes))
                col1.image(user_image, caption="Uploaded Image", use_container_width=True)
                with st.spinner("Processing uploaded image..."):
                    processed_user_image, user_detections = process_image(
                        user_image, yolo_detector_image, classification_model, device,
                        yolo_conf_threshold, classification_conf_threshold
                    )
                col2.image(processed_user_image, caption="Processed Uploaded Image", use_container_width=True)
                if user_detections:
                    col2.write("Detections (Uploaded):")
                    drowning_detected = False
                    for det in user_detections:
                        col2.write(f"- {det['label']} (Confidence: {det['confidence']:.2f})")
                        if det['class'] == 2: drowning_detected = True
                    if drowning_detected: st.error("🚨 ALERT: Potential Drowning Detected!")
                else: col2.info("No persons detected in the uploaded image.")
            except Exception as e: st.error(f"Error processing uploaded image: {e}")
    
    # === Handle Video Upload ===
    elif input_method == "Upload Video":
        col1, col2 = st.columns(2)
        col1.subheader("Original Video Frame")
        col2.subheader("Processed Video Frame")
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        video_path = None

        if uploaded_video is not None:
            with st.spinner("Loading YOLO model for video..."):
                yolo_detector_video = load_yolo_model(yolo_model_path)

            if yolo_detector_video is None or classification_model is None:
                st.error("Error loading models for video processing.")
                return

            # Save uploaded video to a temporary file
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                cap = cv2.VideoCapture(video_path)
            except Exception as e_file:
                st.error(f"Error opening video file: {e_file}")
                return

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0:
                video_fps = 30
                st.warning(f"Could not determine video FPS. Assuming {video_fps} FPS.")

            alert_active_ids_video = set()

            st.session_state.alert_frame_count = 0
            st.session_state.alert_frames_captured = []


            frame_placeholder_orig = col1.empty()
            frame_placeholder_proc = col2.empty()
            alert_placeholder_video = st.empty()
            stop_button = st.button("Stop Processing Video", key="stop_video")

            frame_num = 0
            alert_manager = AlertOverlayManager()
            incident_manager = AlertIncidentManager(base_dir=video_alerts_folder, session_id=st.session_state.session_id)

            detection_start_time = {}
            active_incidents = set()

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                frame_placeholder_orig.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Original Frame {frame_num}", use_container_width=True)

                with st.spinner(f"Processing frame {frame_num}..."):
                    processed_frame_bgr, detections = process_video_frame_for_webrtc(
                        frame, yolo_detector_video, classification_model, device,
                        yolo_conf_threshold, classification_conf_threshold
                    )

                current_drowning_ids = set()
                now = time.time()

                if detections:
                    for det in detections:
                        if det['class'] == 2:
                            track_id = det.get('track_id')
                            if track_id is not None:
                                current_drowning_ids.add(track_id)

                                # Start timing if not already started
                                if track_id not in detection_start_time:
                                    detection_start_time[track_id] = now

                                # If 3 seconds passed, trigger alert (once)
                                elif (now - detection_start_time[track_id] >= 2 and track_id not in alert_active_ids_video):
                                    alert_active_ids_video.add(track_id)
                                    alert_placeholder_video.error(f"🚨 ALERT: Potential Drowning Detected (ID {track_id}) for 2+ Secs!")
                                    threading.Thread(target=play_beep_sound, daemon=True).start()

                                    if track_id not in incident_manager.incidents:
                                        incident_manager.start_incident(track_id)
                                    active_incidents.add(track_id)

                # For each active incident, capture frames for 5 seconds
                for track_id in list(active_incidents):
                    start_time = detection_start_time.get(track_id)
                    if start_time and (now - start_time <= 5):
                        incident_manager.capture_frame_only(track_id, processed_frame_bgr)
                    elif start_time and (now - start_time > 5):
                        active_incidents.remove(track_id)

                # Reset detection timer for IDs not seen in current frame
                ids_to_reset = set(detection_start_time.keys()) - current_drowning_ids
                for track_id in ids_to_reset:
                    if track_id not in alert_active_ids_video:  # don't reset confirmed alerts
                        detection_start_time.pop(track_id, None)

                alert_manager.update_alert(alert_active_ids_video)
                alert_manager.draw_alert(processed_frame_bgr)

                frame_placeholder_proc.image(processed_frame_bgr, channels="BGR", caption=f"Processed Frame {frame_num}", use_container_width=True)

                if stop_button:
                    st.warning("Video processing stopped by user.")
                    alert_placeholder_video.empty()
                    break

            cap.release()
            if video_path:
                try:
                    time.sleep(1)
                    os.unlink(video_path)
                except Exception as e:
                    st.warning(f"Error deleting temporary file {video_path}: {e}")

            # Generate PDF after processing
            for track_id in alert_active_ids_video:
                pdf_generated = incident_manager.generate_pdf(track_id)
                if pdf_generated:
                    st.success(f"📄 Alert report PDF generated for Track ID {track_id}")


            if not stop_button:
                st.success("Video processing complete.")
                if not alert_active_ids_video:
                    alert_placeholder_video.empty()
    
    
    
    # === Handle Live Camera ===
    elif input_method == "Live Camera":
        st.subheader("Live Camera Feed")
        st.caption("Allow camera access. Processing happens server-side.")

        with st.spinner("Loading YOLO model for live camera..."):
            yolo_detector_live = load_yolo_model(yolo_model_path)

        if yolo_detector_live is None or classification_model is None:
            st.error("Error loading models for live camera.")
            return

        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        incident_manager = AlertIncidentManager(base_dir=live_alerts_folder, session_id=st.session_state.session_id)
        alert_overlay_manager = AlertOverlayManager()

        detection_start_time = {}
        active_incidents = set()
        alert_active_ids_live = set()
        capture_start_time = {}

        def video_frame_callback(frame):
            image = frame.to_ndarray(format="bgr24")
            now = time.time()

            processed_img_bgr, detections = process_video_frame_for_webrtc(
                image, yolo_detector_live, classification_model, device,
                yolo_conf_threshold, classification_conf_threshold
            )

            # Detection table
            table_detections = [
                {
                    "ID": det.get("track_id", "N/A"),
                    "Label": det["label"],
                    "Confidence": f"{det['confidence']:.2f}",
                    "Class": det["class"],
                    "BBox": det["bbox"]
                } for det in detections
            ]
            result_queue.put(table_detections) # Put the formatted list in queue

            current_drowning_ids = set()
            for det in detections:
                if det['class'] == 2:
                    track_id = det.get('track_id')
                    if track_id is not None:
                        current_drowning_ids.add(track_id)

                        if track_id not in detection_start_time:
                            detection_start_time[track_id] = now

                        elif (now - detection_start_time[track_id] >= 2 and track_id not in alert_active_ids_live):
                            alert_active_ids_live.add(track_id)
                            st.toast(f"🚨 ALERT: Drowning Detected (ID {track_id})")
                            threading.Thread(target=play_beep_sound, daemon=True).start()
                            capture_start_time[track_id] = now  # Start 5-second capture window


                            if track_id not in incident_manager.incidents:
                                incident_manager.start_incident(track_id)

                            active_incidents.add(track_id)

            # Capture frames for 5 seconds
            for track_id in list(active_incidents):
                capture_start = capture_start_time.get(track_id)
                if capture_start and (now - capture_start <= 5):
                    incident_manager.capture_frame_only(track_id, processed_img_bgr)
                elif capture_start and (now - capture_start > 5):
                    active_incidents.remove(track_id)
                    capture_start_time.pop(track_id, None)
                    if track_id in alert_active_ids_live:
                        incident_manager.generate_pdf(track_id)


            # Reset timers for IDs not in frame
            ids_to_reset = set(detection_start_time.keys()) - current_drowning_ids
            for track_id in ids_to_reset:
                if track_id not in alert_active_ids_live:
                    detection_start_time.pop(track_id, None)

            alert_overlay_manager.update_alert(alert_active_ids_live)
            alert_overlay_manager.draw_alert(processed_img_bgr)

            return av.VideoFrame.from_ndarray(processed_img_bgr, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="live-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.checkbox("Show Detection Details Table", value=True, key="show_live_table"):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                while True:
                    try:
                        # Get the latest detections from the queue
                        result = result_queue.get(timeout=1.0) # Use timeout to prevent blocking indefinitely if queue is empty
                        labels_placeholder.table(result)
                    except queue.Empty:
                        # If the queue is empty after timeout, just continue loop
                        pass
                    # Check if the WebRTC connection is still active, otherwise break the loop
                    if not webrtc_ctx.state.playing:
                        break

        if webrtc_ctx.state.playing:
            st.success("Live feed running.")
        else:
            st.info("Click 'Start' to begin the live camera feed.")
            # Reset session states
            detection_start_time.clear()
            active_incidents.clear()
            alert_active_ids_live.clear()
            # Clear the queue when stopped
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break

if __name__ == "__main__":
    main()