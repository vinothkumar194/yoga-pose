import streamlit as st
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-time Yoga Pose Detection",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Yoga Poses List (107 Poses) ---
yoga_poses = [
    "adho mukha svanasana", "adho mukha vriksasana", "agnistambhasana", "ananda balasana",
    "anantasana", "anjaneyasana", "ardha bhekasana", "ardha chandrasana", "ardha matsyendrasana",
    "ardha pincha mayurasana", "ardha uttanasana", "ashtanga namaskara", "astavakrasana",
    "baddha konasana", "bakasana", "balasana", "bhairavasana", "bharadvajasana i", "bhekasana",
    "bhujangasana", "bhujapidasana", "bitilasana", "camatkarasana", "chakravakasana",
    "chaturanga dandasana", "dandasana", "dhanurasana", "durvasasana", "dwi pada viparita dandasana",
    "eka pada koundinyanasana i", "eka pada koundinyanasana ii", "eka pada rajakapotasana",
    "eka pada rajakapotasana ii", "ganda bherundasana", "garbha pindasana", "garudasana",
    "gomukhasana", "halasana", "hanumanasana", "janu sirsasana", "kapotasana", "krounchasana",
    "kurmasana", "lolasana", "makara adho mukha svanasana", "makarasana", "malasana",
    "marichyasana i", "marichyasana iii", "marjaryasana", "matsyasana", "mayurasana",
    "natarajasana", "padangusthasana", "padmasana", "parighasana", "paripurna navasana",
    "parivrtta janu sirsasana", "parivrtta parsvakonasana", "parivrtta trikonasana",
    "parsva bakasana", "parsvottanasana", "pasasana", "paschimottanasana", "phalakasana",
    "pincha mayurasana", "prasarita padottanasana", "purvottanasana", "salabhasana",
    "salamba bhujangasana", "salamba sarvangasana", "salamba sirsasana", "savasana",
    "setu bandha sarvangasana", "simhasana", "sukhasana", "supta baddha konasana",
    "supta matsyendrasana", "supta padangusthasana", "supta virasana", "tadasana",
    "tittibhasana", "tolasana", "tulasana", "upavistha konasana", "urdhva dhanurasana",
    "urdhva hastasana", "urdhva mukha svanasana", "urdhva prasarita eka padasana", "ustrasana",
    "utkatasana", "uttana shishosana", "uttanasana", "utthita ashwa sanchalanasana",
    "utthita hasta padangusthasana", "utthita parsvakonasana", "utthita trikonasana",
    "vajrasana", "vasisthasana", "viparita karani", "virabhadrasana i", "virabhadrasana ii",
    "virabhadrasana iii", "virasana", "vriksasana", "vrischikasana", "yoganidrasana"
]

# --- Session State Initialization ---
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'yoga_model' not in st.session_state:
    st.session_state.yoga_model = None

# --- Lazy Model Loading with Error Handling ---
def load_models():
    """Load models with comprehensive error handling and version compatibility."""
    if st.session_state.models_loaded:
        return st.session_state.yolo_model, st.session_state.yoga_model
    
    try:
        # Check if model files exist
        model_dir = Path(".")
        yolo_path = model_dir / "yolov8n.pt"
        yoga_path = model_dir / "yoga-model.h5"
        
        if not yolo_path.exists():
            st.error(f"YOLO model file not found: {yolo_path}")
            return None, None
            
        if not yoga_path.exists():
            st.error(f"Yoga model file not found: {yoga_path}")
            return None, None
        
        # Load YOLO model with error handling
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(yolo_path))
            logger.info("YOLO model loaded successfully")
        except ImportError as e:
            st.error("ultralytics package not found. Please install it using: pip install ultralytics")
            return None, None
        except Exception as e:
            st.error(f"Error loading YOLO model: {str(e)}")
            return None, None
        
        # Load TensorFlow model with version compatibility
        try:
            # Try different TensorFlow import strategies
            tensorflow_loaded = False
            
            # Strategy 1: Standard TensorFlow
            try:
                import tensorflow as tf
                # Set memory growth to avoid GPU issues
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                tensorflow_loaded = True
                logger.info("TensorFlow loaded successfully")
            except ImportError:
                pass
            
            # Strategy 2: TensorFlow Lite (fallback)
            if not tensorflow_loaded:
                try:
                    import tensorflow.lite as tflite
                    st.warning("Using TensorFlow Lite. Some features may be limited.")
                    tensorflow_loaded = True
                except ImportError:
                    pass
            
            if not tensorflow_loaded:
                st.error("TensorFlow not found. Please install tensorflow>=2.8.0")
                return None, None
            
            # Build model architecture
            yoga_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', 
                                     activation='relu', input_shape=(64, 64, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(107, activation='softmax')
            ])
            
            # Load weights with error handling
            try:
                yoga_model.load_weights(str(yoga_path))
                logger.info("Yoga model weights loaded successfully")
            except Exception as e:
                st.error(f"Error loading yoga model weights: {str(e)}")
                return None, None
                
        except Exception as e:
            st.error(f"Error setting up TensorFlow model: {str(e)}")
            return None, None
        
        # Cache models in session state
        st.session_state.yolo_model = yolo_model
        st.session_state.yoga_model = yoga_model
        st.session_state.models_loaded = True
        
        return yolo_model, yoga_model
        
    except Exception as e:
        st.error(f"Unexpected error loading models: {str(e)}")
        logger.error(f"Model loading error: {str(e)}")
        return None, None

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """Preprocess image for yoga pose prediction."""
    try:
        target_size = (64, 64)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0  # Ensure float32 for compatibility
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

# --- Predict Yoga Pose Function ---
def predict_yoga_pose(model, image):
    """Predict yoga pose with error handling."""
    try:
        processed_img = preprocess_image(image)
        if processed_img is None:
            return -1, 0.0
            
        prediction = model.predict(processed_img, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))  # Ensure float conversion
        return int(predicted_class_index), confidence
    except Exception as e:
        logger.error(f"Error predicting yoga pose: {str(e)}")
        return -1, 0.0

# --- Detect & Classify Persons in Frame ---
def detect_and_classify_poses(image, yolo_model, yoga_model):
    """Detect and classify poses with comprehensive error handling."""
    try:
        results = yolo_model(image, verbose=False)  # Suppress YOLO verbose output
        annotated_image = image.copy()
        detected_poses = []

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        if cls == 0 and conf > 0.5:  # Person class
                            # Ensure valid crop coordinates
                            h, w = image.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:  # Valid bounding box
                                cropped_img = image[y1:y2, x1:x2]
                                if cropped_img.size > 0:
                                    pose_idx, pose_conf = predict_yoga_pose(yoga_model, cropped_img)

                                    # Safety check for pose index
                                    if 0 <= pose_idx < len(yoga_poses):
                                        pose_name = yoga_poses[pose_idx]
                                    else:
                                        pose_name = "unknown"

                                    # Draw bounding box and label
                                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"{pose_name} ({pose_conf:.2f})"
                                    
                                    # Calculate label position to avoid overflow
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    label_y = max(y1 - 10, label_size[1] + 10)
                                    
                                    cv2.rectangle(annotated_image, (x1, label_y - label_size[1] - 5), 
                                                (x1 + label_size[0], label_y + 5), (0, 255, 0), -1)
                                    cv2.putText(annotated_image, label, (x1, label_y),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                                    detected_poses.append({
                                        'pose': pose_name, 
                                        'confidence': pose_conf,
                                        'bbox': (x1, y1, x2, y2)
                                    })
                    except Exception as e:
                        logger.error(f"Error processing detection box: {str(e)}")
                        continue

        return annotated_image, detected_poses
    except Exception as e:
        logger.error(f"Error in pose detection: {str(e)}")
        return image, []

# --- Camera Control Functions ---
def start_camera(resolution):
    """Start camera with better error handling and device detection."""
    try:
        width, height = resolution
        
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        
        # Try multiple camera indices and backends
        camera_backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]
        indices_to_try = [0, 1, 2, -1]
        
        found = False
        for backend in camera_backends:
            for index in indices_to_try:
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        # Test if we can read a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            st.session_state.cap = cap
                            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
                            st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                            found = True
                            logger.info(f"Camera opened successfully with index {index} and backend {backend}")
                            break
                        else:
                            cap.release()
                    else:
                        cap.release()
                except Exception as e:
                    logger.error(f"Failed to open camera {index} with backend {backend}: {str(e)}")
                    continue
            if found:
                break
        
        if not found:
            st.error("Failed to open any camera device. Please check your camera connection.")
            return False
            
        st.session_state.camera_running = True
        return True
        
    except Exception as e:
        st.error(f"Error starting camera: {str(e)}")
        logger.error(f"Camera start error: {str(e)}")
        return False

def stop_camera():
    """Stop camera with proper cleanup."""
    try:
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.camera_running = False
        logger.info("Camera stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")

# --- Main App Logic ---
def main():
    st.title("🧘‍♀️ Real-time Yoga Pose Detection")
    st.markdown("Click 'Start Camera' to begin real-time detection.")

    # System Information
    with st.expander("System Information"):
        st.write(f"Python Version: {sys.version}")
        try:
            import tensorflow as tf
            st.write(f"TensorFlow Version: {tf.__version__}")
        except ImportError:
            st.write("TensorFlow: Not installed")
        
        try:
            import ultralytics
            st.write(f"Ultralytics Version: {ultralytics.__version__}")
        except ImportError:
            st.write("Ultralytics: Not installed")

    # Sidebar Info
    st.sidebar.title("About")
    st.sidebar.info("Detects yoga poses using YOLO and a custom CNN model.")
    st.sidebar.title("Supported Poses")
    with st.sidebar.expander("View all 107 poses"):
        for i, pose in enumerate(yoga_poses, 1):
            st.write(f"{i}. {pose.replace('_', ' ').title()}")

    # Model Loading Status
    st.subheader("🤖 AI Models")
    model_status = st.empty()
    
    with st.spinner("Loading AI models..."):
        yolo_model, yoga_model = load_models()
    
    if yolo_model is None or yoga_model is None:
        model_status.error("❌ Models failed to load. Please check the error messages above.")
        st.stop()
    else:
        model_status.success("✅ Models loaded successfully!")

    # Camera Resolution Selection
    st.subheader("📹 Camera Settings")
    resolution_options = {
        "320x240 (Fast)": (320, 240),
        "640x480 (Balanced)": (640, 480),
        "1280x720 (High Quality)": (1280, 720),
    }
    selected_resolution = st.selectbox(
        "Select Camera Resolution", 
        options=list(resolution_options.keys()), 
        index=1
    )

    # Camera Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Start Camera", key="start_button"):
            start_camera(resolution_options[selected_resolution])
    with col2:
        if st.button("⏹️ Stop Camera", key="stop_button"):
            stop_camera()
    with col3:
        camera_status = st.empty()

    # Main Content Area
    video_placeholder = st.empty()
    poses_placeholder = st.empty()

    # Camera Loop
    if st.session_state.camera_running and st.session_state.cap:
        camera_status.success("🟢 Camera Running")
        frame_count = 0
        
        try:
            while st.session_state.camera_running:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.warning("Failed to read from camera.")
                    stop_camera()
                    break

                frame = cv2.flip(frame, 1)  # Mirror effect
                frame_count += 1

                # Process every nth frame to improve performance
                if frame_count % 3 == 0:  # Process every 3rd frame
                    # Resize for faster inference
                    small_frame = cv2.resize(frame, (320, 240))
                    annotated_frame, detected_poses = detect_and_classify_poses(
                        small_frame, yolo_model, yoga_model
                    )
                    
                    # Resize back to display size
                    display_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
                else:
                    display_frame = frame
                    detected_poses = []

                # Convert color space for display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame, width=700)

                # Display pose information
                with poses_placeholder.container():
                    st.subheader("📊 Detected Poses")
                    if detected_poses:
                        for i, p in enumerate(detected_poses, 1):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"**Pose {i}:** {p['pose'].replace('_', ' ').title()}")
                            with col2:
                                st.write(f"Confidence: {p['confidence']:.2%}")
                            with col3:
                                if p['confidence'] > 0.8:
                                    st.success("High")
                                elif p['confidence'] > 0.6:
                                    st.warning("Medium")
                                else:
                                    st.error("Low")
                    else:
                        st.info("No pose detected. Stand in front of the camera and perform a yoga pose.")

                time.sleep(0.05)  # Control frame rate
                
        except Exception as e:
            st.error(f"Error during camera processing: {str(e)}")
            logger.error(f"Camera processing error: {str(e)}")
            stop_camera()
    
    elif not st.session_state.camera_running:
        camera_status.info("🔴 Camera Stopped")
        st.info("Click 'Start Camera' to begin detection.")

    # Instructions
    st.markdown("---")
    st.subheader("📝 Instructions")
    st.markdown("""
    1. **Select** your desired camera resolution (lower = faster)
    2. **Click** 'Start Camera' to begin detection
    3. **Stand** in front of the camera and perform yoga poses
    4. **View** the detected pose name and confidence score
    5. **Stop** camera when finished to free resources
    
    **Tips:**
    - Ensure good lighting for better detection
    - Keep your full body in frame
    - Hold poses for a few seconds for better recognition
    """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        - **Camera not opening:** Try different resolution settings or restart the app
        - **Low accuracy:** Ensure good lighting and clear pose visibility
        - **App running slowly:** Use lower resolution (320x240) for better performance
        - **Models not loading:** Check that `yoga-model.h5` and `yolov8n.pt` are in the same directory
        """)

if __name__ == "__main__":
    main()
