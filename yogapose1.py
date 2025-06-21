import streamlit as st
import cv2
import numpy as np
import time

# Try importing TensorFlow, fallback to TensorFlow Lite if needed
try:
    import tensorflow as tf
    USE_TFLITE = False
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        USE_TFLITE = True
    except ImportError:
        st.error("Neither TensorFlow nor TensorFlow Lite runtime found. Please install one of them.")
        st.stop()

from ultralytics import YOLO

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

# --- Load Models (Cached for Efficiency) ---
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO('yolov8n.pt')
        
        if USE_TFLITE:
            # Load TensorFlow Lite model
            interpreter = tflite.Interpreter(model_path='yoga-model.tflite')
            interpreter.allocate_tensors()
            yoga_model = interpreter
        else:
            # Load full TensorFlow model
            yoga_model = tf.keras.models.load_model('yoga-model.h5')
            
        return yolo_model, yoga_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure 'yolov8n.pt' and 'yoga-model.h5' (or 'yoga-model.tflite') files are in your project directory.")
        return None, None

# --- Image Preprocessing Function ---
def preprocess_image(image):
    target_size = (64, 64)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    if USE_TFLITE:
        img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# --- Predict Yoga Pose Function ---
def predict_yoga_pose(model, image):
    processed_img = preprocess_image(image)
    
    if USE_TFLITE:
        # TensorFlow Lite inference
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        model.set_tensor(input_details[0]['index'], processed_img)
        model.invoke()
        prediction = model.get_tensor(output_details[0]['index'])
    else:
        # Full TensorFlow inference
        prediction = model.predict(processed_img, verbose=0)
    
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class_index, confidence

# --- Detect & Classify Persons in Frame ---
def detect_and_classify_poses(image, yolo_model, yoga_model):
    results = yolo_model(image, verbose=False)
    annotated_image = image.copy()
    detected_poses = []

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls == 0 and conf > 0.5:  # Person class
                    cropped_img = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                    if cropped_img.size > 0:
                        pose_idx, pose_conf = predict_yoga_pose(yoga_model, cropped_img)

                        # Safety Check to avoid IndexError
                        if 0 <= pose_idx < len(yoga_poses):
                            pose_name = yoga_poses[pose_idx]
                        else:
                            pose_name = "unknown"

                        # Draw bounding box and label
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{pose_name} ({pose_conf:.2f})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        detected_poses.append({'pose': pose_name, 'confidence': pose_conf})

    return annotated_image, detected_poses

# --- Camera Control Functions ---
def start_camera(resolution):
    width, height = resolution
    if st.session_state.cap is None:
        indices_to_try = [0, 1, 2, -1]
        found = False
        for index in indices_to_try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                st.session_state.cap = cap
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
                found = True
                break
            cap.release()
        if not found:
            st.error("Failed to open any camera device. Camera may not be available in this environment.")
            return False
    if st.session_state.cap.isOpened():
        st.session_state.camera_running = True
        return True
    else:
        st.error("Could not open camera.")
        stop_camera()
        return False

def stop_camera():
    if st.session_state.cap:
        st.session_state.cap.release()
    st.session_state.cap = None
    st.session_state.camera_running = False

# --- Main App Logic ---
def main():
    st.title("🧘‍♀️ Real-time Yoga Pose Detection")
    st.markdown("Click 'Start Camera' to begin real-time detection.")

    # Display current setup
    if USE_TFLITE:
        st.info("🚀 Running with TensorFlow Lite (optimized for deployment)")
    else:
        st.info("🔥 Running with full TensorFlow")

    # Sidebar Info
    st.sidebar.title("About")
    st.sidebar.info("Detects yoga poses using YOLO and a custom CNN model.")
    st.sidebar.title("Supported Poses")
    with st.sidebar.expander("View all 107 poses"):
        for i, pose in enumerate(yoga_poses[:107], 1):
            st.write(f"{i}. {pose.title()}")

    # Load Models
    with st.spinner("Loading AI models..."):
        yolo_model, yoga_model = load_models()
    if yolo_model is None or yoga_model is None:
        st.stop()

    # Camera Resolution Selection
    st.subheader("📹 Camera Settings")
    resolution_options = {
        "320x240": (320, 240),
        "640x480": (640, 480),
        "1280x720": (1280, 720)
    }
    selected_resolution = st.selectbox("Select Camera Resolution", 
                                     options=list(resolution_options.keys()), index=1)

    # Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Start Camera", key="start_button"):
            if start_camera(resolution_options[selected_resolution]):
                st.success("Camera started successfully!")
    with col2:
        if st.button("⏹️ Stop Camera", key="stop_button"):
            stop_camera()
            st.success("Camera stopped.")

    # Video Feed
    video_placeholder = st.empty()
    poses_placeholder = st.empty()

    if st.session_state.camera_running and st.session_state.cap:
        frame_count = 0
        while st.session_state.camera_running:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning("Failed to read from camera.")
                stop_camera()
                break

            frame = cv2.flip(frame, 1)
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                # Resize for faster inference
                small_frame = cv2.resize(frame, (320, 240))
                annotated_frame, detected_poses = detect_and_classify_poses(small_frame, yolo_model, yoga_model)

                # Resize back to original size for display
                display_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame, width=700)

                # Display pose info
                with poses_placeholder.container():
                    st.subheader("📊 Detected Poses")
                    if detected_poses:
                        for i, p in enumerate(detected_poses, 1):
                            col1, col2, col3 = st.columns([2, 1, 1])
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
                        st.info("No pose detected.")
            
            frame_count += 1
            time.sleep(0.03)  # Reduce CPU usage

    elif not st.session_state.camera_running:
        st.info("Click 'Start Camera' to begin detection.")

    # Instructions
    st.markdown("---")
    st.subheader("📝 Instructions:")
    st.markdown("""
    1. Select desired **camera resolution**
    2. Click **Start Camera**
    3. Perform yoga poses in front of the camera
    4. The app will show detected pose name and confidence
    
    **Note:** Camera access may not work in GitHub Codespaces or similar hosted environments.
    For best results, run locally or use file upload instead.
    """)

if __name__ == "__main__":
    main()
