import streamlit as st
import serial
import threading
import queue
import time
import platform
import cv2

# --- CONFIGURATION ---
if platform.system() == "Windows":
    SERIAL_PORT = 'COM7'
else:
    SERIAL_PORT = '/dev/cu.usbmodem2102'
BAUD_RATE = 115200

# --- STYLING ---
def local_css():
    st.markdown("""
        <style>
       .stText span {
            font-size: 7rem; 
        }
        .stAlert p {
            font-size: 3rem;
            font-weight: 600;
        }        
        </style>
        """, unsafe_allow_html=True)

import platform
import cv2
import streamlit as st

@st.cache_data(ttl=60)
def get_available_cameras(max_to_test=3):
    available_indices = []
    system = platform.system()

    for index in range(max_to_test):
        if system == "Windows":
            backend = cv2.CAP_DSHOW
        elif system == "Darwin":  # macOS
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_ANY

        # Attempt to open the camera with the backend
        cap = cv2.VideoCapture(index, backend)
        
        if cap.isOpened():
            if cap.grab():
                available_indices.append(index)
            cap.release()

    return available_indices

# --- STATE MANAGEMENT ---
if "data_queue" not in st.session_state:
    st.session_state.data_queue = queue.Queue()

if 'workout_data' not in st.session_state:
    st.session_state.workout_data = {
        "exercise": "WAITING..", "conf": 0.0, "reps": 0, "error": "NONE", "error_conf": 0.0
    }

# --- SERIAL READER THREAD ---
def read_serial(q):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("DATA:"):
                raw_payload = line.replace("DATA:", "")
                parts = raw_payload.split(",")
                if len(parts) == 5:
                    q.put({
                        "exercise": parts[0],
                        "conf": float(parts[1]),
                        "reps": int(parts[2]),
                        "error": parts[3],
                        "error_conf": float(parts[4])
                    })
    except Exception as e:
        pass

if 'thread_started' not in st.session_state:
    thread = threading.Thread(target=read_serial, args=(st.session_state.data_queue,), daemon=True)
    thread.start()
    st.session_state.thread_started = True

# --- UI RENDER HELPER FUNCTION --- 
def render_ui(data):
    exercise_name.text(data["exercise"], width='stretch')

    exercise_conf.caption(f"Confidence: {data['conf']*100:.0f}%")
    
    rep_count.text(data["reps"])
        
    if "Perfect" in data["error"]:
        feedback.success(data["error"])
    elif data["error"] in ["NONE", "WAITING...", "NO POSE"]:
        feedback.info(data["error"])
    else:
        feedback.error(f"{data['error']}")

    feedback_conf.caption(f"Confidence: {data['error_conf']*100:.0f}%")

# --- DASHBOARD UI ---
st.set_page_config(page_title="NuvoMotion Dashboard", layout="wide")
local_css()

st.title("🏋️ NuvoMotion Dashboard")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

# Left col: exercise data
with col_left:
    st.subheader("Current Exercise")
    exercise_name = st.empty()
    exercise_conf = st.empty()
    
    st.subheader("Rep Count")
    rep_count = st.empty()

    st.subheader("Form Guidance")
    feedback = st.empty()
    feedback_conf = st.empty()

# Right col: camera feed
with col_right:
    st.subheader("Live Camera Feed")
    
    # Camera selection UI
    connected_cameras = get_available_cameras()
    dropdown_options = ["None / Turned Off"] + [f"Camera {idx}" for idx in connected_cameras]

    selected_option = st.selectbox(
        "Select your camera:",
        options=dropdown_options,
        index=0,
        key="camera_selector"
    )

    camera_placeholder = st.empty()

# --- HIGH-SPEED EXECUTION LOOP ---
if selected_option != "None / Turned Off":
    target_idx = int(selected_option.split(" ")[1])
    
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(target_idx, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while st.session_state.camera_selector == selected_option:
            while not st.session_state.data_queue.empty():
                st.session_state.workout_data = st.session_state.data_queue.get()
            
            render_ui(st.session_state.workout_data)

            # Stream Video Frames (This updates constantly every loop cycle)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", width=800)
            
            time.sleep(0.02)  # Yield CPU bound processing overhead
            
    finally:
        cap.release()
else:
    while True:
        while not st.session_state.data_queue.empty():
            st.session_state.workout_data = st.session_state.data_queue.get()
            
        render_ui(st.session_state.workout_data)
            
        time.sleep(0.3)
        
        if st.session_state.camera_selector != "None / Turned Off":
            st.rerun()
