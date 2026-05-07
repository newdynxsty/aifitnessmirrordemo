import streamlit as st
import serial
import threading
import queue
import time
import platform

# --- CONFIGURATION ---
if platform.system() == "Windows":
    SERIAL_PORT = 'COM3'
else:
    SERIAL_PORT = '/dev/cu.usbmodem102'
BAUD_RATE = 115200

# --- FEEDBACK MAPPING ---
FORM_FEEDBACK = {
    "JJ ARM LOW": "Get your arms higher!",
    "JJ LEG NAR": "Jump with your legs wider!",
    "LUNGE LOW": "Lunge deeper!",
    "PUSH KNEE": "Knees off the ground!",
    "SIT CORE": "Get your core higher!",
    "SQUAT LOW": "Squat lower!",
    "JJ GOOD": "Perfect Jumping Jacks!",
    "LUNGE GOOD": "Great Lunge form!",
    "PUSH GOOD": "Solid Pushup!",
    "SIT GOOD": "Nice Situp!",
    "SQUAT GOOD": "Great Squat!",
    "WAITING...": "Waiting for exercise...",
    "NO POSE": "Step into view of the camera."
}

# --- STYLING ---
def local_css():
    st.markdown("""
        <style>
        [data-testid="stMetricLabel"] p, .guidance-header {
            font-size: 28px !important;
            font-weight: bold !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 100px !important;
        }
        .stAlert p {
            font-size: 36px !important;
            font-weight: bold !important;
        }        
        </style>
        """, unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "data_queue" not in st.session_state:
    st.session_state.data_queue = queue.Queue()

if 'workout_data' not in st.session_state:
    st.session_state.workout_data = {
        "exercise": "WAITING...", "conf": 0.0, "reps": 0, "error": "NONE", "error_conf": 0.0
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
        pass # Handle serial disconnects silently in background

if 'thread_started' not in st.session_state:
    thread = threading.Thread(target=read_serial, args=(st.session_state.data_queue,), daemon=True)
    thread.start()
    st.session_state.thread_started = True

# --- UPDATE DATA FROM QUEUE ---
while not st.session_state.data_queue.empty():
    st.session_state.workout_data = st.session_state.data_queue.get()

# --- DASHBOARD UI ---
st.set_page_config(page_title="NuvoMotion Dashboard", layout="wide")
local_css()

st.title("🏋️ NuvoMotion Dashboard")
st.divider()

data = st.session_state.workout_data

# Layout: Two large columns for Exercise and Reps
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Current Exercise", value=data["exercise"])
    st.caption(f"Confidence: {data['conf']*100:.0f}%")

with col2:
    st.metric(label="Rep Count", value=data["reps"])

st.markdown('<p class="guidance-header">Form Guidance</p>', unsafe_allow_html=True)

# --- FEEDBACK AREA ---
display_msg = FORM_FEEDBACK.get(data["error"], data["error"])

if "GOOD" in data["error"]:
    st.success(display_msg)
elif data["error"] in ["NONE", "WAITING...", "NO POSE"]:
    st.info(display_msg)
else:
    st.error(f"{display_msg}")

st.caption(f"Confidence: {data['error_conf']*100:.0f}%")

# Auto-refresh
time.sleep(0.4)
st.rerun()