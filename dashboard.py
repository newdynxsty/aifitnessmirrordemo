import streamlit as st
import serial
import threading
import queue
import time
import platform

# --- CONFIGURATION ---
if platform.system() == "Windows":
    SERIAL_PORT = 'COM7'
else:
    SERIAL_PORT = '/dev/cu.usbmodem102'
BAUD_RATE = 115200

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

# Exercise name
st.metric(label="Current Exercise", value=data["exercise"])
st.caption(f"Confidence: {data['conf']*100:.0f}%")

# Rep count
st.metric(label="Rep Count", value=data["reps"])
st.markdown('<p class="guidance-header">Form Guidance</p>', unsafe_allow_html=True)

# --- FEEDBACK AREA ---

if "Perfect" in data["error"]:
    st.success(data["error"])
elif data["error"] in ["NONE", "WAITING...", "NO POSE"]:
    st.info(data["error"])
else:
    st.error(f"{data["error"]}")

st.caption(f"Confidence: {data['error_conf']*100:.0f}%")

# Auto-refresh
time.sleep(0.4)
st.rerun()