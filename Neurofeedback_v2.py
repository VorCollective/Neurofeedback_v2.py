import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import tempfile
import scipy.io as sio
import mne
from PIL import Image
import fitz  # PyMuPDF
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import random
import threading
import time
from collections import deque

# Robust pylsl import (works across pylsl builds)
try:
    import pylsl
except Exception as e:
    raise ImportError("pylsl is required for Live Muse EEG mode. Install with `pip install pylsl`.") from e

# Try to find a resolver function on the pylsl module
resolve_fn = getattr(pylsl, "resolve_stream", None) \
         or getattr(pylsl, "resolve_streams", None) \
         or getattr(pylsl, "resolve_byprop", None)

if resolve_fn is None:
    raise ImportError(
        "Could not find a stream resolver in pylsl. "
        "Ensure pylsl is installed from the official LabStreamingLayer pylsl package."
    )

# --- Global class names ---
class_names = ["Focused", "Distracted", "Fatigued", "Stressed", "Other"]

# --- Load models (files available in working Dir) ---
rf = joblib.load("rf_model.pkl")
xgb_clf = joblib.load("xgb_model.pkl")
eegnet = load_model("eegnet_model.h5")

# --- Session state initialization ---
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "rec_log" not in st.session_state:
    st.session_state.rec_log = []  # list of dicts: {"timestamp","state","rec"}
if "muse_thread" not in st.session_state:
    st.session_state.muse_thread = None
if "muse_stop_event" not in st.session_state:
    st.session_state.muse_stop_event = None
if "buffer" not in st.session_state:
    st.session_state.buffer = deque()
if "fs" not in st.session_state:
    st.session_state.fs = None
if "expected_samples" not in st.session_state:
    st.session_state.expected_samples = int(eegnet.input_shape[2])

# --- Recommendation variants ---
variants = {
    "Focused": [
        "You’re in the zone — keep riding that wave!",
        "Laser sharp! Stay on task and push through.",
        "Momentum is on your side — keep building."
    ],
    "Distracted": [
        "Mind drift detected. Let’s reel it back in.",
        "Too many tabs open? Simplify and refocus.",
        "Pause notifications — reclaim your attention."
    ],
    "Fatigued": [
        "Energy dip spotted. Hydrate and stretch.",
        "Your body’s flagging — take a 5‑minute reset.",
        "Recharge time: step away and breathe."
    ],
    "Stressed": [
        "Stress spike. Inhale deeply, exhale slowly.",
        "Reset your rhythm with a short walk.",
        "Pause — mindfulness beats overload."
    ],
    "Other": [
        "Something feels off. Adjust your environment.",
        "Experiment with posture or lighting.",
        "Take a reset break — recalibrate."
    ]
}

def adaptive_recommendation(predicted_state):
    """Return a varied recommendation and log it in session_state.rec_log."""
    st.session_state.timeline.append(predicted_state)
    counts = {state: st.session_state.timeline.count(state) for state in class_names}
    if predicted_state == "Fatigued" and counts["Fatigued"] >= 5:
        rec = "Frequent fatigue detected. Consider long-term strategies: sleep hygiene, nutrition, workload balance."
    elif predicted_state == "Stressed" and counts["Stressed"] >= 5:
        rec = "Frequent stress detected. Try structured relaxation routines or workload adjustments."
    elif predicted_state == "Distracted" and counts["Distracted"] >= 5:
        rec = "Repeated distraction detected. Consider environment changes: reduce notifications, set focus hours."
    else:
        rec = random.choice(variants[predicted_state])

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.rec_log.insert(0, {"timestamp": ts, "state": predicted_state, "rec": rec})
    # Keep log bounded
    if len(st.session_state.rec_log) > 1000:
        st.session_state.rec_log = st.session_state.rec_log[:1000]
    return rec

# --- Sidebar feed (sidebar-only recommendations) ---
def show_recommendation_feed(predicted_states):
    st.sidebar.title("🌀 Recommendation Feed")

    gradients = {
        "Focused": "linear-gradient(135deg, #4CAF50, #81C784)",
        "Distracted": "linear-gradient(135deg, #2196F3, #64B5F6)",
        "Fatigued": "linear-gradient(135deg, #FF9800, #FFB74D)",
        "Stressed": "linear-gradient(135deg, #F44336, #E57373)",
        "Other": "linear-gradient(135deg, #9E9E9E, #BDBDBD)"
    }

    icons = {
        "Focused": "🎯 Stay sharp",
        "Distracted": "🔕 Refocus",
        "Fatigued": "💧 Recharge",
        "Stressed": "🌿 Calm down",
        "Other": "⚙️ Reset"
    }

    st.sidebar.markdown(
        """
        <style>
        .fade-in { animation: fadeIn 0.8s ease-in; }
        @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
        .css-1d391kg { overflow-y: auto !important; scroll-behavior: smooth; }
        </style>
        """,
        unsafe_allow_html=True
    )

    for state in predicted_states[:10]:
        rec = adaptive_recommendation(state)
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.sidebar.markdown(
            f"""
            <div class="fade-in" style="background:{gradients[state]}; 
                 padding:12px; border-radius:10px; margin-bottom:12px; color:white;">
                <h4 style="margin:0;">{icons[state]} — {timestamp}</h4>
                <p style="font-size:14px; line-height:1.4; margin:6px 0 0 0;">{rec}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Download recommendations log
    if st.sidebar.button("Download recommendations log as CSV"):
        if st.session_state.rec_log:
            df_log = pd.DataFrame(st.session_state.rec_log)
            csv = df_log.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button("Download CSV", data=csv, file_name="recommendations_log.csv", mime="text/csv")
        else:
            st.sidebar.info("No recommendations yet to download.")

# --- Dashboard plotting ---
def plot_dashboard(timeline, window=10, smooth_sigma=4):
    if not timeline:
        st.info("No predictions yet. Upload a file or start Live Muse EEG.")
        return

    numeric_states = [class_names.index(s) for s in timeline]
    df = pd.DataFrame({"State": timeline, "Index": range(len(timeline))})
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Timeline
    axes[0].plot(df["Index"], numeric_states, marker='o', linestyle='-', color='purple')
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_yticklabels(class_names)
    axes[0].set_title("Predicted Mental State Timeline")

    # Frequency
    counts = df["State"].value_counts()
    axes[1].bar(counts.index, counts.values, color="teal")
    axes[1].set_title("Frequency of States")

    # Trend with smoothing
    if len(numeric_states) >= 2:
        window = min(window, max(3, len(numeric_states)//2))
        rolling = pd.Series(numeric_states).rolling(window).mean()
        axes[2].plot(df["Index"], rolling, color="black", linewidth=2, label=f"Moving Avg (window={window})")

        smoothed = gaussian_filter1d(numeric_states, sigma=smooth_sigma)
        axes[2].plot(df["Index"], smoothed, color="red", linestyle="--", label=f"Smoothed Trend (σ={smooth_sigma})")

        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "Not enough data for trend",
                     ha="center", va="center", fontsize=12)

    axes[2].set_yticks(range(len(class_names)))
    axes[2].set_yticklabels(class_names)
    axes[2].set_title("Longer Trend Over Period")

    st.pyplot(fig)

# --- Muse LSL reader (background thread) ---
def muse_reader(stop_event, epoch_seconds=1.0, max_buffer_seconds=10.0):
    """
    Background thread that resolves an EEG LSL stream, pulls samples,
    accumulates them into epochs, and runs predictions.
    """
    try:
        # Resolve EEG stream (blocks until found or times out)
        # Use the resolver function we found earlier
        streams = resolve_fn('type', 'EEG') if resolve_fn.__name__ != 'resolve_byprop' else resolve_fn('type', 'EEG')
        if not streams:
            # No streams found
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.rec_log.insert(0, {"timestamp": ts, "state": "Error", "rec": "No EEG LSL stream found. Start muselsl stream first."})
            return

        inlet = pylsl.StreamInlet(streams[0], max_chunklen=256)
        info = inlet.info()
        fs = int(info.nominal_srate()) if info.nominal_srate() > 0 else 256
        st.session_state.fs = fs
        expected_samples = st.session_state.expected_samples

        # Buffer to accumulate samples (each sample is array of channels)
        sample_buffer = deque()

        # Convert epoch_seconds to sample count
        epoch_samples = int(epoch_seconds * fs)
        max_buffer_samples = int(max_buffer_seconds * fs)

        while not stop_event.is_set():
            # Pull a chunk (non-blocking with timeout)
            chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=512)
            if chunk:
                for s in chunk:
                    sample_buffer.append(np.array(s))
                # Trim buffer if too large
                while len(sample_buffer) > max_buffer_samples:
                    sample_buffer.popleft()

            # If we have enough samples for an epoch, take the most recent epoch_samples
            if len(sample_buffer) >= epoch_samples:
                # Build epoch from the most recent epoch_samples
                epoch_list = [sample_buffer.popleft() for _ in range(epoch_samples)]
                epoch_array = np.vstack(epoch_list)  # shape (epoch_samples, n_channels)

                # Preprocess: average across channels to single channel (simple approach)
                single_channel = np.mean(epoch_array, axis=1)  # shape (epoch_samples,)

                # Adjust length to expected_samples
                if len(single_channel) >= expected_samples:
                    single_channel = single_channel[:expected_samples]
                else:
                    pad_len = expected_samples - len(single_channel)
                    single_channel = np.concatenate([single_channel, np.zeros(pad_len)])

                # Prepare for EEGNet: (1,1,expected_samples,1)
                X_eegnet = single_channel.reshape((1, 1, expected_samples, 1)).astype(np.float32)

                # Predict with EEGNet (fallback to RF/XGB if needed)
                try:
                    y_pred = eegnet.predict(X_eegnet, verbose=0).argmax(axis=1)[0]
                    state = class_names[int(y_pred)]
                except Exception:
                    feat = np.array([np.mean(single_channel), np.std(single_channel)]).reshape(1, -1)
                    try:
                        y_pred_rf = rf.predict(feat)[0]
                        state = class_names[int(y_pred_rf)]
                    except Exception:
                        state = "Other"

                # Log recommendation (this also appends to timeline)
                adaptive_recommendation(state)

            # small sleep to yield CPU
            time.sleep(0.01)

    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.rec_log.insert(0, {"timestamp": ts, "state": "Error", "rec": f"Muse reader error: {e}"})

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Neurofeedback EEG Mental State Recommender Dashboard")

# Mode selector
mode = st.radio("Choose input mode:", ["Upload File", "Live Muse EEG"])

# Show compact latest recommendations in sidebar
st.sidebar.markdown("### Latest recommendations")
for entry in st.session_state.rec_log[:6]:
    st.sidebar.markdown(f"- **{entry['state']}** — {entry['timestamp']}")

if mode == "Live Muse EEG":
    st.sidebar.header("Live Muse Controls")
    col1, col2 = st.sidebar.columns(2)
    start_clicked = col1.button("Start Live Stream")
    stop_clicked = col2.button("Stop Live Stream")

    if start_clicked:
        # Start background thread if not already running
        if st.session_state.muse_thread is None or not st.session_state.muse_thread.is_alive():
            stop_event = threading.Event()
            st.session_state.muse_stop_event = stop_event
            t = threading.Thread(target=muse_reader, args=(stop_event,), daemon=True)
            st.session_state.muse_thread = t
            t.start()
            st.sidebar.success("Muse reader started (background).")
        else:
            st.sidebar.info("Muse reader already running.")

    if stop_clicked:
        if st.session_state.muse_stop_event:
            st.session_state.muse_stop_event.set()
            st.session_state.muse_thread = None
            st.sidebar.info("Stopping Muse reader...")

    st.subheader("Live Dashboard")
    # Manual refresh button to re-run and show latest timeline
    if st.button("Refresh Dashboard"):
        pass
    plot_dashboard(st.session_state.timeline, window=10)

    # Show full recommendation log in main panel (optional)
    if st.checkbox("Show full recommendation log"):
        df_log = pd.DataFrame(st.session_state.rec_log)
        st.dataframe(df_log)

else:
    # --- File upload mode ---
    uploaded_file = st.file_uploader(
        "Upload EEG or document/image file",
        type=["mat", "edf", "npy", "png", "jpg", "jpeg", "pdf"]
    )

    if uploaded_file is not None:
        suffix = uploaded_file.name.split(".")[-1].lower()

        if suffix == "npy":
            try:
                X_feat = np.load(uploaded_file)
                y_pred_rf = rf.predict(X_feat)
                y_pred_xgb = xgb_clf.predict(X_feat)
                states_rf = [class_names[i] for i in y_pred_rf[:10]]
                states_xgb = [class_names[i] for i in y_pred_xgb[:10]]
                show_recommendation_feed(states_rf + states_xgb)
            except Exception as e:
                st.error(f"Failed to process NPY file: {e}")

        elif suffix == "mat":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmpfile:
                    tmpfile.write(uploaded_file.read())
                    tmp_path = tmpfile.name
                mat = sio.loadmat(tmp_path)
                st.write("MAT file uploaded")
                expected_samples = st.session_state.expected_samples
                if "o" in mat:
                    try:
                        o_struct = mat["o"][0,0]
                        eeg_data = np.asarray(o_struct["data"])
                        X_raw = eeg_data[:,0]
                        if len(X_raw) >= expected_samples:
                            X_raw = X_raw[:len(X_raw)//expected_samples*expected_samples].reshape(-1, expected_samples)
                            X_eegnet = X_raw.reshape((X_raw.shape[0], 1, X_raw.shape[1], 1))
                            y_pred_eegnet = eegnet.predict(X_eegnet).argmax(axis=1)
                            states_eegnet = [class_names[i] for i in y_pred_eegnet[:10]]
                            show_recommendation_feed(states_eegnet)
                        else:
                            st.warning(f"MAT file too short for expected {expected_samples}-sample epochs.")
                    except Exception as e:
                        st.error(f"MAT file structure found but could not extract EEG data: {e}")
                else:
                    st.warning("No 'o' key found in MAT file.")
            except Exception as e:
                st.error(f"Failed to load MAT file: {e}")

        elif suffix == "edf":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmpfile:
                    tmpfile.write(uploaded_file.read())
                    tmp_path = tmpfile.name
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                st.write(f"EDF file uploaded: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")
                st.line_chart(raw.get_data()[0][:1000])

                expected_samples = st.session_state.expected_samples
                data = raw.get_data()[0]
                if len(data) >= expected_samples:
                    X_raw = data[:len(data)//expected_samples*expected_samples].reshape(-1, expected_samples)
                    X_eegnet = X_raw.reshape((X_raw.shape[0], 1, X_raw.shape[1], 1))
                    y_pred_eegnet = eegnet.predict(X_eegnet).argmax(axis=1)
                    states_eegnet = [class_names[i] for i in y_pred_eegnet[:10]]
                    show_recommendation_feed(states_eegnet)
                else:
                    st.warning(f"EDF file too short for expected {expected_samples}-sample epochs.")
            except Exception as e:
                st.error(f"Failed to load EDF file: {e}")

        elif suffix in ["png", "jpg", "jpeg"]:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to display image: {e}")

        elif suffix == "pdf":
            try:
                pdf_bytes = uploaded_file.read()
                pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                st.write(f"PDF uploaded with {pdf.page_count} pages")
                for page_num in range(min(3, pdf.page_count)):
                    page = pdf[page_num]
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption=f"Page {page_num+1}", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to load PDF file: {e}")

        else:
            st.error("Unsupported file format")

    # Show dashboard if timeline has data
    if st.session_state.timeline:
        st.subheader("Dashboard")
        plot_dashboard(st.session_state.timeline, window=10)
