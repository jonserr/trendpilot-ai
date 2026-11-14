import os
import subprocess

import requests
import streamlit as st

from app.api_client import API_URL, api_headers
from app.data_cache import missing_data_message
from src.config import get_config

LOG_TAIL_LINES = 30

st.set_page_config(page_title="Model Training", page_icon="🧠")

st.title("🧠 Model Training")

cfg = get_config()

st.markdown("""
Trigger the training pipeline for the TensorFlow/Keras neural network
(`src/modeling/train.py`). Progress and logs stream live below; artifacts are
saved to the shared `models/` volume and the inference API picks up the new
model automatically — no restart needed.

One training or data-refresh run can run at a time. Wait for an existing run
to finish before starting another.
""")

data_available = cfg.data_path.exists()
if not data_available:
    st.warning(missing_data_message(cfg))

col1, col2 = st.columns(2)
with col1:
    epochs = st.number_input("Epochs", min_value=10, max_value=500, value=100, step=10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

with col2:
    learning_rate = st.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)

if st.button("Train Model", type="primary", disabled=not data_available):
    env = os.environ.copy()
    env["EPOCHS"] = str(epochs)
    env["BATCH_SIZE"] = str(batch_size)
    env["LEARNING_RATE"] = str(learning_rate)
    env["PYTHONUNBUFFERED"] = "1"  # stream logs line by line

    all_lines: list = []
    try:
        process = subprocess.Popen(
            ["python", "src/modeling/train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        with st.status("Training model… (live logs below)", expanded=True) as status:
            log_box = st.empty()
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                all_lines.append(line)
                log_box.code("\n".join(all_lines[-LOG_TAIL_LINES:]), language=None)
            return_code = process.wait()

            if return_code == 0:
                status.update(label="Training completed successfully ✅", state="complete")
            elif return_code == 2:
                status.update(label="Another training or data-refresh run is active", state="error")
            else:
                status.update(label="Training failed", state="error")

        if return_code == 0:
            st.success("Training complete — artifacts saved.")
            # Nudge the API so it reloads the new model in the background.
            try:
                requests.get(f"{API_URL}/health", timeout=3, headers=api_headers())
                st.info(
                    "The inference API is reloading the new model in the background — "
                    "the home page will show it as loaded in a few seconds."
                )
            except requests.exceptions.RequestException:
                st.info(
                    "Model published. The API will load it on its next readiness or health check."
                )
        elif return_code == 2:
            st.warning("Another training or data-refresh run is active. Retry after it finishes.")
        else:
            st.error("Training failed — see the logs above for the reason.")

        if all_lines:
            with st.expander("Full training logs"):
                st.code("\n".join(all_lines), language=None)
    except Exception as e:
        st.error(f"Error executing training script: {str(e)}")
