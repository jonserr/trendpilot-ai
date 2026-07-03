import os
import time

import requests
import streamlit as st

from src.config import get_config

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_WAIT_ATTEMPTS = 10

st.set_page_config(page_title="Trendpilot AI", page_icon="📈", layout="wide")

cfg = get_config()

st.title("📈 Trendpilot AI Forecasting")
st.markdown(f"""
**Trendpilot AI** is an end-to-end machine learning forecasting platform built
with TensorFlow/Keras, FastAPI, and Streamlit. It is currently configured for
the **{cfg.name}** dataset (target: `{cfg.target_col}`, entity: `{cfg.entity_col}`).

### Application Overview
Use the sidebar to navigate through the workflow:
1. **Data Explorer**: Browse the dataset by entity.
2. **Model Training**: Trigger the training pipeline for the Keras neural network.
3. **Forecast Viewer**: Query the inference API for entity-level predictions.
4. **Model Evaluation**: Compare the Keras model against a scikit-learn baseline.
5. **Scenario Simulator**: Test "what-if" scenarios by adjusting driver features.
6. **Prediction History**: Review predictions logged to PostgreSQL (optional).
""")


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def wait_for_api():
    """Poll /health with a progress bar instead of failing on the first try —
    right after `make up` the API container may still be booting."""
    health = check_api_health()
    if health is not None:
        return health

    progress = st.progress(0, text="Waiting for the inference API to come online…")
    for attempt in range(1, API_WAIT_ATTEMPTS + 1):
        time.sleep(1)
        health = check_api_health()
        progress.progress(
            attempt / API_WAIT_ATTEMPTS,
            text=f"Waiting for the inference API… (attempt {attempt}/{API_WAIT_ATTEMPTS})",
        )
        if health is not None:
            break
    progress.empty()
    return health


st.subheader("System Health")
health_data = wait_for_api()

if health_data:
    st.success(f"✅ Inference API is ONLINE at `{API_URL}`")

    col1, col2 = st.columns(2)
    with col1:
        if health_data.get("model_loaded"):
            st.info("🧠 Model is loaded in memory and ready for inference.")
        elif health_data.get("model_loading"):
            st.info("⏳ Model is loading in the background — refresh in a few seconds.")
        else:
            st.warning(
                "⚠️ No trained model found. Train one on the **Model Training** page "
                "(or run `make train`) — the API picks it up automatically."
            )

    with col2:
        if health_data.get("metadata_exists"):
            st.info("📊 Model metadata is available.")
        else:
            st.warning("⚠️ Model metadata is missing.")
else:
    st.error(
        f"❌ Cannot connect to the inference API at `{API_URL}`. "
        "Ensure the service is running (`make up`), then retry."
    )
    if st.button("Retry"):
        st.rerun()
