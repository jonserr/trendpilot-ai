import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Model Evaluation", page_icon="🎯")

st.title("🎯 Model Evaluation")


@st.cache_data
def fetch_metadata():
    try:
        response = requests.get(f"{API_URL}/model-metadata", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


metadata = fetch_metadata()

if metadata:
    st.subheader(f"Model: {metadata.get('model_type', 'Unknown')}")
    st.write(f"Trained on: {metadata.get('training_date', 'Unknown')}")
    st.write(
        f"Dataset: `{metadata.get('dataset_name', 'unknown')}` — "
        f"target `{metadata.get('target_column', 'unknown')}`"
    )

    st.markdown("### Performance Comparison")

    metrics = metadata.get("metrics", {})
    baseline_metrics = metadata.get("baseline_metrics", {})
    unit = metadata.get("target_unit", "") or ""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Keras MAE",
            f"{unit}{metrics.get('mae', 0):,.2f}",
            delta=f"{baseline_metrics.get('mae', 0) - metrics.get('mae', 0):,.2f} vs Baseline",
        )
    with col2:
        st.metric(
            "Keras RMSE",
            f"{unit}{metrics.get('rmse', 0):,.2f}",
            delta=f"{baseline_metrics.get('rmse', 0) - metrics.get('rmse', 0):,.2f} vs Baseline",
        )
    with col3:
        st.metric(
            "Keras MAPE",
            f"{metrics.get('mape', 0):.2f}%",
            delta=f"{baseline_metrics.get('mape', 0) - metrics.get('mape', 0):.2f}% vs Baseline",
        )
    with col4:
        st.metric(
            "Keras R²",
            f"{metrics.get('r2', 0):.4f}",
            delta=f"{metrics.get('r2', 0) - baseline_metrics.get('r2', 0):.4f} vs Baseline",
        )

    st.markdown("---")
    st.subheader("Model Input Columns")
    st.write(metadata.get("feature_columns", []))

else:
    st.warning("Model metadata is not available. Please train the model first.")
