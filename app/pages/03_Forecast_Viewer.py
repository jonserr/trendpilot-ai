import os

import requests
import streamlit as st

from app.data_cache import load_engineered_dataset
from src.config import get_config

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Forecast Viewer", page_icon="📈")

st.title("📈 Forecast Viewer")

cfg = get_config()

st.markdown(
    f"Select a `{cfg.entity_col}` and request a prediction from the inference API. "
    "Historical lag features are pre-filled from the latest engineered observation."
)

df = load_engineered_dataset()

categorical_values: dict = {}
latest = None

if df is not None:
    hierarchy = [c for c in cfg.entity_hierarchy if c in df.columns]
    sub = df
    cols = st.columns(len(hierarchy))
    for i, col in enumerate(hierarchy):
        options = sorted(sub[col].unique().tolist())
        with cols[i]:
            chosen = st.selectbox(col.title(), options)
        sub = sub[sub[col] == chosen]
        categorical_values[col] = str(chosen)
    latest = sub.iloc[-1]
    for col in cfg.categorical_features:
        categorical_values.setdefault(col, str(latest[col]))
else:
    st.info("Dataset not found — enter entity and history manually.")
    for col in dict.fromkeys((*cfg.entity_hierarchy, *cfg.categorical_features)):
        categorical_values[col] = st.text_input(col.title(), value="")


def default(col, fallback=0.0):
    return float(latest[col]) if latest is not None else fallback


st.subheader("Driver Features")
numeric_inputs: dict = {}
grid = st.columns(2)
for i, col in enumerate(cfg.numeric_features):
    with grid[i % 2]:
        numeric_inputs[col] = st.number_input(
            col.replace("_", " ").title(), value=default(col), format="%.4f"
        )

st.subheader("Historical Context (Leakage-Safe Lag Features)")
history_cols = [*cfg.lag_cols, *cfg.rolling_cols]
grid = st.columns(3)
for i, col in enumerate(history_cols):
    label = col.replace(f"{cfg.target_col}_", "").replace("_", " ").title()
    with grid[i % 3]:
        numeric_inputs[col] = st.number_input(label, value=default(col), format="%.2f")

c1, c2, c3 = st.columns(3)
with c1:
    month = st.selectbox("Month", list(range(1, 13)), index=int(default("month", 6)) - 1)
with c2:
    year = st.number_input("Year", value=int(default("year", 2026)), step=1)
with c3:
    horizon = st.number_input("Forecast Horizon (periods)", 1, 24, 1)
quarter = (month - 1) // 3 + 1

if st.button("Get Forecast", type="primary"):
    payload = {
        "features": {
            **categorical_values,
            **numeric_inputs,
            "month": month,
            "quarter": quarter,
            "year": int(year),
        },
        "horizon": int(horizon),
    }

    try:
        with st.spinner("Requesting forecast from the inference API…"):
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            unit = cfg.target_unit or ""
            st.success(
                f"### Predicted {data['target_column'].replace('_', ' ').title()} "
                f"({data['entity_id']}): {unit}{data['predicted_value']:,.2f}"
            )
            st.info(f"Model Version: {data['model_version']}")
            if data.get("warnings"):
                st.warning(" / ".join(data["warnings"]))
        else:
            st.error(f"API Error: {res.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
