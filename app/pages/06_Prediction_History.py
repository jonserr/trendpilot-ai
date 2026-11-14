import pandas as pd
import requests
import streamlit as st

from app.api_client import API_URL, api_headers
from src.config import get_config

st.set_page_config(page_title="Prediction History", page_icon="🗄️")

st.title("🗄️ Prediction History")

cfg = get_config()

st.markdown(
    "Recent predictions logged to PostgreSQL. Requires the optional `DATABASE_URL` "
    "configuration — see `.env.example`."
)

col1, col2 = st.columns(2)
with col1:
    limit = st.slider("Number of records", 10, 200, 50, step=10)
with col2:
    entity_filter = st.text_input(f"Filter by {cfg.entity_col} (optional)")

params = {"limit": limit}
if entity_filter:
    params["entity_id"] = entity_filter

try:
    with st.spinner("Fetching prediction history…"):
        res = requests.get(
            f"{API_URL}/prediction-history", params=params, timeout=10, headers=api_headers()
        )
    if res.status_code != 200:
        st.error(f"API Error: {res.text}")
    else:
        data = res.json()
        if not data.get("enabled"):
            st.info(
                "Prediction logging is disabled. Set `DATABASE_URL` in your environment "
                "(see `.env.example`) and restart the API to enable it."
            )
        elif not data.get("records"):
            st.warning("Logging is enabled but no predictions have been recorded yet.")
        else:
            records = data["records"]
            unit = cfg.target_unit or ""
            df = pd.DataFrame(
                [
                    {
                        "Timestamp": r.get("created_at"),
                        "Endpoint": r.get("endpoint"),
                        "Entity": r.get("entity_id"),
                        "Horizon": r.get("horizon"),
                        f"Prediction ({unit})" if unit else "Prediction": (
                            r.get("prediction") or {}
                        ).get("predicted_value"),
                        "Model": r.get("model_version"),
                        "Duration (ms)": round(r.get("duration_ms") or 0, 2),
                    }
                    for r in records
                ]
            )
            st.dataframe(df, width="stretch")

            with st.expander("Raw records (request payloads, adjustments, warnings)"):
                st.json(records)
except Exception as e:
    st.error(f"Failed to connect to API: {e}")
