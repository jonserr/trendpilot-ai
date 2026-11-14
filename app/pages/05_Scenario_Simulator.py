import requests
import streamlit as st

from app.api_client import API_URL, api_headers
from app.data_cache import load_engineered_dataset, missing_data_message
from src.config import get_config

st.set_page_config(page_title="Scenario Simulator", page_icon="⚙️")

st.title("⚙️ Scenario Simulator")

cfg = get_config()

st.markdown(
    "Adjust driver features to see how the predicted "
    f"`{cfg.target_col}` changes for a `{cfg.entity_col}`."
)

df = load_engineered_dataset()

if df is None:
    st.warning(missing_data_message(cfg))
    st.stop()

st.subheader("Entity")
hierarchy = [c for c in cfg.entity_hierarchy if c in df.columns]
sub = df
cols = st.columns(len(hierarchy))
categorical_values: dict = {}
for i, col in enumerate(hierarchy):
    options = sorted(sub[col].unique().tolist())
    with cols[i]:
        chosen = st.selectbox(col.title(), options)
    sub = sub[sub[col] == chosen]
    categorical_values[col] = str(chosen)
latest = sub.iloc[-1]
for col in cfg.categorical_features:
    categorical_values.setdefault(col, str(latest[col]))

base_features = {
    **categorical_values,
    **{
        col: float(latest[col])
        for col in (
            *cfg.numeric_features,
            *cfg.lag_cols,
            *cfg.rolling_cols,
            "month",
            "quarter",
            "year",
        )
    },
}
base_features["month"] = int(base_features["month"])
base_features["quarter"] = int(base_features["quarter"])
base_features["year"] = int(base_features["year"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Base Assumptions")
    st.caption("Taken from the latest observation for the selected entity.")
    st.json({k: base_features[k] for k in cfg.numeric_features})

with col2:
    st.subheader("Scenario Adjustments")
    st.info("These adjustments overwrite the base assumptions for the scenario prediction.")
    adjust_cols = st.multiselect(
        "Features to adjust",
        list(cfg.numeric_features),
        default=list(cfg.numeric_features)[:2],
    )
    adjustments: dict = {}
    for col in adjust_cols:
        adjustments[col] = st.number_input(
            f"Adjusted {col.replace('_', ' ').title()}",
            value=float(base_features[col]),
            format="%.4f",
        )

if st.button("Simulate Scenario", type="primary"):
    base_payload = {"features": base_features}
    scenario_payload = {**base_payload, "scenario_adjustments": adjustments}

    try:
        with st.spinner("Running base and scenario predictions…"):
            base_res = requests.post(
                f"{API_URL}/predict", json=base_payload, timeout=30, headers=api_headers()
            )
            scenario_res = requests.post(
                f"{API_URL}/predict", json=scenario_payload, timeout=30, headers=api_headers()
            )

        if base_res.status_code == 200 and scenario_res.status_code == 200:
            base_value = base_res.json()["predicted_value"]
            scenario_value = scenario_res.json()["predicted_value"]
            unit = cfg.target_unit or ""

            st.markdown(f"### Simulation Results — {categorical_values[cfg.entity_col]}")
            c1, c2, _ = st.columns(3)
            c1.metric("Base Prediction", f"{unit}{base_value:,.2f}")
            c2.metric(
                "Scenario Prediction",
                f"{unit}{scenario_value:,.2f}",
                delta=f"{unit}{scenario_value - base_value:,.2f} impact",
            )
        else:
            st.error("API Error during simulation.")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
