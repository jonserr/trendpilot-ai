import plotly.express as px
import streamlit as st

from app.data_cache import load_raw_dataset, missing_data_message
from src.config import get_config

st.set_page_config(page_title="Data Explorer", page_icon="📊")

st.title("📊 Data Explorer")

cfg = get_config()

df = load_raw_dataset()

if df is not None:
    hierarchy = [c for c in cfg.entity_hierarchy if c in df.columns]
    top_col = hierarchy[0]
    leaf_col = hierarchy[-1]

    col1, col2 = st.columns(2)
    with col1:
        top_values = sorted(df[top_col].unique().tolist())
        selected_top = st.selectbox(top_col.title(), ["All"] + top_values)
    with col2:
        if leaf_col != top_col:
            if selected_top != "All":
                leaves = sorted(df.loc[df[top_col] == selected_top, leaf_col].unique())
            else:
                leaves = sorted(df[leaf_col].unique())
            selected_leaf = st.selectbox(leaf_col.title(), ["All"] + list(leaves))
        else:
            selected_leaf = "All"

    plot_df = df
    if selected_top != "All":
        plot_df = plot_df[plot_df[top_col] == selected_top]
    if selected_leaf != "All":
        plot_df = plot_df[plot_df[leaf_col] == selected_leaf]

    st.subheader(f"{cfg.target_col.replace('_', ' ').title()} Trend")
    if selected_leaf == "All" and leaf_col != top_col:
        # One line per entity within the current selection
        fig = px.line(
            plot_df,
            x=cfg.date_col,
            y=cfg.target_col,
            color=leaf_col,
            title=f"Trend for {selected_top if selected_top != 'All' else 'all entities'}",
        )
    else:
        label = selected_leaf if selected_leaf != "All" else selected_top
        fig = px.line(plot_df, x=cfg.date_col, y=cfg.target_col, title=f"Trend for {label}")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Raw Data Preview")
    st.dataframe(plot_df.head(200))

    st.subheader("Descriptive Statistics")
    st.dataframe(plot_df.select_dtypes("number").describe())
else:
    st.warning(missing_data_message(cfg))
