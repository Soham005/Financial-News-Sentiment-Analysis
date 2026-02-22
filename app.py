"""Streamlit app for FinBERT + LSTM hybrid inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference import load_or_init_artifacts, predict_probability  # noqa: E402


st.set_page_config(page_title="FinBERT + LSTM Stock Movement", layout="wide")
st.title("Financial News Sentiment Driven Stock Movement Prediction")

st.markdown("Run interactive inference with optional checkpoint loading.")

checkpoint_path = st.text_input(
    "Checkpoint path (optional)", value="artifacts/finbert_lstm.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=True)
def get_artifacts(path: str):
    return load_or_init_artifacts(path, device=device)

artifacts = get_artifacts(checkpoint_path)
if not Path(checkpoint_path).exists():
    st.info("No checkpoint found; using default initialized model weights.")

st.caption(
    f"Running on {device.type.upper()} | seq_len={artifacts.seq_len} | "
    f"features={list(artifacts.market_feature_cols)}"
)

news_text = st.text_area(
    "News text",
    value="Company reports strong quarterly earnings and raises forward guidance.",
    height=120,
)

columns = list(artifacts.market_feature_cols)
default_value = {col: 100.0 for col in columns}
default_df = pd.DataFrame([default_value] * artifacts.seq_len)

st.subheader("Market sequence input")
market_df = st.data_editor(
    default_df,
    num_rows="fixed",
    use_container_width=True,
    key="market_table",
)

if st.button("Predict movement probability", type="primary"):
    if market_df.shape[0] != artifacts.seq_len:
        st.error(f"Expected exactly {artifacts.seq_len} rows.")
        st.stop()

    try:
        rows = market_df[columns].astype(float).values.tolist()
    except Exception:
        st.error("All market feature values must be numeric.")
        st.stop()

    prob_up = predict_probability(
        artifacts=artifacts,
        market_rows=rows,
        news_text=news_text,
        device=device,
    )

    st.metric("Probability of upward movement", f"{prob_up:.2%}")
    st.write("Prediction:", "UP" if prob_up >= 0.5 else "DOWN")
