"""Inference utilities for FinBERT + LSTM hybrid model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from data import SequenceConfig
from model import FinBERTLSTMHybrid


@dataclass
class InferenceArtifacts:
    model: FinBERTLSTMHybrid
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    sequence_config: SequenceConfig


def load_artifacts(checkpoint_path: str | Path, device: torch.device) -> InferenceArtifacts:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = SequenceConfig(
        seq_len=int(checkpoint["seq_len"]),
        market_feature_cols=tuple(checkpoint["market_feature_cols"]),
        target_col=str(checkpoint.get("target_col", "target")),
    )

    model = FinBERTLSTMHybrid(
        finbert_model_name=checkpoint["finbert_model_name"],
        market_input_size=int(checkpoint["market_input_size"]),
        market_hidden_size=int(checkpoint["market_hidden_size"]),
        market_layers=int(checkpoint["market_layers"]),
        fusion_hidden_size=int(checkpoint["fusion_hidden_size"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return InferenceArtifacts(
        model=model,
        scaler_mean=np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
        scaler_scale=np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
        sequence_config=config,
    )


def prepare_market_window(
    market_rows: Sequence[Sequence[float]],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    market = np.asarray(market_rows, dtype=np.float32)
    scaled = (market - scaler_mean) / np.maximum(scaler_scale, 1e-8)
    return torch.tensor(scaled[None, :, :], dtype=torch.float32, device=device)


def predict_probability(
    artifacts: InferenceArtifacts,
    market_rows: Sequence[Sequence[float]],
    news_text: str,
    device: torch.device,
) -> float:
    market_tensor = prepare_market_window(
        market_rows=market_rows,
        scaler_mean=artifacts.scaler_mean,
        scaler_scale=artifacts.scaler_scale,
        device=device,
    )
    with torch.no_grad():
        logit = artifacts.model(market_tensor, [news_text])
        return float(torch.sigmoid(logit).item())
