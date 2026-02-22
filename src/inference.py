"""Inference utilities for FinBERT + LSTM hybrid model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from model import FinBERTLSTMHybrid


@dataclass
class InferenceArtifacts:
    model: FinBERTLSTMHybrid
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    seq_len: int
    market_feature_cols: tuple[str, ...]


def load_or_init_artifacts(checkpoint_path: str | Path, device: torch.device) -> InferenceArtifacts:
    model = FinBERTLSTMHybrid().to(device)
    seq_len = 10
    market_feature_cols = ("open", "high", "low", "close", "volume")
    scaler_mean = np.zeros(len(market_feature_cols), dtype=np.float32)
    scaler_scale = np.ones(len(market_feature_cols), dtype=np.float32)

    path = Path(checkpoint_path)
    if path.exists():
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            seq_len = int(checkpoint.get("seq_len", seq_len))
            market_feature_cols = tuple(checkpoint.get("market_feature_cols", market_feature_cols))
            scaler_mean = np.asarray(checkpoint.get("scaler_mean", scaler_mean), dtype=np.float32)
            scaler_scale = np.asarray(checkpoint.get("scaler_scale", scaler_scale), dtype=np.float32)

    model.eval()
    return InferenceArtifacts(
        model=model,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        seq_len=seq_len,
        market_feature_cols=market_feature_cols,
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
