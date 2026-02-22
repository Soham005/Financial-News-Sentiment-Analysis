"""Data utilities for Financial News + Market sequence modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class SequenceConfig:
    seq_len: int = 10
    market_feature_cols: Sequence[str] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    target_col: str = "target"


class NewsMarketDataset(Dataset):
    """Builds rolling windows over market features paired with aligned news text."""

    def __init__(
        self,
        frame: pd.DataFrame,
        texts: List[str],
        config: SequenceConfig,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ) -> None:
        if len(frame) != len(texts):
            raise ValueError("Market frame and texts must have the same length.")

        self.config = config
        self.texts = texts
        self.market = frame.loc[:, list(config.market_feature_cols)].to_numpy(dtype=np.float32)
        self.targets = frame.loc[:, config.target_col].to_numpy(dtype=np.float32)

        self.scaler = scaler or StandardScaler()
        if fit_scaler:
            self.market = self.scaler.fit_transform(self.market)
        else:
            self.market = self.scaler.transform(self.market)

    def __len__(self) -> int:
        return max(0, len(self.market) - self.config.seq_len + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        start = idx
        end = idx + self.config.seq_len
        market_window = self.market[start:end]
        target = self.targets[end - 1]
        news_text = self.texts[end - 1]

        return {
            "market": torch.tensor(market_window, dtype=torch.float32),
            "text": news_text,
            "label": torch.tensor(target, dtype=torch.float32),
        }
