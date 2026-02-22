"""FinBERT + LSTM hybrid model for stock movement prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FinBERTLSTMHybrid(nn.Module):
    def __init__(
        self,
        finbert_model_name: str = "ProsusAI/finbert",
        market_input_size: int = 5,
        market_hidden_size: int = 64,
        market_layers: int = 1,
        fusion_hidden_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
        self.finbert = AutoModel.from_pretrained(finbert_model_name)
        finbert_dim = self.finbert.config.hidden_size

        self.market_encoder = nn.LSTM(
            input_size=market_input_size,
            hidden_size=market_hidden_size,
            num_layers=market_layers,
            batch_first=True,
            dropout=dropout if market_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(finbert_dim + market_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        outputs = self.finbert(**tokens)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, market_seq: torch.Tensor, texts: list[str]) -> torch.Tensor:
        device = market_seq.device
        text_repr = self.encode_text(texts, device)
        _, (hidden, _) = self.market_encoder(market_seq)
        market_repr = hidden[-1]
        logits = self.classifier(torch.cat([text_repr, market_repr], dim=1))
        return logits.squeeze(1)
