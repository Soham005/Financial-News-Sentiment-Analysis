"""Training entrypoint for FinBERT + LSTM hybrid model."""

from __future__ import annotations

import argparse


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import NewsMarketDataset, SequenceConfig
from model import FinBERTLSTMHybrid


def collate_batch(batch: list[dict]) -> dict:
    return {
        "market": torch.stack([item["market"] for item in batch]),
        "text": [item["text"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
    }


def make_dummy_data(rows: int = 400) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(42)
    prices = rng.normal(100, 3, size=(rows, 4))
    volume = rng.integers(1_000, 100_000, size=(rows, 1))
    returns = rng.normal(0, 1, size=(rows, 1))

    frame = pd.DataFrame(
        np.concatenate([prices, volume, returns], axis=1),
        columns=["open", "high", "low", "close", "volume", "ret"],
    )
    frame["target"] = (frame["ret"] > 0).astype(np.float32)
    texts = ["Market outlook is stable with mixed signals." for _ in range(rows)]
    return frame, texts



def train(args: argparse.Namespace) -> None:
    frame, texts = make_dummy_data(args.rows)
    train_idx, val_idx = train_test_split(
        np.arange(len(frame)), test_size=0.2, shuffle=False
    )

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    val_frame = frame.iloc[val_idx].reset_index(drop=True)
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]

    config = SequenceConfig(seq_len=args.seq_len)
    scaler = StandardScaler()

    train_ds = NewsMarketDataset(train_frame, train_texts, config, scaler=scaler, fit_scaler=True)
    val_ds = NewsMarketDataset(val_frame, val_texts, config, scaler=scaler, fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinBERTLSTMHybrid().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            market = batch["market"].to(device)
            labels = batch["label"].to(device)
            logits = model(market, batch["text"])
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * market.size(0)

        train_loss /= max(len(train_ds), 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                market = batch["market"].to(device)
                labels = batch["label"].to(device)
                logits = model(market, batch["text"])
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--output-path", type=str, default="artifacts/finbert_lstm.pt")
    train(parser.parse_args())
