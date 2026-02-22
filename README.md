# Financial News Sentiment Driven Stock Movement Prediction using FinBERT and LSTM Hybrid Model

This repository provides a starter implementation for a **hybrid prediction pipeline** that combines:

- **FinBERT** for financial-news sentiment/context encoding.
- **LSTM** for sequential market feature modeling.
- A **fusion head** for binary stock movement prediction (up/down).

## Architecture

1. **Text Branch (FinBERT)**
   - Input: daily or intraday aggregated financial news text.
   - Output: `[CLS]` embedding from FinBERT.

2. **Market Branch (LSTM)**
   - Input: rolling window of market features (`open`, `high`, `low`, `close`, `volume`).
   - Output: last hidden state of LSTM.

3. **Fusion + Classifier**
   - Concatenate FinBERT and LSTM representations.
   - Feed through MLP and output a logit for binary movement classification.

## Repository Structure

- `src/data.py`: sequence dataset and scaling utilities.
- `src/model.py`: FinBERT + LSTM hybrid network.
- `src/train.py`: end-to-end training loop with dummy data generation.
- `src/inference.py`: checkpoint loading and prediction utilities.
- `app.py`: Streamlit inference app.
- `requirements.txt`: Python dependencies.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and Save a Checkpoint

Run a demo training job (uses generated dummy data) and save model artifacts:

```bash
python src/train.py --epochs 2 --batch-size 8 --seq-len 10 --save-checkpoint --output-path artifacts/finbert_lstm.pt
```

## Run with Streamlit

After creating a checkpoint, start the app:

```bash
streamlit run app.py
```

Then in the browser:
- provide the checkpoint path (default: `artifacts/finbert_lstm.pt`),
- input/adjust market sequence rows,
- paste financial news text,
- click **Predict movement probability**.

## Expected Real-World Data Format

For production use, replace `make_dummy_data` with your own ingestion pipeline.

### Market Data Columns

- `open`
- `high`
- `low`
- `close`
- `volume`
- `target` (0/1 movement label)

### News Data

A text entry aligned to each market row (or timestamp bucket), e.g.:

- Headline-only text
- Headline + summary
- Concatenated daily news snippets

## Next Improvements

- Add temporal attention in market branch.
- Add multi-horizon outputs (`t+1`, `t+5`, `t+10`).
- Use calibrated probabilities (temperature scaling / isotonic).
- Integrate backtesting and risk-adjusted metrics.
