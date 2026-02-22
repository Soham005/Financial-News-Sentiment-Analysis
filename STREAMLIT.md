# Streamlit Usage

This repo includes a lightweight Streamlit app for interactive inference.

## Install

```bash
pip install -r requirements-streamlit.txt
```

## Train and Save Checkpoint

```bash
python src/train.py --epochs 2 --batch-size 8 --seq-len 10 --save-checkpoint --output-path artifacts/finbert_lstm.pt
```

Use this checkpoint path in the app:

- `artifacts/finbert_lstm.pt`

## Run

```bash
streamlit run app.py
```

## Notes

- If `artifacts/finbert_lstm.pt` exists and contains `model_state_dict`, the app loads it.
- If no checkpoint is found, the app still runs but predictions come from an untrained model.
- The market table is prefilled with a realistic trend example and is fully editable.
- If no checkpoint is found, the app still runs using default initialized model weights.
- For meaningful predictions, train and save your own checkpoint in your workflow.
