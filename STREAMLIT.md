# Streamlit Usage

This repo includes a lightweight Streamlit app for interactive inference.

## Install

```bash
pip install -r requirements-streamlit.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- If `artifacts/finbert_lstm.pt` exists and contains `model_state_dict`, the app loads it.
- If no checkpoint is found, the app still runs using default initialized model weights.
- For meaningful predictions, train and save your own checkpoint in your workflow.
