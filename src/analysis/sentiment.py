# src/analysis/sentiment.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class SentimentPredictor:
    _tok = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    _model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

    @classmethod
    def predict(cls, docs: list[str], batch: int = 32) -> np.ndarray:
        scores = []
        for i in range(0, len(docs), batch):
            toks = cls._tok(
                docs[i : i + batch],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = cls._model(**toks).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend(np.argmax(probs, 1) - 1)  # [-1,0,1]
        return np.asarray(scores, dtype=int)


def run_sentiment(
    docs: list[str],
    out_path: str = "results/sentiment.csv",
) -> pd.DataFrame:
    labels = SentimentPredictor.predict(docs)
    df = pd.DataFrame({"sentiment": labels})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[sentiment] Saved → {out_path}")
    return df
