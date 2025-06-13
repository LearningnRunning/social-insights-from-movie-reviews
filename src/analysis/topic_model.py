# src/analysis/topic_model.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def run_bertopic(
    docs: list[str],
    out_dir: str | Path = "results",
    seed: int = 42,
) -> pd.DataFrame:
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic(
        embedding_model=embedder,
        language="english",  # 리뷰가 모두 영문
        calculate_probabilities=True,
        verbose=True,
        random_state=seed,
    )
    topics, probs = topic_model.fit_transform(docs)

    info = topic_model.get_topic_info()
    info.to_csv(Path(out_dir) / "topic_info.csv", index=False)
    pd.DataFrame({"topic": topics, "prob": probs}).to_csv(
        Path(out_dir) / "topic_assignment.csv", index=False
    )
    print(f"[topic_model] Saved topic_info.csv, topic_assignment.csv → {out_dir}")
    return info
