# src/analysis/robustness.py
"""
Keyword vs Topic 겹침비, Topic-별 평균 감성을 계산
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def jaccard(set1: Iterable[str], set2: Iterable[str]) -> float:
    a, b = set(set1), set(set2)
    return len(a & b) / len(a | b) if a | b else 0.0


def compare_keywords_topics(
    kw_graph_csv: str,
    topic_info_csv: str,
    top_n: int = 30,
) -> float:
    kw_df = pd.read_csv(kw_graph_csv)
    kw_top = kw_df.nlargest(top_n, "tfidf")["keyword"]
    top_words = [w.split("_")[0] for _, w in kw_top.items()]  # unigram 우선

    topics = pd.read_csv(topic_info_csv)
    topic_terms = topics[topics.Topic != -1]["Name"].str.split(";").explode()
    topic_top = topic_terms.head(top_n)

    score = jaccard(top_words, topic_top)
    print(f"[robustness] Keyword vs Topic Jaccard({top_n}) = {score:.3f}")
    return score
