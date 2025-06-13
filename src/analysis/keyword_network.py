# src/analysis/keyword_network.py
"""
TF-IDF 기반 키워드-네트워크 분석
→ GraphML 파일 저장
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def build_cooc_graph(docs: list[str], window: int = 2, min_df: int = 10) -> nx.Graph:
    """
    동시출현(co-occurrence) 그래프 생성
    Parameters
    ----------
    docs : list of sentences
    window : 슬라이딩 윈도 크기
    min_df : 최소 문서 등장빈도
    """
    # 1. 단어 카운트
    cv = CountVectorizer(min_df=min_df, stop_words="english", ngram_range=(1, 2))
    X = cv.fit_transform(docs)
    vocab = cv.get_feature_names_out()

    # 2. TF-IDF로 가중치
    tfidf = TfidfTransformer(norm=None)
    X_tfidf = tfidf.fit_transform(X)

    # 3. 윈도우 내 공동 등장 빈도 계산
    cooc = Counter()
    for doc in docs:
        tokens = doc.split()
        for i, tok in enumerate(tokens):
            for j in range(i + 1, min(i + 1 + window, len(tokens))):
                pair = tuple(sorted([tok, tokens[j]]))
                cooc[pair] += 1

    # 4. 그래프 구축
    G = nx.Graph()
    for (w1, w2), cnt in cooc.items():
        if w1 in vocab and w2 in vocab:
            G.add_edge(w1, w2, weight=cnt)
    for word, idx in cv.vocabulary_.items():
        G.nodes[word]["tfidf"] = X_tfidf[:, idx].sum()

    return G


def save_graph(G: nx.Graph, out_path: str) -> None:
    nx.write_graphml(G, out_path)


def run_keyword_network(
    docs: list[str],
    out_path: str = "results/keywords.graphml",
    window: int = 2,
    min_df: int = 10,
) -> None:
    G = build_cooc_graph(docs, window, min_df)
    save_graph(G, out_path)
    print(f"[keyword_network] Saved → {out_path}")
