# src/run_pipeline.py
"""
End-to-End 파이프라인:
1) 리뷰 로드·전처리
2) 키워드 네트워크 + TF-IDF 표 + 국가×키워드 빈도표
3) BERTopic 토픽 모델 + 국가×토픽 빈도표
4) Robustness: 키워드·토픽 Jaccard 유사도
5) 정량 검정: 두 빈도표 χ² & Cramer’s V
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from analysis.keyword_network import build_cooc_graph, save_graph
from analysis.robustness import compare_keywords_topics
from analysis.topic_model import run_bertopic
from preprocess import load_reviews
from sklearn.feature_extraction.text import CountVectorizer
from stats_tests import chi_square_table


def keyword_tables(
    docs: list[str],
    countries: pd.Series,
    out_dir: Path,
    kw_cfg: dict,
) -> pd.DataFrame:
    """
    ① GraphML 저장
    ② TF-IDF 상위표(keyword_tfidf.csv)
    ③ 국가×키워드 빈도표(keyword_freq.csv)  반환
    """
    # --- 네트워크 & TF-IDF --------------------------------------------------
    G = build_cooc_graph(docs, window=kw_cfg["window"], min_df=kw_cfg["min_df"])
    save_graph(G, out_dir / "keyword_network.graphml")

    tfidf_df = (
        pd.DataFrame(
            [(n, d.get("tfidf", 0.0)) for n, d in G.nodes(data=True)],
            columns=["keyword", "tfidf"],
        )
        .sort_values("tfidf", ascending=False)
        .reset_index(drop=True)
    )
    tfidf_df.to_csv(out_dir / "keyword_tfidf.csv", index=False)

    # --- 국가 × 키워드 빈도표 ---------------------------------------------
    cv = CountVectorizer(
        min_df=kw_cfg["min_df"],
        stop_words="english",
        ngram_range=tuple(kw_cfg.get("ngram_range", (1, 1))),
    )
    X = cv.fit_transform(docs)
    vocab = cv.get_feature_names_out()

    df_counts = pd.DataFrame.sparse.from_spmatrix(X, columns=vocab)
    df_counts["country"] = countries.values
    country_tab = df_counts.groupby("country").sum()

    # 상위 N 키워드만 사용해 검정(희소 행렬 완화)
    top_n = kw_cfg.get("top_n", 50)
    top_keywords = country_tab.sum(axis=0).nlargest(top_n).index.tolist()
    kw_tab = country_tab[top_keywords]
    kw_tab.to_csv(out_dir / "keyword_freq.csv")

    return kw_tab


def topic_table(
    docs: list[str],
    countries: pd.Series,
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """BERTopic 실행 후 국가×토픽 빈도표(topic_freq.csv) 반환"""
    run_bertopic(docs, out_dir, seed=seed)
    assign = pd.read_csv(out_dir / "topic_assignment.csv")  # topic, prob
    assign["country"] = countries.values
    tab = assign.pivot_table(
        index="country", columns="topic", aggfunc="size", fill_value=0
    ).astype(int)
    tab.to_csv(out_dir / "topic_freq.csv")
    return tab


def main(cfg_path: str = "conf/config.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(cfg.get("mlflow_experiment", "intern_ageism"))
    with mlflow.start_run():
        mlflow.log_params(cfg)

        # 1. Load & preprocess ------------------------------------------------
        docs, meta = load_reviews(**cfg["dataset"])
        meta.to_csv(out_dir / "meta.csv", index=False)

        # 2. 키워드 네트워크 + 빈도표 ----------------------------------------
        kw_tab = keyword_tables(docs, meta["country"], out_dir, cfg["keyword"])
        kw_stats = chi_square_table(kw_tab)
        mlflow.log_metrics({f"kw_{k}": float(v) for k, v in kw_stats.items()})

        # 3. 토픽 모델 + 빈도표 ---------------------------------------------
        topic_tab = topic_table(
            docs, meta["country"], out_dir, seed=cfg["dataset"]["seed"]
        )
        topic_stats = chi_square_table(topic_tab)
        mlflow.log_metrics({f"topic_{k}": float(v) for k, v in topic_stats.items()})

        # 4. Robustness: 키워드 vs 토픽 Jaccard ------------------------------
        jac = compare_keywords_topics(
            out_dir / "keyword_tfidf.csv",
            out_dir / "topic_info.csv",
            top_n=cfg["keyword"].get("top_n", 50),
        )
        mlflow.log_metric("robustness_jaccard", jac)

        # 5. 콘솔 요약 --------------------------------------------------------
        print("\n=== χ² & Cramer’s V ===")
        print("Keyword table :", kw_stats)
        print("Topic   table :", topic_stats)
        print(f"Robustness Jaccard({cfg['keyword'].get('top_n', 50)}) = {jac:.3f}")
        print(f"\n[Pipeline] All results saved to → {out_dir.absolute()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="conf/config.yaml")
    main(ap.parse_args().cfg)
