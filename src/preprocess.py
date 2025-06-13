# src/preprocess.py
"""
공통 영문 리뷰 전처리 모듈
"""

from __future__ import annotations

import glob
import hashlib
import re
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from sklearn.utils import shuffle

#########################
# 1-A. 텍스트 정제 함수 #
#########################
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<.*?>")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """URL·HTML 태그 제거 → 소문자화 → 공백 정규화"""
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)  # 구두점 제거
    text = _MULTI_SPACE.sub(" ", text).strip().lower()
    return text


####################################
# 1-B. 데이터 로딩 + 파이프라인 함수 #
####################################
def _hash_row(row: pd.Series) -> str:
    """중복 제거용 해시값 생성"""
    m = hashlib.md5()
    m.update(row["review_text"].encode("utf-8"))
    m.update(str(row.get("country", "")).encode())
    return m.hexdigest()


def load_reviews(
    paths: Sequence[str | Path] | str,
    text_col: str = "review_text",
    country_col: str = "country",
    seed: int | None = 42,
) -> Tuple[list[str], pd.DataFrame]:
    """
    여러 CSV/JSON 파일을 읽어 하나의 DataFrame으로 합친 뒤
    중복 제거·전처리를 수행한다.

    Returns
    -------
    docs : list[str]           (정제된 리뷰 문장 리스트)
    meta : pd.DataFrame        (review_id, country 등 메타)
    """
    # 1. 파일 목록 수집
    file_list = glob.glob(paths) if isinstance(paths, str) else [str(p) for p in paths]
    if not file_list:
        raise FileNotFoundError(f"No input files matched: {paths}")

    dfs = []
    for fp in file_list:
        df = (
            pd.read_json(fp, lines=True)
            if fp.endswith(".json") or fp.endswith(".jsonl")
            else pd.read_csv(fp)
        )
        if text_col not in df.columns:
            raise KeyError(f"{text_col} not in {fp}")
        # 열 이름 표준화
        df = df.rename(columns={text_col: "review_text", country_col: "country"})
        df["source_file"] = Path(fp).name
        dfs.append(df[["review_text", "country", "source_file"]])

    data = pd.concat(dfs, ignore_index=True)

    # 2. 전처리
    data["review_text"] = data["review_text"].astype(str).map(clean_text)
    data["row_hash"] = data.apply(_hash_row, axis=1)
    data = data.drop_duplicates("row_hash").drop(columns="row_hash")

    # 3. 셔플(재현성 위해 seed 고정 selectable)
    if seed is not None:
        data = shuffle(data, random_state=seed).reset_index(drop=True)

    return data["review_text"].tolist(), data[["country", "source_file"]]
