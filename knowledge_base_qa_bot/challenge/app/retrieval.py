"""
Retrieval 層：把 BM25 排名跟 threshold 結合，產出 RetrievalResult。

threshold 邏輯（DESIGN.md §6.1）：
- 如果 top-1 score < threshold → 視為「沒有相關內容」、回 fallback=True
- fallback=True 時 呼叫端應該直接回 cannot-confirm、不要再呼 LLM
"""

from app.bm25 import BM25Index
from app.types import RetrievalResult


def search(
    query: str,
    index: BM25Index,
    k: int,
    threshold: float,
) -> RetrievalResult:
    """檢索 + threshold 判斷。

    參數:
        query: 使用者問題原文
        index: 已 build 好的 BM25Index
        k: 想要的 top-K 數量
        threshold: top-1 score 至少要超過這個值，否則 fallback

    回傳:
        RetrievalResult；fallback=True 時 sections / scores 都是空 list。
    """
    ranked = index.top_k(query, k)

    # 空 index → fallback
    if not ranked:
        return RetrievalResult(fallback=True)

    # 解構 top-1 看分數；用具名變數比 ranked[0][1] 直白
    _, top_score = ranked[0]
    if top_score < threshold:
        return RetrievalResult(fallback=True)

    # zip(*iterable) 是 unzip 慣用法：把 [(s1, sc1), (s2, sc2), ...] 轉成
    # ([s1, s2, ...], [sc1, sc2, ...])。比兩次 list comprehension 更慣用、只走訪一次。
    # 注意：zip 回 tuple，所以下面要 list(...) 把它轉成 list。
    # strict=True：ranked 裡每個元素都是 (section, score) 2-tuple，解開必等長
    sections_tuple, scores_tuple = zip(*ranked, strict=True)
    return RetrievalResult(
        sections=list(sections_tuple),
        scores=list(scores_tuple),
        fallback=False,
    )
