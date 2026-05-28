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

    # 沒有任何 section（空 index）或 top-1 分數太低 -> fallback
    if not ranked or ranked[0][1] < threshold:
        return RetrievalResult(sections=[], scores=[], fallback=True)

    # 拆 tuple list 成兩個平行 list；list comprehension 是 Python 慣用寫法
    # pair[0] 是 Section 物件、pair[1] 是 BM25 score（float）
    sections = [pair[0] for pair in ranked]
    scores = [pair[1] for pair in ranked]
    return RetrievalResult(sections=sections, scores=scores, fallback=False)
