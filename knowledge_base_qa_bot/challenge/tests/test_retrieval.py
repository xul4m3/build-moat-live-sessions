"""驗證 retrieval.search：top-k 排序 + threshold fallback。"""
from app.bm25 import BM25Index
from app.retrieval import search
from app.types import Section


def _index() -> BM25Index:
    return BM25Index.build([
        Section("refund.md", "Refund Timeline", "refund-timeline",
                "Approved refunds are processed within 5-7 business days."),
        Section("account.md", "Reset Password", "reset-password",
                "Customers can reset password from the sign-in page."),
        Section("shipping.md", "Standard Shipping", "standard-shipping",
                "Standard shipping takes 3-5 business days."),
    ])


def test_search_returns_top_k_above_threshold():
    """正常 query -> 拿到 top-k、fallback=False。"""
    result = search("how long do refunds take", _index(), k=3, threshold=0.1)
    assert result.fallback is False
    assert len(result.sections) == 3
    assert result.sections[0].heading == "Refund Timeline"
    assert len(result.scores) == 3
    # 分數應該由大到小
    assert result.scores[0] >= result.scores[1] >= result.scores[2]


def test_search_all_below_threshold_returns_fallback():
    """top-1 score < threshold -> fallback=True、sections 空。"""
    # 用一個對 corpus 完全陌生的 query
    result = search("which restaurants are nearby", _index(), k=3, threshold=0.5)
    assert result.fallback is True
    assert result.sections == []
    assert result.scores == []


def test_search_empty_index_returns_fallback():
    """空 index -> fallback。"""
    empty_idx = BM25Index.build([])
    result = search("anything", empty_idx, k=3, threshold=0.5)
    assert result.fallback is True


def test_search_k_larger_than_corpus_does_not_error():
    """k > section 數，不報錯、回有的就好。"""
    result = search("shipping", _index(), k=99, threshold=0.0)
    assert len(result.sections) == 3
