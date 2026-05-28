"""驗證 tokenize + BM25Index。"""
from app.bm25 import tokenize, BM25Index
from app.types import Section


def test_tokenize_lowercases_and_splits_words():
    """基本切詞：lowercase、按非英數切。"""
    assert tokenize("Hello World") == ["hello", "world"]
    assert tokenize("Refund Timeline") == ["refund", "timeline"]


def test_tokenize_drops_stopwords():
    """常見英文 stopword 要去掉（the, is, a, an...）。"""
    tokens = tokenize("the refund is processed in 5 days")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "in" not in tokens
    assert "refund" in tokens
    assert "processed" in tokens
    assert "5" in tokens
    assert "days" in tokens


def test_tokenize_handles_punctuation():
    """標點被當作 separator、不會混進 token。"""
    tokens = tokenize("Refunds: 5-7 business days!")
    assert tokens == ["refunds", "5", "7", "business", "days"]


def _make_sections() -> list[Section]:
    """共用 helper：3 個 section 模擬 sample corpus。

    不是 fixture（沒有 @pytest.fixture），所以叫底線開頭表示 module-private、
    test 直接呼叫即可。
    """
    return [
        Section("refund.md", "Refund Timeline", "refund-timeline",
                "Approved refunds are processed within 5-7 business days."),
        Section("account.md", "Reset Password", "reset-password",
                "Customers can reset password from the sign-in page."),
        Section("shipping.md", "Standard Shipping", "standard-shipping",
                "Standard shipping takes 3-5 business days."),
    ]


def test_bm25index_build_from_sections():
    """build() 應該回 BM25Index 物件、sections 保留原順序。"""
    sections = _make_sections()
    idx = BM25Index.build(sections)
    assert idx.sections == sections
    # 每個 section 都該有 tokenized body
    assert len(idx.tokens) == 3
    # body 是 "Approved refunds are processed within 5-7 business days."
    # 沒做 stemming，所以 token 也是 "refunds"（複數）而非 "refund"
    assert "refunds" in idx.tokens[0]
    assert "processed" in idx.tokens[0]


def test_bm25index_top_k_ranks_relevant_section_first():
    """有強相關詞 -> 對應 section 排第一。"""
    idx = BM25Index.build(_make_sections())
    ranked = idx.top_k("how long do refunds take", k=3)

    assert len(ranked) == 3
    top_section, top_score = ranked[0]
    assert top_section.heading == "Refund Timeline"
    assert top_score > 0


def test_bm25index_top_k_unknown_query_returns_zero_scores():
    """完全沒命中的 query -> 所有 score 為 0。"""
    idx = BM25Index.build(_make_sections())
    ranked = idx.top_k("which restaurants are nearby", k=3)
    # rank_bm25 在沒命中時會回 0
    assert all(score == 0 for _, score in ranked)


def test_bm25index_top_k_respects_k_limit():
    """k 比 section 數小 -> 只回 k 個。"""
    idx = BM25Index.build(_make_sections())
    ranked = idx.top_k("shipping", k=1)
    assert len(ranked) == 1


def test_bm25index_top_k_larger_than_corpus_returns_all():
    """k > section 數 -> 回全部、不報錯。"""
    idx = BM25Index.build(_make_sections())
    ranked = idx.top_k("shipping", k=10)
    assert len(ranked) == 3
