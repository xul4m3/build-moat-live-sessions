"""驗證 tokenize（BM25Index 的測試在後續 sub-task 加入）。"""
from app.bm25 import tokenize


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
