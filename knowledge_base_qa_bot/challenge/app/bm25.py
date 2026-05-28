"""
BM25 inverted index 封裝。

責任：
- tokenize(text) → list[str]：lowercase + 拆字 + 去 stopwords
- BM25Index：封裝 rank_bm25.BM25Okapi，保留 Section 對應、提供 top_k 介面
  （在後續 sub-task 加入）
"""
import re


# 常見英文 stopword 集合（set literal：大括號 {} 包字串，O(1) 查詢）。
# BM25 對 stopword 不敏感（它本來就會用 IDF 壓低高頻詞），
# 不過去掉能讓 tokens 更精簡、index.json 更小、log 更乾淨。
STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "i", "in", "is", "it", "its", "of", "on",
    "that", "the", "to", "was", "were", "will", "with", "you", "your",
    "can", "this", "these", "they", "them", "their", "or", "but", "if",
    "do", "does", "did", "not", "no",
}


def tokenize(text: str) -> list[str]:
    """把字串切成 lowercase token list、丟掉 stopwords。

    例: "The Refund is 5-7 days" -> ["refund", "5", "7", "days"]

    步驟：
    1. text.lower()：全部轉小寫
    2. re.findall(r"[a-z0-9]+", ...)：regex 找出所有連續英數字元 chunk
       （非英數字元自動被當 separator，不會出現在結果裡）
    3. list comprehension [t for t in raw if ...]：過濾掉 stopwords
       （`t not in STOPWORDS` 是 set membership test，O(1)）
    """
    # re.findall 回傳所有符合 pattern 的 substring list
    raw = re.findall(r"[a-z0-9]+", text.lower())
    # list comprehension：[運算式 for 變數 in 可迭代物 if 條件]
    return [t for t in raw if t not in STOPWORDS]
