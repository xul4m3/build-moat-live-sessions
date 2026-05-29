"""
BM25 inverted index 封裝。

責任：
- tokenize(text) → list[str]：lowercase + 拆字 + 去 stopwords
- BM25Index：封裝 rank_bm25.BM25Okapi，保留 Section 對應、提供 top_k 介面
"""
import re
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from app.types import Section


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


@dataclass
class BM25Index:
    """封裝 rank_bm25.BM25Okapi、保留 Section 對應。

    建議用 BM25Index.build(sections) 建立，這個 classmethod 會幫你 tokenize。
    直接呼 __init__ 通常是 store.load() 反序列化時用的（tokens 已經算過了）。
    """
    sections: list[Section]
    tokens: list[list[str]]
    # field() 是 dataclass 的進階設定工具：
    # - init=False：不放進 __init__ 參數（由 __post_init__ 自行建立）
    # - repr=False：不放進 __repr__ 輸出（避免印出整個 BM25 內部物件）
    # - compare=False：不放進 __eq__ 比較（兩個 index 相等看 sections + tokens 就夠）
    # 為什麼用單底線 _bm25 而不是雙底線 __bm25：雙底線會觸發 Python name mangling，
    # 把欄位名改成 _BM25Index__bm25，dataclass field() 對不上會 silently 出錯。
    _bm25: BM25Okapi = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """dataclass 在 __init__ 跑完之後自動呼這個 hook。

        我們用它來建 rank_bm25 的內部物件 —— sections + tokens 由 caller 提供，
        _bm25 在這裡重建就好。

        空 corpus 處理：rank_bm25.BM25Okapi 對空 list 會 raise（divide by zero）；
        給它一個 dummy doc [""] 讓初始化成功。top_k 已有 `if not self.sections` 短路，
        所以 dummy doc 不會影響查詢結果。

        不變量：build() 跟 store.load() 都會把 sections 跟 tokens 維持平行
        （len 相等、index 對應）。因此 `not self.tokens` 在實務上等同 `not self.sections`，
        dummy doc 路徑只會在「真的空索引」時觸發。
        """
        # self.tokens or [[""]]：若 tokens 是空 list（falsy），改用含一個空 doc 的 list
        self._bm25 = BM25Okapi(self.tokens if self.tokens else [[""]])

    @classmethod
    def build(cls, sections: list[Section]) -> "BM25Index":
        """從 Section list 建索引：tokenize 每個 body、丟給 BM25Okapi。

        classmethod 是「屬於 class 而非 instance 的方法」，第一個參數是 cls（類別本身）
        而非 self（實例）。這裡用它當 factory method，讓 caller 不需要自己 tokenize。
        """
        # list comprehension：對每個 section s，呼 tokenize(s.body) 拿到 token list
        tokens = [tokenize(s.body) for s in sections]
        return cls(sections=sections, tokens=tokens)

    def top_k(self, query: str, k: int) -> list[tuple[Section, float]]:
        """回傳分數最高的 k 個 (section, score)，分數由大到小。

        k > section 數 → 自動 clamp 成 len(sections)（sorted 後切片 [:k] 自然處理）。
        所有 score 為 0 → 仍然回 k 個，由 retrieval 層用 threshold 過濾。
        """
        if not self.sections:
            return []

        query_tokens = tokenize(query)
        # get_scores 回 numpy ndarray，每個元素對應一個 section 的 BM25 分數
        scores = self._bm25.get_scores(query_tokens)
        # zip(a, b)：把兩個序列對位配對成 (a[i], b[i]) 的 iterator
        # sorted(..., key=lambda pair: pair[1], reverse=True)：
        #   lambda 是匿名函式，pair[1] 取第二個元素（score）作為排序鍵，reverse=True 降冪
        ranked = sorted(
            zip(self.sections, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        # list comprehension + 切片 [:k]（k > len 時自動回全部，不報錯）
        # float() 把 numpy float64 轉成 Python 原生 float，避免序列化問題
        return [(sec, float(score)) for sec, score in ranked[:k]]
