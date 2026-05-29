"""
共用 dataclass 集中地：
- Section: Markdown 切出來的最小檢索單位
- RetrievalResult: retrieval 模組的回傳型別（封裝 sections + scores + fallback flag）

dataclass 是 Python 3.7+ 內建的「自動產生 __init__、__repr__、__eq__」裝飾器，
省掉手動寫 boilerplate。
@property 讓 method 可以像 attribute 一樣存取（s.citation 不是 s.citation()）。
field() 是 dataclass 給「需要 default 但 default 是 mutable 物件（list / dict / set）」用的工具 ——
若直接寫 `sections: list = []`，所有 instance 會共用同一個 list（Python 經典踩雷），
要改寫 `sections: list = field(default_factory=list)` 讓每個 instance 拿到自己的空 list。
"""

from dataclasses import dataclass, field


@dataclass
class Section:
    """從 Markdown 切出來的一個段落，是 BM25 檢索的最小單位。"""

    filename: str  # 例: "refund_policy.md"
    heading: str  # 原始 heading 文字，例: "Refund Timeline"
    heading_slug: str  # URL 友善的 slug，例: "refund-timeline"
    body: str  # 段落內文（不含 heading 那一行）

    @property
    def citation(self) -> str:
        """產出 PROMPT.md 要求的 'filename#heading-slug' 引用格式。"""
        return f"{self.filename}#{self.heading_slug}"


@dataclass
class RetrievalResult:
    """retrieval.search() 的回傳值。

    fallback=True 代表所有 section 的 BM25 score 都低於 threshold，
    呼叫端應該直接回 cannot-confirm、不要再呼 LLM。

    注意：`list[Section]` 在 class body 內會立即被解析（沒有用
    `from __future__ import annotations`），所以 Section 必須定義在
    RetrievalResult 之上 —— 順序不能反，反了會在 import 時 NameError。
    """

    sections: list[Section] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    fallback: bool = False
