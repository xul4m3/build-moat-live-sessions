"""驗證 Section / RetrievalResult dataclass 的基本欄位 + citation property。"""
from app.types import Section, RetrievalResult


def test_section_stores_fields():
    """確認 Section 能儲存四個必要欄位。"""
    s = Section(
        filename="refund_policy.md",
        heading="Refund Timeline",
        heading_slug="refund-timeline",
        body="Approved refunds are processed within 5-7 business days.",
    )
    assert s.filename == "refund_policy.md"
    assert s.heading == "Refund Timeline"
    assert s.heading_slug == "refund-timeline"
    assert "5-7 business days" in s.body


def test_section_citation_combines_filename_and_slug():
    """citation property 要產出 PROMPT.md 規格的 filename#heading-slug 格式。"""
    s = Section(
        filename="account_help.md",
        heading="Change Email Address",
        heading_slug="change-email-address",
        body="...",
    )
    assert s.citation == "account_help.md#change-email-address"


def test_retrieval_result_defaults_are_empty_and_not_shared():
    """RetrievalResult() 不帶參數時應得到空 sections / 空 scores / fallback=False。

    額外驗證 field(default_factory=list) 沒踩到 mutable-default 陷阱：
    兩個獨立 instance 拿到的 sections 不能是同一個 list 物件。
    """
    r1 = RetrievalResult()
    r2 = RetrievalResult()

    assert r1.sections == []
    assert r1.scores == []
    assert r1.fallback is False

    # mutate r1.sections 不應該影響 r2.sections
    r1.sections.append(Section("x.md", "X", "x", "x body"))
    assert r2.sections == []
    assert r1.sections is not r2.sections
