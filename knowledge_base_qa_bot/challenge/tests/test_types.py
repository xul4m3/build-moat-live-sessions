"""驗證 Section dataclass 的基本欄位 + citation property。"""
from app.types import Section


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
