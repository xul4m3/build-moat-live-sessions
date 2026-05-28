"""驗證 prompt 組裝：system instruction + context + query。"""
from app.prompt import build_messages
from app.types import Section


def _sections() -> list[Section]:
    return [
        Section("refund.md", "Refund Timeline", "refund-timeline",
                "Approved refunds are processed within 5-7 business days."),
        Section("shipping.md", "Standard Shipping", "standard-shipping",
                "Standard shipping takes 3-5 business days."),
    ]


def test_build_messages_returns_system_then_user():
    """OpenAI messages 慣例：system 在前、user 在後。"""
    msgs = build_messages("how long do refunds take", _sections())
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_system_message_includes_grounding_instruction():
    """system message 要明確告訴 LLM 只能根據 context 回答 + cite sources。"""
    msgs = build_messages("any", _sections())
    system_content = msgs[0]["content"].lower()
    assert "context" in system_content or "knowledge base" in system_content
    assert "filename#heading" in msgs[0]["content"] or "[" in msgs[0]["content"]
    assert "cannot" in system_content


def test_user_message_includes_every_citation_tag():
    """每個 retrieved section 的 citation 都要出現在 user message 裡。"""
    msgs = build_messages("any", _sections())
    user_content = msgs[1]["content"]
    assert "refund.md#refund-timeline" in user_content
    assert "shipping.md#standard-shipping" in user_content


def test_user_message_includes_query_verbatim():
    """user 的原始問題要原封不動進 prompt。"""
    query = "Why is shipping so slow?"
    msgs = build_messages(query, _sections())
    assert query in msgs[1]["content"]


def test_user_message_includes_section_bodies():
    """section body 要進 prompt（不然 LLM 沒辦法引用內容）。"""
    msgs = build_messages("any", _sections())
    user_content = msgs[1]["content"]
    assert "5-7 business days" in user_content
    assert "3-5 business days" in user_content


def test_context_section_order_matches_input():
    """sections 的順序原樣保留 -> refund 在 shipping 之前。

    contract test：未來若有人在 build_messages 加排序邏輯，test 會擋下。
    排序是 retrieval 的職責，prompt 層不該動。
    """
    msgs = build_messages("any", _sections())
    user_content = msgs[1]["content"]
    refund_pos = user_content.index("refund.md#refund-timeline")
    shipping_pos = user_content.index("shipping.md#standard-shipping")
    assert refund_pos < shipping_pos
