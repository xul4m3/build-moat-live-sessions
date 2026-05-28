"""驗證 markdown loader：slugify、parse_markdown、load_docs。"""
from pathlib import Path
from app.loader import slugify, parse_markdown, load_docs


def test_slugify_lowercases_and_dasherizes():
    """heading 文字轉成 URL 友善的 slug。"""
    assert slugify("Refund Timeline") == "refund-timeline"
    assert slugify("Change Email Address") == "change-email-address"


def test_slugify_handles_punctuation():
    """標點、多空白都要清掉。"""
    assert slugify("FAQ: Shipping & Returns!") == "faq-shipping-returns"
    assert slugify("  Multiple   Spaces  ") == "multiple-spaces"


def test_parse_markdown_splits_on_h2_headings():
    """一個檔多個 ## heading -> 切多個 Section。"""
    content = (
        "# Refund Policy\n"
        "\n"
        "## Cancellation Window\n"
        "Customers can cancel within 24 hours.\n"
        "\n"
        "## Refund Timeline\n"
        "Refunds are processed within 5-7 business days.\n"
    )
    sections = parse_markdown(content, "refund_policy.md")

    assert len(sections) == 2
    assert sections[0].heading == "Cancellation Window"
    assert sections[0].heading_slug == "cancellation-window"
    assert sections[0].filename == "refund_policy.md"
    assert "24 hours" in sections[0].body

    assert sections[1].heading == "Refund Timeline"
    assert "5-7 business days" in sections[1].body


def test_parse_markdown_no_h2_uses_filename_as_heading():
    """沒有 ## 的檔 -> 整檔當一個 section，heading 用檔名去 .md。"""
    content = "# Some Title\n\nJust a flat paragraph with no section headings.\n"
    sections = parse_markdown(content, "flat.md")

    assert len(sections) == 1
    assert sections[0].heading == "flat"
    assert sections[0].heading_slug == "flat"
    assert "flat paragraph" in sections[0].body


def test_parse_markdown_ignores_content_before_first_h2():
    """## 之前的內容（含 # H1、前言）不算 section body。"""
    content = (
        "# Top Heading\n"
        "Some intro that nobody asked for.\n"
        "\n"
        "## Real Section\n"
        "This is the actual content.\n"
    )
    sections = parse_markdown(content, "intro.md")

    assert len(sections) == 1
    assert sections[0].heading == "Real Section"
    assert "intro that nobody asked for" not in sections[0].body


def test_parse_markdown_empty_content_returns_one_section_with_empty_body():
    """空內容也不該 crash，回退用 filename。"""
    sections = parse_markdown("", "empty.md")
    assert len(sections) == 1
    assert sections[0].heading == "empty"
    assert sections[0].body == ""


def test_parse_markdown_h3_treated_as_body_not_section_boundary():
    """### H3 不應該被當成 section 邊界 -> 整段含 ### 都會進 H2 section 的 body。"""
    content = (
        "## Section\n"
        "### Sub\n"
        "some text\n"
    )
    sections = parse_markdown(content, "x.md")
    assert len(sections) == 1
    assert sections[0].heading == "Section"
    # ### Sub 那一行應該保留在 body 裡（不被當 heading）
    assert "### Sub" in sections[0].body
    assert "some text" in sections[0].body


def test_parse_markdown_heading_with_no_body_produces_empty_body():
    """連續 ## 中間沒文字 -> 前一個 section body 是空字串。"""
    content = (
        "## First\n"
        "## Second\n"
        "some text\n"
    )
    sections = parse_markdown(content, "x.md")
    assert len(sections) == 2
    assert sections[0].heading == "First"
    assert sections[0].body == ""
    assert sections[1].heading == "Second"
    assert sections[1].body == "some text"


def test_load_docs_reads_all_md_files(sample_docs_dir: Path):
    """conftest.py 的 sample_docs_dir 寫了兩個檔；load_docs 應該把全部 section 攤平。"""
    sections = load_docs(sample_docs_dir)
    assert len(sections) == 2
    headings = {s.heading for s in sections}
    assert headings == {"Refund Timeline", "Reset Password"}


def test_load_docs_ignores_non_md_files(sample_docs_dir: Path):
    """目錄裡有非 .md 檔 -> 不會被讀進來。"""
    (sample_docs_dir / "ignore.txt").write_text("not markdown")
    sections = load_docs(sample_docs_dir)
    assert len(sections) == 2


def test_load_docs_missing_dir_returns_empty_list(tmp_path: Path):
    """目錄不存在 -> 回空 list、不要 raise。"""
    missing = tmp_path / "does-not-exist"
    sections = load_docs(missing)
    assert sections == []
