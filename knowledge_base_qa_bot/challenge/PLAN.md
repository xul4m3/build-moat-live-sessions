# Knowledge Base Q&A Bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `knowledge_base_qa_bot/challenge/` 從零實作一個 BM25-based Markdown KB Q&A bot，通過 `../PROMPT.md` L60-105 的 7 個 curl 驗證。

**Architecture:** Python 3.12 + FastAPI + `rank-bm25` + OpenAI structured output。Section 為檢索單位、top-3 + score threshold、找不到答案直接回 cannot-confirm 不呼 LLM。索引在 Docker build 階段預建、`docs/` 跟 image 走 git PR。

**Tech Stack:** FastAPI、uvicorn、rank-bm25、openai (≥ 1.40, 支援 structured output)、pydantic、python-dotenv、pytest、httpx (給 FastAPI TestClient 用)。

**Spec:** `./DESIGN.md`

---

## Task 0: Bootstrap（環境 + 依賴 + 空殼）

**Files:**
- Create: `challenge/requirements.txt`
- Create: `challenge/.env.example`
- Create: `challenge/.gitignore`
- Create: `challenge/.dockerignore`
- Create: `challenge/README.md`
- Create: `challenge/app/__init__.py`（空檔）
- Create: `challenge/tests/__init__.py`（空檔）
- Create: `challenge/tests/conftest.py`

- [ ] **Step 0.1: Create `challenge/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
rank-bm25==0.2.2
openai>=1.40.0
pydantic>=2.0
python-dotenv==1.0.1
pytest==8.3.3
httpx==0.27.2
```

- [ ] **Step 0.2: Create `challenge/.env.example`**

```
# OpenAI API key for chat completion. Get one at https://platform.openai.com/
OPENAI_API_KEY=sk-replace-me

# LLM model used for generating grounded answers
OPENAI_MODEL=gpt-4o-mini

# Score threshold below which the bot returns cannot-confirm instead of calling LLM.
# Tune empirically against your corpus; see DESIGN.md §6.1.
BM25_SCORE_THRESHOLD=0.5

# Paths (override for K8s; relative for local dev)
KB_DOCS_DIR=../docs
KB_INDEX_PATH=.kb/index.json

# Logging
LOG_LEVEL=INFO
ENV_NAME=local
```

- [ ] **Step 0.3: Create `challenge/.gitignore`**

```
# Python
__pycache__/
*.pyc
.venv/
.pytest_cache/

# Local secrets
.env

# Generated artifacts
.kb/
```

- [ ] **Step 0.4: Create `challenge/.dockerignore`**

放在 `knowledge_base_qa_bot/.dockerignore`（build context 在那層）。先建 placeholder 在 `challenge/`，Task 10 會搬：

```
**/__pycache__
**/*.pyc
**/.venv
**/.pytest_cache
**/.env
**/.kb
**/tests
scaffold/
```

- [ ] **Step 0.5: Create `challenge/README.md`**

```markdown
# KB Q&A Bot — Challenge Track

從零實作的 Markdown KB + BM25 Q&A bot。設計細節見 `DESIGN.md`、施工順序見 `PLAN.md`。

## Setup

```bash
cd challenge
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # 然後把 OPENAI_API_KEY 填進去
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest tests/ -v             # unit + integration（不打外部 API）
bash scripts/smoke.sh        # 真實打 OpenAI（需要 OPENAI_API_KEY）
```
```

- [ ] **Step 0.6: Create `challenge/app/__init__.py` 和 `challenge/tests/__init__.py`**

兩個空檔即可。Python 用這個檔案標記資料夾為 package。

- [ ] **Step 0.7: Create `challenge/tests/conftest.py`（fixtures 集中地）**

```python
"""
pytest 的全域 fixture 集中地。conftest.py 是 pytest 的特殊檔名：
同層或上層目錄裡所有 test 都會自動拿到這裡定義的 fixture，不用 import。
"""
import pytest
from pathlib import Path


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """建一個暫存 docs/ 目錄、寫兩個樣本 .md 進去，回傳路徑。

    tmp_path 是 pytest 內建 fixture：每個 test 拿到一個獨立的暫存資料夾，
    test 結束自動清掉，不會污染 working tree。
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text(
        "# Refund Policy\n\n"
        "## Refund Timeline\n\n"
        "Approved refunds are processed within 5-7 business days.\n",
        encoding="utf-8",
    )
    (docs / "account.md").write_text(
        "# Account\n\n"
        "## Reset Password\n\n"
        "Password reset link expires after 30 minutes.\n",
        encoding="utf-8",
    )
    return docs
```

- [ ] **Step 0.8: Verify environment**

```bash
cd knowledge_base_qa_bot/challenge
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
pytest --version
```

Expected: `pytest 8.3.3` 印出來。

- [ ] **Step 0.9: Commit**

```bash
git add knowledge_base_qa_bot/challenge/requirements.txt \
        knowledge_base_qa_bot/challenge/.env.example \
        knowledge_base_qa_bot/challenge/.gitignore \
        knowledge_base_qa_bot/challenge/.dockerignore \
        knowledge_base_qa_bot/challenge/README.md \
        knowledge_base_qa_bot/challenge/app/__init__.py \
        knowledge_base_qa_bot/challenge/tests/__init__.py \
        knowledge_base_qa_bot/challenge/tests/conftest.py
git commit -m "kb-qa-bot: bootstrap challenge skeleton"
```

---

## Task 1: types.py（Section dataclass）

**Files:**
- Create: `challenge/app/types.py`
- Create: `challenge/tests/test_types.py`

- [ ] **Step 1.1: Write failing test `tests/test_types.py`**

```python
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
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
pytest tests/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.types'`

- [ ] **Step 1.3: Implement `app/types.py`**

```python
"""
共用 dataclass 集中地：
- Section: Markdown 切出來的最小檢索單位
- RetrievalResult: retrieval 模組的回傳型別（封裝 sections + scores + fallback flag）

dataclass 是 Python 3.7+ 內建的「自動產生 __init__、__repr__、__eq__」裝飾器，
省掉手動寫 boilerplate。@property 讓 method 可以像 attribute 一樣存取（s.citation 不是 s.citation()）。
"""
from dataclasses import dataclass, field


@dataclass
class Section:
    """從 Markdown 切出來的一個段落，是 BM25 檢索的最小單位。"""
    filename: str       # 例: "refund_policy.md"
    heading: str        # 原始 heading 文字，例: "Refund Timeline"
    heading_slug: str   # URL 友善的 slug，例: "refund-timeline"
    body: str           # 段落內文（不含 heading 那一行）

    @property
    def citation(self) -> str:
        """產出 PROMPT.md 要求的 'filename#heading-slug' 引用格式。"""
        return f"{self.filename}#{self.heading_slug}"


@dataclass
class RetrievalResult:
    """retrieval.search() 的回傳值。

    fallback=True 代表所有 section 的 BM25 score 都低於 threshold，
    呼叫端應該直接回 cannot-confirm、不要再呼 LLM。
    """
    sections: list[Section] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    fallback: bool = False
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
pytest tests/test_types.py -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```bash
git add app/types.py tests/test_types.py
git commit -m "kb-qa-bot: add Section and RetrievalResult dataclasses"
```

---

## Task 2: loader.py（parse Markdown → list[Section]）

**Files:**
- Create: `challenge/app/loader.py`
- Create: `challenge/tests/test_loader.py`

### 2A. slugify

- [ ] **Step 2.1: Write failing test for slugify**

加進 `tests/test_loader.py`：

```python
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
```

- [ ] **Step 2.2: Run to fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError` — `app.loader` 還沒寫。

- [ ] **Step 2.3: Implement `slugify` in `app/loader.py`**

```python
"""
Markdown loader：把 docs/*.md 切成 list[Section]。

Heading 規則（per DESIGN.md §6.2）：
- "## " 開頭那一行為 section 分界
- 如果整檔沒有任何 "## "，整檔當一個 section，heading 用檔名（去 .md）
- heading 文字過 slugify() 後變成 URL 友善的 slug、用於 citation
"""
import re
from pathlib import Path
from app.types import Section


def slugify(text: str) -> str:
    """把 heading 文字轉成 URL 友善的 slug。

    例: "Refund Timeline" -> "refund-timeline"
        "FAQ: Shipping & Returns!" -> "faq-shipping-returns"

    步驟：
    1. lowercase
    2. 非英數字元（含標點、空白）一律換成 "-"
    3. 連續 "-" 合併成單個
    4. 前後 "-" trim 掉
    """
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)   # 非英數變 "-"
    s = re.sub(r"-+", "-", s)            # 多個 "-" 合一
    return s.strip("-")
```

- [ ] **Step 2.4: Run slugify tests to pass**

```bash
pytest tests/test_loader.py::test_slugify_lowercases_and_dasherizes tests/test_loader.py::test_slugify_handles_punctuation -v
```

Expected: 2 passed.

### 2B. parse_markdown

- [ ] **Step 2.5: Write failing tests for parse_markdown**

加進 `tests/test_loader.py`：

```python
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
    # intro 文字不該出現在 body 裡
    assert "intro that nobody asked for" not in sections[0].body


def test_parse_markdown_empty_content_returns_one_section_with_empty_body():
    """空內容也不該 crash，回退用 filename。"""
    sections = parse_markdown("", "empty.md")
    assert len(sections) == 1
    assert sections[0].heading == "empty"
    assert sections[0].body == ""
```

- [ ] **Step 2.6: Run to fail**

```bash
pytest tests/test_loader.py -v
```

Expected: 4 個新 test 都失敗（`parse_markdown` 未定義）。

- [ ] **Step 2.7: Implement `parse_markdown` in `app/loader.py`**

接在 `slugify` 後面加：

```python
def parse_markdown(content: str, filename: str) -> list[Section]:
    """把單個 markdown 字串切成 list[Section]，以 ## 開頭那一行為 section 邊界。

    沒有 ## 的檔 → 整檔當一個 section，heading 用檔名（去 .md 副檔名）。

    參數:
        content: markdown 原始字串
        filename: 例 "refund_policy.md"，用於 Section.filename 跟 fallback heading

    回傳:
        list[Section]；空 content 也至少回一個空 body 的 fallback Section。
    """
    sections: list[Section] = []
    current_heading: str | None = None
    current_body_lines: list[str] = []

    def flush() -> None:
        """把累積的 body lines 收成一個 Section、加進 sections。"""
        if current_heading is not None:
            heading_text = current_heading
            sections.append(Section(
                filename=filename,
                heading=heading_text,
                heading_slug=slugify(heading_text),
                body="\n".join(current_body_lines).strip(),
            ))

    for line in content.splitlines():
        if line.startswith("## "):
            # 遇到新 section -> 先把上一個 flush 掉
            flush()
            current_heading = line[3:].strip()    # 去掉 "## " 前綴
            current_body_lines = []
        elif current_heading is not None:
            current_body_lines.append(line)
        # current_heading is None 表示在第一個 ## 之前、文字丟掉

    flush()  # 最後一個 section

    # 沒有任何 ## -> fallback：整檔當一個 section
    if not sections:
        fallback_heading = filename.removesuffix(".md")
        sections.append(Section(
            filename=filename,
            heading=fallback_heading,
            heading_slug=slugify(fallback_heading),
            body=content.strip(),
        ))

    return sections
```

- [ ] **Step 2.8: Run to pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 6 passed（含前面 2 個 slugify）。

### 2C. load_docs

- [ ] **Step 2.9: Write failing test for load_docs**

加進 `tests/test_loader.py`：

```python
def test_load_docs_reads_all_md_files(sample_docs_dir: Path):
    """conftest.py 的 sample_docs_dir 寫了兩個檔；load_docs 應該把全部 section 攤平。"""
    sections = load_docs(sample_docs_dir)

    # refunds.md 有 1 個 ## section、account.md 也是 1 個
    assert len(sections) == 2
    headings = {s.heading for s in sections}
    assert headings == {"Refund Timeline", "Reset Password"}


def test_load_docs_ignores_non_md_files(sample_docs_dir: Path):
    """目錄裡有非 .md 檔 -> 不會被讀進來。"""
    (sample_docs_dir / "ignore.txt").write_text("not markdown")

    sections = load_docs(sample_docs_dir)
    assert len(sections) == 2  # 跟前一個 test 一樣，沒多出來


def test_load_docs_missing_dir_returns_empty_list(tmp_path: Path):
    """目錄不存在 -> 回空 list、不要 raise。"""
    missing = tmp_path / "does-not-exist"
    sections = load_docs(missing)
    assert sections == []
```

- [ ] **Step 2.10: Run to fail**

```bash
pytest tests/test_loader.py -v
```

Expected: 3 個新 test 失敗。

- [ ] **Step 2.11: Implement `load_docs` in `app/loader.py`**

接在 `parse_markdown` 後面加：

```python
def load_docs(directory: Path) -> list[Section]:
    """讀目錄下所有 .md 檔、攤平成單一 list[Section]。

    參數:
        directory: docs/ 目錄路徑

    回傳:
        list[Section]，目錄不存在或沒有 .md 時回空 list（不 raise）。
    """
    if not directory.exists() or not directory.is_dir():
        return []

    sections: list[Section] = []
    # sorted 確保檔案讀取順序穩定（OS 不保證 iterdir 的順序）
    for md_path in sorted(directory.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        sections.extend(parse_markdown(content, md_path.name))
    return sections
```

- [ ] **Step 2.12: Run to pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 9 passed.

- [ ] **Step 2.13: Commit**

```bash
git add app/loader.py tests/test_loader.py
git commit -m "kb-qa-bot: parse markdown docs into Section list"
```

---

## Task 3: bm25.py（tokenize + BM25Index）

**Files:**
- Create: `challenge/app/bm25.py`
- Create: `challenge/tests/test_bm25.py`

### 3A. tokenize

- [ ] **Step 3.1: Write failing test**

`tests/test_bm25.py`：

```python
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
```

- [ ] **Step 3.2: Run to fail**

```bash
pytest tests/test_bm25.py -v
```

Expected: ImportError。

- [ ] **Step 3.3: Implement `tokenize` in `app/bm25.py`**

```python
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


# 常見英文 stopword 列表。BM25 對 stopword 不敏感（它本來就會用 IDF 壓低高頻詞），
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
    """
    # 非英數一律當 separator，re.findall 抓出所有英數 chunk
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in raw if t not in STOPWORDS]
```

- [ ] **Step 3.4: Run to pass**

```bash
pytest tests/test_bm25.py::test_tokenize_lowercases_and_splits_words tests/test_bm25.py::test_tokenize_drops_stopwords tests/test_bm25.py::test_tokenize_handles_punctuation -v
```

Expected: 3 passed.

### 3B. BM25Index

- [ ] **Step 3.5: Write failing tests for BM25Index**

加進 `tests/test_bm25.py`：

```python
def _make_sections() -> list[Section]:
    """共用 fixture：3 個 section 模擬 sample corpus。"""
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
    assert "refund" in idx.tokens[0]


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
```

- [ ] **Step 3.6: Run to fail**

```bash
pytest tests/test_bm25.py -v
```

Expected: 5 個新 test 失敗（`BM25Index` 還沒寫）。

- [ ] **Step 3.7: Implement `BM25Index`**

接在 `tokenize` 後面加：

```python
@dataclass
class BM25Index:
    """封裝 rank_bm25.BM25Okapi、保留 Section 對應。

    建議用 BM25Index.build(sections) 建立，這個 classmethod 會幫你 tokenize。
    直接呼 __init__ 通常是 store.load() 反序列化時用的（tokens 已經算過了）。
    """
    sections: list[Section]
    tokens: list[list[str]]
    # _bm25 是 internal cache；不放進 dataclass field 直接序列化，
    # 因為它是衍生物、可以從 tokens 重建。
    _bm25: BM25Okapi = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """dataclass 在 __init__ 跑完之後自動呼這個 hook。

        我們用它來建 rank_bm25 的內部物件 —— sections + tokens 由 caller 提供，
        _bm25 在這裡重建就好。
        """
        self._bm25 = BM25Okapi(self.tokens)

    @classmethod
    def build(cls, sections: list[Section]) -> "BM25Index":
        """從 Section list 建索引：tokenize 每個 body、丟給 BM25Okapi。"""
        tokens = [tokenize(s.body) for s in sections]
        return cls(sections=sections, tokens=tokens)

    def top_k(self, query: str, k: int) -> list[tuple[Section, float]]:
        """回傳分數最高的 k 個 (section, score)，分數由大到小。

        k > section 數 → 自動 clamp 成 len(sections)。
        所有 score 為 0 → 仍然回 k 個，由 retrieval 層用 threshold 過濾。
        """
        if not self.sections:
            return []

        query_tokens = tokenize(query)
        # get_scores 回 numpy array，逐個 section 的 BM25 分數
        scores = self._bm25.get_scores(query_tokens)
        # zip + sorted by score desc
        ranked = sorted(
            zip(self.sections, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(sec, float(score)) for sec, score in ranked[:k]]
```

- [ ] **Step 3.8: Run to pass**

```bash
pytest tests/test_bm25.py -v
```

Expected: 8 passed.

- [ ] **Step 3.9: Commit**

```bash
git add app/bm25.py tests/test_bm25.py
git commit -m "kb-qa-bot: build BM25 index with top-k retrieval"
```

---

## Task 4: store.py（save/load index JSON）

**Files:**
- Create: `challenge/app/store.py`
- Create: `challenge/tests/test_store.py`

- [ ] **Step 4.1: Write failing tests**

`tests/test_store.py`：

```python
"""驗證 index 序列化：save → load 拿回等價物件。"""
from pathlib import Path

import pytest

from app.bm25 import BM25Index
from app.store import save, load
from app.types import Section


def _sample_index() -> BM25Index:
    sections = [
        Section("a.md", "Alpha", "alpha", "the alpha section body"),
        Section("b.md", "Beta", "beta", "the beta section body"),
    ]
    return BM25Index.build(sections)


def test_save_then_load_roundtrips(tmp_path: Path):
    """save 後 load 回來，sections 跟 tokens 都該一致。"""
    idx = _sample_index()
    path = tmp_path / ".kb" / "index.json"

    save(idx, path)

    assert path.exists(), "save 應該建好 .kb/ 目錄並寫檔"

    loaded = load(path)
    assert loaded is not None
    assert loaded.sections == idx.sections
    assert loaded.tokens == idx.tokens


def test_load_missing_file_returns_none(tmp_path: Path):
    """檔不存在 -> 回 None、不要 raise。"""
    assert load(tmp_path / "nope.json") is None


def test_load_corrupt_json_returns_none(tmp_path: Path):
    """JSON 解析失敗 -> 回 None、不要 crash。"""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert load(bad) is None


def test_load_missing_keys_returns_none(tmp_path: Path):
    """JSON 是合法但缺欄位 -> 回 None。"""
    bad = tmp_path / "incomplete.json"
    bad.write_text('{"sections": []}', encoding="utf-8")  # 缺 tokens
    assert load(bad) is None


def test_saved_json_is_human_readable(tmp_path: Path):
    """Karpathy 風格的核心：.kb/index.json 要 cat 起來看得懂。"""
    idx = _sample_index()
    path = tmp_path / "index.json"
    save(idx, path)

    raw = path.read_text(encoding="utf-8")
    assert "alpha" in raw                   # 看得到 section heading slug
    assert "the alpha section body" in raw  # 看得到原文
    assert "\n" in raw                      # 有 indent、不是一坨
```

- [ ] **Step 4.2: Run to fail**

```bash
pytest tests/test_store.py -v
```

Expected: ImportError。

- [ ] **Step 4.3: Implement `app/store.py`**

```python
"""
BM25Index 的序列化 / 反序列化：寫 .kb/index.json 跟讀回來。

設計目的：
- index.json 要能 `cat`、人讀得懂（Karpathy 風格核心，DESIGN.md §10 提到）
- 損壞或缺檔不要 crash，回 None 讓呼叫端處理
"""
import json
from pathlib import Path

from app.bm25 import BM25Index
from app.types import Section


def save(index: BM25Index, path: Path) -> None:
    """把 BM25Index 寫成 JSON。會自動建父目錄。

    格式：
        {
          "sections": [
            {"filename": "...", "heading": "...", "heading_slug": "...", "body": "..."},
            ...
          ],
          "tokens": [["t1", "t2", ...], ...]
        }
    """
    data = {
        "sections": [
            {
                "filename": s.filename,
                "heading": s.heading,
                "heading_slug": s.heading_slug,
                "body": s.body,
            }
            for s in index.sections
        ],
        "tokens": index.tokens,
    }
    # parents=True 連同中間缺的目錄一起建；exist_ok=True 已存在不抱怨
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 + ensure_ascii=False 讓檔案人讀得舒服、中文不會被 \u 編碼掉
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(path: Path) -> BM25Index | None:
    """從 JSON 載回 BM25Index。檔不存在 / 解析失敗 / 欄位缺都回 None、不 raise。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sections = [Section(**raw) for raw in data["sections"]]
        tokens = data["tokens"]
    except (json.JSONDecodeError, KeyError, TypeError):
        # JSONDecodeError: 不是合法 JSON
        # KeyError: 缺 sections 或 tokens
        # TypeError: section 欄位對不上 Section dataclass
        return None
    return BM25Index(sections=sections, tokens=tokens)
```

- [ ] **Step 4.4: Run to pass**

```bash
pytest tests/test_store.py -v
```

Expected: 5 passed.

- [ ] **Step 4.5: Commit**

```bash
git add app/store.py tests/test_store.py
git commit -m "kb-qa-bot: serialize BM25 index to inspectable JSON"
```

---

## Task 5: retrieval.py（search + threshold）

**Files:**
- Create: `challenge/app/retrieval.py`
- Create: `challenge/tests/test_retrieval.py`

- [ ] **Step 5.1: Write failing tests**

`tests/test_retrieval.py`：

```python
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
```

- [ ] **Step 5.2: Run to fail**

```bash
pytest tests/test_retrieval.py -v
```

Expected: ImportError。

- [ ] **Step 5.3: Implement `app/retrieval.py`**

```python
"""
Retrieval 層：把 BM25 排名跟 threshold 結合，產出 RetrievalResult。

threshold 邏輯（DESIGN.md §6.1）：
- 如果 top-1 score < threshold → 視為「沒有相關內容」、回 fallback=True
- fallback=True 時 呼叫端應該直接回 cannot-confirm、不要再呼 LLM
"""
from app.bm25 import BM25Index
from app.types import RetrievalResult


def search(
    query: str,
    index: BM25Index,
    k: int,
    threshold: float,
) -> RetrievalResult:
    """檢索 + threshold 判斷。

    參數:
        query: 使用者問題原文
        index: 已 build 好的 BM25Index
        k: 想要的 top-K 數量
        threshold: top-1 score 至少要超過這個值，否則 fallback

    回傳:
        RetrievalResult；fallback=True 時 sections / scores 都是空 list。
    """
    ranked = index.top_k(query, k)

    # 沒有任何 section（空 index）或 top-1 分數太低 -> fallback
    if not ranked or ranked[0][1] < threshold:
        return RetrievalResult(sections=[], scores=[], fallback=True)

    # 拆 tuple list 成兩個平行 list；list comprehension 是 Python 慣用寫法
    sections = [pair[0] for pair in ranked]
    scores = [pair[1] for pair in ranked]
    return RetrievalResult(sections=sections, scores=scores, fallback=False)
```

- [ ] **Step 5.4: Run to pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: 4 passed.

- [ ] **Step 5.5: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "kb-qa-bot: retrieval with score threshold fallback"
```

---

## Task 6: prompt.py（build LLM messages）

**Files:**
- Create: `challenge/app/prompt.py`
- Create: `challenge/tests/test_prompt.py`

- [ ] **Step 6.1: Write failing tests**

`tests/test_prompt.py`：

```python
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
    # 至少要有「只能根據 context」「cite 用 filename#heading 格式」「找不到要誠實」這三個概念
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
```

- [ ] **Step 6.2: Run to fail**

```bash
pytest tests/test_prompt.py -v
```

Expected: ImportError。

- [ ] **Step 6.3: Implement `app/prompt.py`**

```python
"""
Prompt 組裝：把 retrieved sections + user query 變成 OpenAI 的 messages list。

格式遵守 OpenAI chat completion 慣例：
    [
        {"role": "system", "content": "<grounding instruction>"},
        {"role": "user",   "content": "<context + question>"},
    ]
"""
from app.types import Section


_SYSTEM_PROMPT = (
    "You are a customer support assistant answering strictly from the provided "
    "knowledge base sections below. Rules:\n"
    "1. Only use facts that appear in the context. Do not invent details.\n"
    "2. Cite every source you used in the 'sources' field using the exact "
    "[filename#heading] tag shown next to each section.\n"
    "3. If the context does not contain the answer, reply that you cannot "
    "confirm this from the knowledge base, and leave 'sources' empty.\n"
)


def build_messages(query: str, sections: list[Section]) -> list[dict]:
    """組成 OpenAI chat completion 的 messages list。

    參數:
        query: 使用者原問題
        sections: retrieval.search 拿到的 top-K sections

    回傳:
        2 個 dict 的 list：system instruction + user content。
    """
    # 把每個 section 排成 "[citation]\nbody" 的 block、用空行分隔
    context_blocks = [
        f"[{s.citation}]\n{s.body}"
        for s in sections
    ]
    context = "\n\n".join(context_blocks)

    user_content = f"Context:\n{context}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
```

- [ ] **Step 6.4: Run to pass**

```bash
pytest tests/test_prompt.py -v
```

Expected: 5 passed.

- [ ] **Step 6.5: Commit**

```bash
git add app/prompt.py tests/test_prompt.py
git commit -m "kb-qa-bot: build grounded LLM prompt with citation tags"
```

---

## Task 7: llm.py（OpenAI structured output client）

**Files:**
- Create: `challenge/app/llm.py`
- Create: `challenge/tests/test_llm.py`

- [ ] **Step 7.1: Write failing tests**

`tests/test_llm.py`：

```python
"""驗證 LLMClient：透過 monkeypatch mock OpenAI client，不打真實 API。"""
from unittest.mock import MagicMock

import pytest

from app.llm import LLMClient, LLMResponse


@pytest.fixture
def mock_openai(monkeypatch):
    """把 OpenAI() 換成 MagicMock；test 拿到 mock 物件可以 assert 呼叫紀錄。"""
    mock_client = MagicMock()
    # llm.py 裡會做 OpenAI(api_key=...)；攔截 constructor
    monkeypatch.setattr("app.llm.OpenAI", lambda **kwargs: mock_client)
    return mock_client


def test_ask_returns_parsed_response(mock_openai):
    """LLM 回的 parsed 物件要原樣傳出去。"""
    # 模擬 OpenAI 回傳結構：completion.choices[0].message.parsed
    fake_parsed = LLMResponse(answer="5-7 business days", sources=["refund.md#refund-timeline"])
    fake_message = MagicMock(parsed=fake_parsed)
    fake_choice = MagicMock(message=fake_message)
    fake_completion = MagicMock(choices=[fake_choice])
    mock_openai.beta.chat.completions.parse.return_value = fake_completion

    client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
    result = client.ask([{"role": "user", "content": "anything"}])

    assert result.answer == "5-7 business days"
    assert result.sources == ["refund.md#refund-timeline"]


def test_ask_passes_messages_and_model_to_openai(mock_openai):
    """messages 跟 model 要原封不動傳給 OpenAI。"""
    fake_completion = MagicMock(choices=[MagicMock(message=MagicMock(parsed=LLMResponse(answer="x", sources=[])))])
    mock_openai.beta.chat.completions.parse.return_value = fake_completion

    client = LLMClient(api_key="sk-test", model="gpt-4o")
    messages = [{"role": "user", "content": "hi"}]
    client.ask(messages)

    # 檢查呼叫參數
    call_kwargs = mock_openai.beta.chat.completions.parse.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["messages"] == messages
    assert call_kwargs["response_format"] is LLMResponse
```

- [ ] **Step 7.2: Run to fail**

```bash
pytest tests/test_llm.py -v
```

Expected: ImportError。

- [ ] **Step 7.3: Implement `app/llm.py`**

```python
"""
OpenAI 結構化輸出 client。

用 openai SDK 的 beta.chat.completions.parse() API + pydantic BaseModel 強制輸出格式 ——
這樣 LLM 一定回 {answer: str, sources: list[str]}、不會漏 sources、不會亂格式。

⚠️ openai SDK 的 structured-output API 持續演進；實作時可能需要對照 OpenAI 官方文件確認
   current call shape。我們鎖 `openai>=1.40` 是因為 1.40 之後 parse() 已穩定。
"""
from openai import OpenAI
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """強制 LLM 回傳的 schema。pydantic 會自動轉成 OpenAI 需要的 JSON Schema。"""
    answer: str
    sources: list[str]


class LLMClient:
    """OpenAI chat completion 的薄包裝。"""

    def __init__(self, api_key: str, model: str):
        """初始化。

        參數:
            api_key: OpenAI API key（從 config 來）
            model: 模型名，例如 "gpt-4o-mini"
        """
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def ask(self, messages: list[dict]) -> LLMResponse:
        """呼叫 OpenAI、強制 structured output。

        參數:
            messages: prompt.build_messages() 的回傳值

        回傳:
            LLMResponse(answer, sources)；OpenAI 一定回符合 schema 的物件。

        例外:
            openai SDK 自己的 exceptions（網路失敗、認證、rate limit）由 caller 處理。
        """
        completion = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=LLMResponse,
        )
        # parse() 已經把 message.content 反序列化進 .parsed
        return completion.choices[0].message.parsed
```

- [ ] **Step 7.4: Run to pass**

```bash
pytest tests/test_llm.py -v
```

Expected: 2 passed.

- [ ] **Step 7.5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "kb-qa-bot: OpenAI client with structured-output enforcement"
```

---

## Task 8: config.py（env vars 集中讀取）

**Files:**
- Create: `challenge/app/config.py`
- Create: `challenge/tests/test_config.py`

- [ ] **Step 8.1: Write failing tests**

`tests/test_config.py`：

```python
"""驗證 config 從 env var 讀取、提供預設值、缺必要欄位時 raise。"""
import pytest

from app.config import Config, load_config


def test_load_config_uses_env_values(monkeypatch):
    """env var 有值 -> 用 env 的值。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "1.5")
    monkeypatch.setenv("KB_DOCS_DIR", "/custom/docs")
    monkeypatch.setenv("KB_INDEX_PATH", "/custom/index.json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENV_NAME", "qat")

    cfg = load_config()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_model == "gpt-4o"
    assert cfg.bm25_score_threshold == 1.5
    assert str(cfg.kb_docs_dir) == "/custom/docs"
    assert str(cfg.kb_index_path) == "/custom/index.json"
    assert cfg.log_level == "DEBUG"
    assert cfg.env_name == "qat"


def test_load_config_uses_defaults(monkeypatch):
    """除了 OPENAI_API_KEY 必填，其他都該有合理預設值。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # 確保其他 env 是空的
    for k in ("OPENAI_MODEL", "BM25_SCORE_THRESHOLD", "KB_DOCS_DIR",
              "KB_INDEX_PATH", "LOG_LEVEL", "ENV_NAME"):
        monkeypatch.delenv(k, raising=False)

    cfg = load_config()
    assert cfg.openai_model == "gpt-4o-mini"
    assert cfg.bm25_score_threshold == 0.5
    assert cfg.log_level == "INFO"
    assert cfg.env_name == "local"


def test_load_config_missing_api_key_raises(monkeypatch):
    """OPENAI_API_KEY 沒設 -> 直接 raise、不要默默讓 server 起來。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        load_config()
```

- [ ] **Step 8.2: Run to fail**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError。

- [ ] **Step 8.3: Implement `app/config.py`**

```python
"""
集中讀取 env vars 的 config 模組。

設計理念（DESIGN.md §10.1）：
- 啟動時 load_dotenv() 把 .env 內容塞進 os.environ（本機開發）
- K8s 部署時 .env 不存在、load_dotenv() 靜默 no-op；env vars 由 K8s 注入
- App 一律從 os.environ 讀，不關心來源
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """所有 runtime 設定值。frozen=True 表示建構後不可改、避免被誤動。"""
    openai_api_key: str
    openai_model: str
    bm25_score_threshold: float
    kb_docs_dir: Path
    kb_index_path: Path
    log_level: str
    env_name: str


def load_config() -> Config:
    """讀 .env (有的話) + os.environ，組成 Config 物件。

    raise RuntimeError 如果 OPENAI_API_KEY 缺。
    """
    # load_dotenv 找 .env 檔（cwd 開始往上找），把裡面的 KEY=VALUE 塞進 os.environ。
    # 預設不覆寫 already-set 的環境變數 -> K8s 注入的值優先於 .env。
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Set it in .env (local) or "
            "via K8s Secret (production)."
        )

    return Config(
        openai_api_key=api_key,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        bm25_score_threshold=float(os.environ.get("BM25_SCORE_THRESHOLD", "0.5")),
        kb_docs_dir=Path(os.environ.get("KB_DOCS_DIR", "../docs")),
        kb_index_path=Path(os.environ.get("KB_INDEX_PATH", ".kb/index.json")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        env_name=os.environ.get("ENV_NAME", "local"),
    )
```

- [ ] **Step 8.4: Run to pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 8.5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "kb-qa-bot: centralized env-var config with fail-fast on missing key"
```

---

## Task 9: main.py（FastAPI app + 3 endpoints + integration test）

**Files:**
- Create: `challenge/app/main.py`
- Create: `challenge/tests/test_api.py`

### 9A. /health

- [ ] **Step 9.1: Write failing test for /health**

`tests/test_api.py`：

```python
"""FastAPI integration test。

用 fastapi.testclient.TestClient 模擬 HTTP 呼叫；OpenAI client 用 monkeypatch mock。
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_factory(monkeypatch, sample_docs_dir, tmp_path):
    """工廠 fixture：建一個 FastAPI app instance，並 mock 掉外部依賴。

    回傳一個 callable，呼叫它拿到 (TestClient, mock_llm_ask)。
    """
    def _build(initial_index: bool = False):
        # 1. 設好 env var，讓 load_config() 拿到測試用的值
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("BM25_SCORE_THRESHOLD", "0.1")  # 寬鬆好驗 grounded path
        monkeypatch.setenv("KB_DOCS_DIR", str(sample_docs_dir))
        monkeypatch.setenv("KB_INDEX_PATH", str(tmp_path / ".kb" / "index.json"))

        # 2. mock 掉 LLMClient.ask；要在 import main 之前 patch
        mock_ask = MagicMock()
        monkeypatch.setattr("app.llm.LLMClient.ask", mock_ask)

        # 3. import 並建 app
        from importlib import reload
        from app import main
        reload(main)  # 確保拿到最新 env 變數

        # 4. 如果 test 想要預先有 index，先呼一次 /index
        client = TestClient(main.app)
        if initial_index:
            client.post("/index")
        return client, mock_ask

    return _build


def test_health_returns_ok(app_factory):
    client, _ = app_factory()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 9.2: Run to fail**

```bash
pytest tests/test_api.py -v
```

Expected: ImportError（`app.main` 還沒寫）。

- [ ] **Step 9.3: Implement minimal `app/main.py` with /health**

```python
"""
FastAPI app entry point。組裝所有模組、暴露 3 個 endpoint：

- GET  /health  → 健康檢查
- POST /index   → 讀 docs/、建 BM25 index、寫 .kb/index.json
- POST /chat    → query → retrieval → LLM（或 cannot-confirm）

startup hook 會嘗試自動載入既有的 .kb/index.json（若存在）；不存在就讓 /chat 友善回應。
"""
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import load_config
from app.bm25 import BM25Index
from app.llm import LLMClient
from app.loader import load_docs
from app.prompt import build_messages
from app.retrieval import search
from app.store import save, load


# ============ 啟動初始化 ============

# 設 logger；level 之後從 config 套
logger = logging.getLogger(__name__)

_config = load_config()
logging.basicConfig(level=_config.log_level)

# state：用 dict 而不是 module-level variable，方便 test reload。
state: dict[str, Any] = {
    "config": _config,
    "index": None,           # BM25Index | None
    "llm": LLMClient(api_key=_config.openai_api_key, model=_config.openai_model),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 啟動 / 關閉 hook。Startup: 嘗試載入既有 index。"""
    existing = load(state["config"].kb_index_path)
    if existing is not None:
        state["index"] = existing
        logger.info(f"loaded index from {state['config'].kb_index_path}: "
                    f"{len(existing.sections)} sections")
    else:
        logger.info("no existing index; /chat will return 'not indexed' "
                    "until /index is called")
    yield
    # 沒有特別的 shutdown 工作


app = FastAPI(lifespan=lifespan)


# ============ Endpoints ============

@app.get("/health")
def health() -> dict[str, str]:
    """簡單 liveness 檢查。"""
    return {"status": "ok"}
```

- [ ] **Step 9.4: Run test to pass**

```bash
pytest tests/test_api.py::test_health_returns_ok -v
```

Expected: 1 passed.

### 9B. /index

- [ ] **Step 9.5: Write failing test for /index**

加進 `tests/test_api.py`：

```python
def test_index_endpoint_builds_and_persists(app_factory, tmp_path):
    client, _ = app_factory()

    r = client.post("/index")
    assert r.status_code == 200

    body = r.json()
    assert body["files_indexed"] == 2          # conftest 寫 2 個檔
    assert body["sections_indexed"] == 2       # 各 1 個 ## section

    # index 檔被寫出來
    index_path = tmp_path / ".kb" / "index.json"
    assert index_path.exists()
```

- [ ] **Step 9.6: Run to fail**

```bash
pytest tests/test_api.py::test_index_endpoint_builds_and_persists -v
```

Expected: 404（endpoint 還沒寫）。

- [ ] **Step 9.7: Add /index in `app/main.py`**

接在 `health()` 後面加：

```python
class IndexResponse(BaseModel):
    """POST /index 的回應格式。"""
    files_indexed: int
    sections_indexed: int


@app.post("/index", response_model=IndexResponse)
def build_index() -> IndexResponse:
    """讀 KB_DOCS_DIR 下所有 .md、建 BM25 index、寫到 KB_INDEX_PATH。"""
    cfg = state["config"]
    sections = load_docs(cfg.kb_docs_dir)
    index = BM25Index.build(sections)
    save(index, cfg.kb_index_path)
    state["index"] = index

    # 算有多少不同檔案
    files_indexed = len({s.filename for s in sections})
    logger.info(f"indexed {files_indexed} files, {len(sections)} sections "
                f"→ {cfg.kb_index_path}")
    return IndexResponse(
        files_indexed=files_indexed,
        sections_indexed=len(sections),
    )
```

- [ ] **Step 9.8: Run to pass**

```bash
pytest tests/test_api.py -v
```

Expected: 2 passed.

### 9C. /chat

- [ ] **Step 9.9: Write failing tests for /chat (3 paths)**

加進 `tests/test_api.py`：

```python
from app.llm import LLMResponse


def test_chat_before_index_returns_friendly_message(app_factory):
    """/index 之前呼 /chat -> 友善訊息、不要 500。"""
    client, mock_ask = app_factory()  # 不呼 /index

    r = client.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 200
    body = r.json()
    assert "not been indexed" in body["answer"].lower() \
        or "not indexed" in body["answer"].lower()
    assert body["sources"] == []
    # LLM 不該被呼叫
    mock_ask.assert_not_called()


def test_chat_grounded_query_calls_llm_and_returns_citation(app_factory):
    """正常 query -> 命中 retrieval -> 呼 LLM -> 回 answer + sources。"""
    client, mock_ask = app_factory(initial_index=True)

    # mock LLM 回一個合理結果
    mock_ask.return_value = LLMResponse(
        answer="Approved refunds are processed within 5-7 business days.",
        sources=["refunds.md#refund-timeline"],
    )

    r = client.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 200
    body = r.json()
    assert "5-7 business days" in body["answer"]
    assert "refunds.md#refund-timeline" in body["sources"]
    mock_ask.assert_called_once()


def test_chat_out_of_scope_query_returns_cannot_confirm_without_calling_llm(
    app_factory, monkeypatch
):
    """完全沒命中的 query -> fallback、不呼 LLM。"""
    # 把 threshold 拉高，讓任何 query 都過不了
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "999.0")
    client, mock_ask = app_factory(initial_index=True)

    r = client.post("/chat", json={"query": "Which restaurants are nearby?"})
    assert r.status_code == 200
    body = r.json()
    assert "cannot confirm" in body["answer"].lower()
    assert body["sources"] == []
    mock_ask.assert_not_called()
```

- [ ] **Step 9.10: Run to fail**

```bash
pytest tests/test_api.py -v
```

Expected: 3 個新 test 失敗（/chat endpoint 沒寫）。

- [ ] **Step 9.11: Add /chat in `app/main.py`**

接在 `build_index()` 後面加：

```python
class ChatRequest(BaseModel):
    """POST /chat 的 body。"""
    query: str


class ChatResponse(BaseModel):
    """POST /chat 的回應。"""
    answer: str
    sources: list[str]


_NOT_INDEXED_MSG = (
    "The knowledge base has not been indexed yet. "
    "Please POST /index before asking questions."
)
_CANNOT_CONFIRM_MSG = (
    "I cannot confirm this from the knowledge base."
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """以 grounded 方式回答問題。三種路徑：

    1. 還沒 /index → 友善訊息
    2. 命中 retrieval（top-1 score >= threshold）→ 呼 LLM、回 answer + sources
    3. 沒命中 → cannot-confirm、不呼 LLM
    """
    cfg = state["config"]
    index = state["index"]

    # Path 1: 未建索引
    if index is None:
        logger.info(f"/chat called before /index: query={req.query!r}")
        return ChatResponse(answer=_NOT_INDEXED_MSG, sources=[])

    # 跑 retrieval
    result = search(
        query=req.query,
        index=index,
        k=3,
        threshold=cfg.bm25_score_threshold,
    )

    # Path 3: 弱檢索 → fallback、不呼 LLM
    if result.fallback:
        logger.info(f"/chat fallback (no section above threshold): query={req.query!r}")
        return ChatResponse(answer=_CANNOT_CONFIRM_MSG, sources=[])

    # Path 2: 命中 → 呼 LLM
    messages = build_messages(req.query, result.sections)
    llm_response = state["llm"].ask(messages)
    logger.info(
        f"/chat answered: query={req.query!r} "
        f"top_score={result.scores[0]:.3f} "
        f"sources={llm_response.sources}"
    )
    return ChatResponse(
        answer=llm_response.answer,
        sources=llm_response.sources,
    )
```

- [ ] **Step 9.12: Run to pass**

```bash
pytest tests/test_api.py -v
```

Expected: 5 passed（health + index + 3 個 chat path）。

### 9D. Index persistence test

- [ ] **Step 9.13: Add startup-load test**

加進 `tests/test_api.py`：

```python
def test_startup_loads_existing_index(app_factory, sample_docs_dir, tmp_path, monkeypatch):
    """建 index、銷掉 app、重新起一個 app -> 不需要再呼 /index。"""
    # 第一次：建 index
    client1, _ = app_factory(initial_index=True)
    assert (tmp_path / ".kb" / "index.json").exists()

    # 第二次：用同樣的 KB_INDEX_PATH 起一個新 client，但不呼 /index
    client2, mock_ask = app_factory()

    # mock LLM 防止真的打 API
    mock_ask.return_value = LLMResponse(answer="x", sources=["refunds.md#refund-timeline"])

    r = client2.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 200
    # 重點：沒呼 /index 還是能正常回應 -> 證明 startup 載入成功
    assert "not been indexed" not in r.json()["answer"].lower()
```

- [ ] **Step 9.14: Run all tests to pass**

```bash
pytest tests/ -v
```

Expected: 全綠（types + loader + bm25 + store + retrieval + prompt + llm + config + api 全部加總大約 30+ 個 test）。

- [ ] **Step 9.15: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "kb-qa-bot: FastAPI app with /health /index /chat + lifespan load"
```

---

## Task 10: build_index.py + Dockerfile（build-time bake）

**Files:**
- Create: `challenge/app/build_index.py`
- Create: `challenge/Dockerfile`
- Move: `challenge/.dockerignore` → `knowledge_base_qa_bot/.dockerignore`

- [ ] **Step 10.1: Create `challenge/app/build_index.py`（CLI script）**

```python
"""
Docker build 階段執行的 CLI：讀 docs/、建 BM25 index、寫到指定路徑。

跟 FastAPI 沒關係 —— 純粹「load_docs + BM25Index.build + save」的 shell。
Dockerfile 在 builder stage 跑這個、把產物 COPY 進 runtime image。

用法:
    python -m app.build_index --docs-dir docs/ --output .kb/index.json
"""
import argparse
import sys
from pathlib import Path

from app.bm25 import BM25Index
from app.loader import load_docs
from app.store import save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BM25 index from docs/")
    parser.add_argument("--docs-dir", type=Path, required=True,
                        help="Directory containing .md files to index")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for index.json")
    args = parser.parse_args(argv)

    sections = load_docs(args.docs_dir)
    if not sections:
        print(f"warning: no .md files found in {args.docs_dir}",
              file=sys.stderr)
        return 1

    index = BM25Index.build(sections)
    save(index, args.output)

    files = len({s.filename for s in sections})
    print(f"indexed {files} files, {len(sections)} sections → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 10.2: Manual verify the CLI works**

```bash
cd knowledge_base_qa_bot/challenge
python -m app.build_index --docs-dir ../docs --output /tmp/test-index.json
```

Expected output: `indexed 3 files, 9 sections → /tmp/test-index.json`，並 `cat /tmp/test-index.json` 能看到結構。

- [ ] **Step 10.3: Move `.dockerignore` to parent**

```bash
mv challenge/.dockerignore knowledge_base_qa_bot/.dockerignore
```

（build context 在 `knowledge_base_qa_bot/`，`.dockerignore` 要放在那層。）

- [ ] **Step 10.4: Create `challenge/Dockerfile`**

```dockerfile
# === Stage 1: builder ===
# 用 slim 版減少 image 體積；只在 builder 跑、最後不會留在 runtime
FROM python:3.12-slim AS builder

WORKDIR /build

# 先裝依賴。先 copy requirements.txt 才 copy code 是 Docker layer cache 慣例：
# 改 code 不會讓 pip install 整個 layer 重跑
COPY challenge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + KB 都帶進來
COPY challenge/app/ ./app/
COPY docs/ ./docs/

# Build 階段就把 index 算好
RUN python -m app.build_index \
    --docs-dir docs/ \
    --output .kb/index.json

# === Stage 2: runtime ===
FROM python:3.12-slim

WORKDIR /app

# 跟 builder 一樣裝依賴（但不帶 build 工具，image 比較精簡）
COPY challenge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 從 builder 拿三樣產物
COPY --from=builder /build/app/  ./app/
COPY --from=builder /build/docs/ ./docs/
COPY --from=builder /build/.kb/  ./.kb/

# Runtime 預設參數；K8s manifest 可覆寫
ENV KB_DOCS_DIR=/app/docs \
    KB_INDEX_PATH=/app/.kb/index.json \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 10.5: Build the Docker image**

```bash
cd knowledge_base_qa_bot
docker build -f challenge/Dockerfile -t kb-qa-bot:dev .
```

Expected: build 成功、最後一行有 `naming to docker.io/library/kb-qa-bot:dev done`。

- [ ] **Step 10.6: Smoke-run the container**

```bash
docker run --rm -e OPENAI_API_KEY=dummy -p 8000:8000 kb-qa-bot:dev
```

另開一個 terminal：
```bash
curl http://localhost:8000/health
# 預期: {"status":"ok"}
```

`Ctrl-C` 停掉 container。

- [ ] **Step 10.7: Commit**

```bash
git add knowledge_base_qa_bot/.dockerignore \
        knowledge_base_qa_bot/challenge/app/build_index.py \
        knowledge_base_qa_bot/challenge/Dockerfile
git commit -m "kb-qa-bot: Dockerfile bakes index at build time"
```

---

## Task 11: smoke.sh + 最終驗收

**Files:**
- Create: `challenge/scripts/smoke.sh`

- [ ] **Step 11.1: Create `challenge/scripts/smoke.sh`**

```bash
#!/usr/bin/env bash
# PROMPT.md L60-105 的 7 個 curl 驗證，照順序跑、檢查回應。
# 用法: 先 export OPENAI_API_KEY=sk-... 並啟動 server，然後 bash scripts/smoke.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"

echo "=== 1. /health ==="
curl -sS "$BASE/health"
echo

echo "=== 2. /chat before /index (應該回 not indexed) ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "How long do refunds take?"}'
echo

echo "=== 3. /index ==="
curl -sS -X POST "$BASE/index"
echo

echo "=== 4. cat .kb/index.json (前 30 行) ==="
head -n 30 .kb/index.json || true
echo

echo "=== 5. /chat grounded: refunds ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "How long do refunds take?"}'
echo

echo "=== 6. /chat grounded: email ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "Can I change my email address?"}'
echo

echo "=== 7. /chat out-of-scope: restaurants ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "Which restaurants are nearby?"}'
echo

echo "=== Smoke test done. Review output above against PROMPT.md expectations. ==="
```

加可執行權限（Linux/macOS）：
```bash
chmod +x scripts/smoke.sh
```

Windows 用 git bash 或 WSL 跑。

- [ ] **Step 11.2: Run smoke test locally (真實打 OpenAI)**

```bash
cd knowledge_base_qa_bot/challenge
# 已經 venv activated、有 .env
uvicorn app.main:app --port 8000 &
sleep 2
bash scripts/smoke.sh
# 看完 Ctrl-C 殺掉 uvicorn
```

逐項對照 PROMPT.md 的「Expected」欄位：
- `/health` → `{"status":"ok"}`
- 第一次 `/chat` → 訊息提示未 indexed
- `/index` → `{"files_indexed":3,"sections_indexed":9}`
- `cat .kb/index.json` → 人讀得懂、有 sections + tokens 兩個 key
- "How long do refunds take?" → 回答含 "5-7 business days"、`sources` 含 `refund_policy.md#refund-timeline`
- "Can I change my email address?" → 回答指向 Account Settings、`sources` 含 `account_help.md#change-email-address`
- "Which restaurants are nearby?" → 回答含 "cannot confirm"、`sources: []`

- [ ] **Step 11.3: 調 threshold（如果 out-of-scope query 沒成功 fallback）**

在 server log 裡看 grounded query 的 `top_score` 跟 out-of-scope 的 `top_score`：
- grounded 的 top_score 通常 > 1.0
- restaurants 的 top_score 通常 = 0.0 或極低

把 `BM25_SCORE_THRESHOLD` 調到中間（例如 0.3），確保 grounded 通過、restaurants 被擋下來。改 `.env`、重啟 server、再跑一次 smoke.sh。

- [ ] **Step 11.4: 全套 unit + integration test 再跑一遍**

```bash
pytest tests/ -v
```

Expected: 全綠。

- [ ] **Step 11.5: Commit**

```bash
git add scripts/smoke.sh .env  # 注意 .env 應在 .gitignore，不會被 add 進去
# 如果 threshold 在 .env 改過，那是本機事；不要 commit .env
git add scripts/smoke.sh
git commit -m "kb-qa-bot: add smoke.sh covering PROMPT.md verification"
```

- [ ] **Step 11.6: 最終驗收 (Definition of Done checklist)**

對照 DESIGN.md §12：

- [ ] PROMPT.md L60-105 的 7 個 curl test 全部通過
- [ ] `pytest tests/ -v` 全綠
- [ ] `.kb/index.json` 可以 `cat`、人能讀
- [ ] 重啟 server 後不需要重 index（startup hook 自動載入）
- [ ] Out-of-scope query 不呼 LLM、直接回 cannot-confirm
- [ ] 沒有 API key 或敏感資料進版控（`git log -p` 檢查）
- [ ] Dockerfile build 成功、image 起來能 serve `/health`

全 ✅ → 推 branch，開 MR / PR 給 reviewer。

---

## 自我檢視 / Reviewer Checkpoints

每完成一個 Task 後，停下來檢查：

1. **新增 test 都跑得過嗎？** `pytest tests/test_<task>.py -v` 應該全綠
2. **舊 test 還是綠的嗎？** `pytest tests/ -v` 不能有 regression
3. **Commit message 有沒有遵循格式 `kb-qa-bot: <verb> <object>`？**
4. **`.env`、`.kb/` 沒被 add 進 git 吧？** `git status` 確認
5. **DESIGN.md 還有沒有沒實作的東西？**（最後 Task 之後做一次完整 trace）

## Out of Scope（依 DESIGN.md §13）

本 plan **不包含**：streaming endpoint、browser UI、multi-format import、CLI/MCP 介面、wiki 生成、answer filing、conversation memory、runtime mutation endpoint、Elasticsearch/Tantivy 持久化。
這些是 stretch goals，core 完成後再開新 plan 處理。
