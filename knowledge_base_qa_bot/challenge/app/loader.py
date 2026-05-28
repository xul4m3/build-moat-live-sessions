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
    1. lowercase：str.lower() 把所有大寫轉小寫
    2. 非英數字元（含標點、空白）一律換成 "-"：
       re.sub(pattern, replacement, string) 把符合 pattern 的地方換成 replacement。
       r"[^a-z0-9]+" 是 regex：[^...] 表示「不在這個集合裡的字元」，+ 表示「一個以上」
    3. 連續 "-" 合併（其實步驟 2 的 + 已經合併了，這行是保險）
    4. str.strip("-") 去掉開頭結尾多餘的 "-"
    """
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)   # 非英數換成 "-"（+ 讓連續標點一次換掉）
    s = re.sub(r"-+", "-", s)            # 多個 "-" 合一（防禦性保險）
    return s.strip("-")                  # 去掉首尾 "-"


def parse_markdown(content: str, filename: str) -> list[Section]:
    """把單個 markdown 字串切成 list[Section]，以 ## 開頭那一行為 section 邊界。

    沒有 ## 的檔 → 整檔當一個 section，heading 用檔名（去 .md 副檔名）。

    參數:
        content: markdown 原始字串
        filename: 例 "refund_policy.md"，用於 Section.filename 跟 fallback heading

    回傳:
        list[Section]；空 content 也至少回一個空 body 的 fallback Section。

    Closure 說明：
        flush() 是定義在 parse_markdown 內部的 inner function（巢狀函式）。
        它可以「讀取」外層函式的 current_heading 和 current_body_lines，
        但無法「重新賦值」外層的變數（那需要 nonlocal 關鍵字）。
        這裡不需要 nonlocal，因為：
        - sections.append() 是「修改 list 的內容」，不是重新賦值 sections 本身
        - flush() 只讀取 current_heading / current_body_lines，不重新賦值它們
        - 重新賦值（current_heading = ...、current_body_lines = []）發生在 flush()「外面」，
          在外層 for 迴圈裡，所以不需要 nonlocal
    """
    sections: list[Section] = []
    current_heading: str | None = None   # None 表示還沒遇到第一個 ## 行
    current_body_lines: list[str] = []   # 累積目前 section 的 body 各行

    def flush() -> None:
        """把累積的 body lines 收成一個 Section、加進 sections。

        flush() 是 closure：它捕捉外層的 current_heading、current_body_lines、sections。
        呼叫時機：遇到新的 ## 行（先 flush 舊的）+ 整個迴圈跑完（flush 最後一個）。
        """
        if current_heading is not None:
            heading_text = current_heading
            # "\n".join(list) 把 list 的元素用換行符接起來，還原成多行字串
            sections.append(Section(
                filename=filename,
                heading=heading_text,
                heading_slug=slugify(heading_text),
                body="\n".join(current_body_lines).strip(),  # strip() 去掉多餘空白行
            ))

    # content.splitlines() 把多行字串切成逐行 list，不帶換行符
    for line in content.splitlines():
        if line.startswith("## "):
            # 遇到新 section -> 先把上一個 flush 掉，再重設 heading + body
            flush()
            current_heading = line[3:].strip()   # line[3:] 去掉 "## " 前綴（3 個字元）
            current_body_lines = []              # 重置 body 收集器
        elif current_heading is not None:
            # 在某個 ## section 內的普通行 -> 累進 body
            current_body_lines.append(line)
        # current_heading is None：還在第一個 ## 之前（H1、前言），整個丟掉

    flush()  # 迴圈結束後還有最後一個 section 沒 flush

    # 整個檔沒有任何 ## -> sections 還是空的，用 filename 當 fallback heading
    if not sections:
        # str.removesuffix() 是 Python 3.9+ 的字串方法，去掉特定後綴
        fallback_heading = filename.removesuffix(".md")
        sections.append(Section(
            filename=filename,
            heading=fallback_heading,
            heading_slug=slugify(fallback_heading),
            body=content.strip(),   # 整個 content 當 body；空字串 strip 還是空字串
        ))

    return sections


def load_docs(directory: Path) -> list[Section]:
    """讀目錄下所有 .md 檔、攤平成單一 list[Section]。

    參數:
        directory: docs/ 目錄路徑（Path 物件，代表檔案系統路徑）

    回傳:
        list[Section]，目錄不存在或沒有 .md 時回空 list（不 raise）。

    Path API 說明：
        directory.exists()       → 路徑存在嗎（可以是檔案或目錄）
        directory.is_dir()       → 是目錄嗎
        directory.glob("*.md")   → 列出符合 glob pattern 的路徑 iterator
                                   "*.md" 匹配所有 .md 檔（不遞迴子目錄）
        md_path.read_text(...)   → 讀檔案內容成字串
        md_path.name             → 只取檔名部分，例如 Path("/docs/refund.md").name == "refund.md"
    """
    if not directory.exists() or not directory.is_dir():
        return []

    sections: list[Section] = []
    # sorted() 確保跨 OS 的檔案讀取順序一致（glob 不保證回傳順序）
    for md_path in sorted(directory.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        # list.extend() 把 parse_markdown 回傳的 list 展開加進 sections
        # （跟 append 差別：append 加一個元素，extend 加多個）
        sections.extend(parse_markdown(content, md_path.name))
    return sections
