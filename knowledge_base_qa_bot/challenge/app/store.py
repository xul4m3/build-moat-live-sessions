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
    # dict comprehension：[{...} for s in index.sections]
    # 把每個 Section dataclass 的欄位手動轉成 dict
    # （JSON 只認識 dict / list / str / int / float / bool / None，
    #  不認識自訂 class，所以要自己攤平）
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
    # path.parent：取路徑的上層目錄（例如 .kb/index.json 的 parent 是 .kb/）
    # mkdir(parents=True)：連同中間缺的目錄一起建（類似 mkdir -p）
    # exist_ok=True：目錄已存在不抱怨（不加的話會 raise FileExistsError）
    path.parent.mkdir(parents=True, exist_ok=True)
    # json.dumps 把 Python dict 序列化成 JSON 字串：
    # - indent=2：每層縮排 2 個空格，讓人讀得舒服（不加就是一坨壓縮 JSON）
    # - ensure_ascii=False：中文 / 非 ASCII 字元直接輸出，不轉成 中 這種 escape
    # path.write_text：把字串寫進檔案，encoding="utf-8" 確保跨平台一致
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(path: Path) -> BM25Index | None:
    """從 JSON 載回 BM25Index。檔不存在 / 解析失敗 / 欄位缺都回 None、不 raise。"""
    # path.exists()：Path 的方法，等同 os.path.exists()；檔不存在就提早回 None
    if not path.exists():
        return None
    try:
        # json.loads：把 JSON 字串反序列化成 Python dict / list
        # path.read_text：把整個檔案讀成字串
        data = json.loads(path.read_text(encoding="utf-8"))
        # Section(**raw)：**kwargs 展開語法
        # 假設 raw = {"filename": "a.md", "heading": "Alpha", ...}
        # 則 Section(**raw) 等同 Section(filename="a.md", heading="Alpha", ...)
        # 這樣可以直接把 JSON dict 餵給 dataclass，不用逐欄位手寫
        sections = [Section(**raw) for raw in data["sections"]]
        tokens = data["tokens"]
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        # json.JSONDecodeError：字串不是合法 JSON（例如 "{ this is not json"）
        # KeyError：dict 裡找不到 "sections" 或 "tokens" 這個 key
        # TypeError：Section(**raw) 時欄位名不符（多欄位或少欄位都會 raise）
        # OSError：path 是目錄、檔案被 lock、權限不足等 I/O 問題（Windows 常見）
        return None

    # tokens 結構驗證：必須是 list[list[str]]。
    # 若被人手改 JSON 改成扁平 list[str]，BM25Okapi 會把每個字串當 iterable
    # 拆成單字元 token，導致所有查詢 score=0、silent failure。
    if not isinstance(tokens, list) or (tokens and not isinstance(tokens[0], list)):
        return None

    return BM25Index(sections=sections, tokens=tokens)
