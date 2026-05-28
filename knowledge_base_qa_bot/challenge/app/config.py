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
    """所有 runtime 設定值。frozen=True 表示建構後不可改、避免被誤動。

    frozen=True 的 dataclass 等同於 readonly：嘗試 cfg.openai_model = "x" 會 raise FrozenInstanceError。
    這對 config 物件特別有用 —— 設定一次、整個 process 不該被偷偷改。
    """
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
    # 沒有 .env 檔不會 raise，只是 no-op。
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
        # float() 把字串轉浮點數；env var 永遠是字串，要自己轉
        bm25_score_threshold=float(os.environ.get("BM25_SCORE_THRESHOLD", "0.5")),
        # Path(...) 把字串轉成 pathlib.Path 物件，方便後續做 .exists()、.read_text() 等操作
        kb_docs_dir=Path(os.environ.get("KB_DOCS_DIR", "../docs")),
        kb_index_path=Path(os.environ.get("KB_INDEX_PATH", ".kb/index.json")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        env_name=os.environ.get("ENV_NAME", "local"),
    )
