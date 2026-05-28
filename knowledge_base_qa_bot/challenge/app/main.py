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

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import load_config
from app.bm25 import BM25Index
from app.llm import LLMClient, LLMRefusalError
from app.loader import load_docs
from app.prompt import build_messages
from app.retrieval import search
from app.store import save, load


# ============ 啟動初始化 ============

logger = logging.getLogger(__name__)

_config = load_config()
logging.basicConfig(level=_config.log_level)

# state：用 dict 而不是 module-level variable，方便 test reload。
# reload(main) 會重建這個 dict，每個 test 拿到的都是全新的乾淨狀態，
# 避免 test 之間互相汙染（例如上一個 test 留下的 index 被下一個 test 用到）。
state: dict[str, Any] = {
    "config": _config,
    "index": None,           # BM25Index | None
    "llm": LLMClient(api_key=_config.openai_api_key, model=_config.openai_model),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 啟動 / 關閉 hook。Startup: 嘗試載入既有 index。

    asynccontextmanager 把一個 async generator function 包成
    「進入 (yield 之前) -> yield -> 離開 (yield 之後)」三段式 context manager：

        startup code    ← yield 之前這段，server 起來時跑
        yield           ← 這裡 FastAPI 開始接受請求
        shutdown code   ← yield 之後這段，server 關閉時跑

    FastAPI 在 app start 時跑 yield 前面那段、shutdown 時跑 yield 後面那段。
    沒有 yield 就會直接 raise，所以一定要有。
    """
    existing = load(state["config"].kb_index_path)
    if existing is not None:
        state["index"] = existing
        logger.info(
            f"loaded index from {state['config'].kb_index_path}: "
            f"{len(existing.sections)} sections"
        )
    else:
        logger.info(
            "no existing index; /chat will return 'not indexed' "
            "until /index is called"
        )
    yield
    # 沒有特別的 shutdown 工作


# @app.get 是 FastAPI 的「路由裝飾器」：
# 它告訴 FastAPI「當收到 GET /health 請求時，呼叫底下這個函式處理」。
# FastAPI 會自動把函式回傳值序列化成 JSON 回應。
app = FastAPI(lifespan=lifespan)


# ============ Endpoints ============

@app.get("/health")
def health() -> dict[str, str]:
    """簡單 liveness 檢查。

    這個 endpoint 讓 K8s liveness probe / load balancer 確認 server 還活著。
    不做任何 DB 或外部服務的健康檢查 —— 那是 readiness probe 的事。
    """
    return {"status": "ok"}


class IndexResponse(BaseModel):
    """POST /index 的回應格式。

    pydantic BaseModel 用於 FastAPI response_model：
    - FastAPI 自動產 OpenAPI schema（可在 /docs 看到）
    - 驗證輸出格式（如果 handler 不小心回錯型別，FastAPI 會 raise 而不是靜默傳錯）
    - 序列化成 JSON 回應
    """
    files_indexed: int
    sections_indexed: int


@app.post("/index", response_model=IndexResponse)
def build_index() -> IndexResponse:
    """讀 KB_DOCS_DIR 下所有 .md、建 BM25 index、寫到 KB_INDEX_PATH。

    這個 endpoint 冪等：重複呼叫會覆蓋舊 index，不會累積。
    Docker build 階段會呼一次此 endpoint（或直接呼 build_index.py），
    讓 image 內就有 .kb/index.json，container 起來不需要重跑。
    """
    cfg = state["config"]
    sections = load_docs(cfg.kb_docs_dir)
    index = BM25Index.build(sections)
    save(index, cfg.kb_index_path)
    state["index"] = index

    # set comprehension：{表達式 for 變數 in 集合} 自動去重
    # {s.filename for s in sections} 收集所有不同的 filename
    files_indexed = len({s.filename for s in sections})
    logger.info(
        f"indexed {files_indexed} files, {len(sections)} sections "
        f"→ {cfg.kb_index_path}"
    )
    return IndexResponse(
        files_indexed=files_indexed,
        sections_indexed=len(sections),
    )


# ============ /chat ============

class ChatRequest(BaseModel):
    """POST /chat 的 request body。

    FastAPI 看到函式參數型別是 BaseModel 子類，會自動從 HTTP body 反序列化 JSON。
    這比手動 `request.json()` 乾淨，也自動驗證格式（缺 query 就回 422）。
    """
    query: str


class ChatResponse(BaseModel):
    """POST /chat 的回應。"""
    answer: str
    sources: list[str]


# 把常數抽到 module level，方便 test assert、也不需要 hardcode 字串在 handler 裡
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

    1. 還沒 /index → 友善訊息、不呼 LLM
    2. 命中 retrieval（top-1 score >= threshold）→ 呼 LLM、回 answer + sources
    3. 沒命中 → cannot-confirm、不呼 LLM

    設計原則：只有在有可靠 context 的情況下才呼 LLM，
    避免「沒有根據的幻覺」且省下不必要的 API 費用。
    """
    cfg = state["config"]
    index = state["index"]

    # Path 1: 未建索引
    if index is None:
        logger.info(f"/chat called before /index: query={req.query!r}")
        return ChatResponse(answer=_NOT_INDEXED_MSG, sources=[])

    # 跑 retrieval：BM25 排名 + threshold 判斷
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
    # build_messages 把 sections + query 組成 OpenAI messages list
    messages = build_messages(req.query, result.sections)
    try:
        llm_response = state["llm"].ask(messages)
    except (openai.OpenAIError, LLMRefusalError) as exc:
        # DESIGN.md §6.2：LLM 失敗回 503，不要漏 traceback 給 client。
        # openai.OpenAIError 涵蓋 SDK 自家所有例外（網路、認證、rate limit、5xx）。
        logger.error(f"/chat LLM error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Please try again later.",
        )
    logger.info(
        f"/chat answered: query={req.query!r} "
        f"top_score={result.scores[0]:.3f} "
        f"sources={llm_response.sources}"
    )
    return ChatResponse(
        answer=llm_response.answer,
        sources=llm_response.sources,
    )
