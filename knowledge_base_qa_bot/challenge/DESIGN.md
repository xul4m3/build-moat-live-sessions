# Knowledge Base Q&A Bot — Design Doc

**Date**: 2026-05-28
**Branch**: `feat/kb-qa-bot`
**Track**: Challenge Track（從零實作，不用 scaffold）
**Source spec**: `../PROMPT.md`

## 1. Overview

對 `../docs/*.md`（3 個檔案、9 個 sections 的小型電商客服 FAQ）建一個 Q&A bot：

- 走 **Markdown KB + BM25** 策略（Karpathy 風格、不用 embeddings）
- 提供 `/health`、`/index`、`/chat` 三個 HTTP endpoint
- 答案 grounded、強制引用 `filename#heading`、找不到就誠實說 cannot-confirm

## 2. 決策摘要（Q&A 答案）

| # | 題目 | 決定 |
|---|------|------|
| Q1 | 檢索策略 | Markdown KB + BM25 |
| Q2 | retrieval unit | Section（以 `##` heading 切） |
| Q3 | 進 prompt 的策略 | Top-K=3 + score threshold |
| Q4 | citation 機制 | OpenAI structured output（JSON schema 強制） |
| Q5 | 弱檢索 fallback | 直接回 cannot-confirm、不呼 LLM |
| Q6 | 何時切到 Vector RAG | 語意查詢明顯多於關鍵字查詢 |
| Q7 | 何時切回 Markdown KB | 需要 inspectable / debuggable 的 index |
| Q8 | 100k files 時最先動 | Index 持久化換成獨立服務（Elasticsearch / Tantivy / SQLite FTS5）+ 增量更新 |

額外決策：

| 議題 | 決定 |
|------|------|
| 設定管理 | `.env` + `python-dotenv`（本機）/ env var injection（K8s） |
| docs/ 部署 | COPY 進 Docker image，跟 code 一起走 PR review |
| index 部署 | Docker build 階段預建、COPY 進 final image（不用 emptyDir） |
| Audit 來源 | git log + image tag —— 不另建 audit log table |

## 3. 架構

```
┌─────────────────────────────────────────────────┐
│ FastAPI App (uvicorn)                           │
│                                                 │
│   GET  /health   → 200 {"status": "ok"}         │
│   POST /index    → 讀 docs/*.md、建 BM25、寫    │
│                    .kb/index.json               │
│   POST /chat     → query → BM25 → top-3         │
│                    (+threshold) → OpenAI        │
│                    structured output            │
│                                                 │
│   啟動時：自動載入 .kb/index.json（若存在）     │
└─────────────────────────────────────────────────┘
        │                            │
        ▼                            ▼
  docs/*.md (source)        .kb/index.json (persisted index)
        │                            │
        └──── POST /index ───────────┘
                  rebuilds
```

**核心特性**：

- BM25 index 全部 in-memory，啟動時從 `.kb/index.json` 載入
- 沒有 background worker、沒有 DB —— 一個 process、一份 JSON
- LLM 只在 retrieval 過 threshold 時呼叫
- citation 由 OpenAI structured output 強制保證格式

**規模假設**：MVP 針對 < 100 個檔案、< 1000 sections。Q8 的 100k 級才換獨立服務。

## 4. 元件

依責任切成 7 個獨立模組：

| 模組 | 檔案 | 責任 | 輸入 → 輸出 |
|------|------|------|-------------|
| loader | `app/loader.py` | 讀 `docs/*.md`、按 `##` heading 切 section | 路徑 → `list[Section]` |
| bm25 | `app/bm25.py` | 建立 inverted index、計算 BM25 分數 | `list[Section]` → `BM25Index` 物件 |
| store | `app/store.py` | index 序列化 / 反序列化到 `.kb/index.json` | `BM25Index` ↔ JSON 檔 |
| retrieval | `app/retrieval.py` | query → top-K + threshold 判斷 | `query: str` → `RetrievalResult` |
| prompt | `app/prompt.py` | 把 retrieved sections 組成 LLM prompt | `list[Section] + query` → `str` |
| llm | `app/llm.py` | 呼叫 OpenAI、強制 structured output | `prompt: str` → `{answer, sources}` |
| main | `app/main.py` | FastAPI 路由、組裝以上模組 | HTTP → HTTP |
| build_index | `app/build_index.py` | CLI script，給 Docker build 階段用 | docs 路徑 → index 檔 |

**型別定義**（`app/types.py`）：

```python
from dataclasses import dataclass

@dataclass
class Section:
    """從 Markdown 切出來的一個段落，是檢索的最小單位。"""
    filename: str       # 例: "refund_policy.md"
    heading: str        # 例: "Refund Timeline"
    heading_slug: str   # 例: "refund-timeline"（用於 citation 的 #heading 部分）
    body: str           # 段落內文（不含 heading 那一行）

    @property
    def citation(self) -> str:
        """產出 PROMPT.md 要求的 citation 格式 'filename#heading-slug'。"""
        return f"{self.filename}#{self.heading_slug}"
```

**依賴方向**：`main → loader, store, retrieval, prompt, llm`；`retrieval → bm25`；所有人 → `types.py`。單向、沒有循環。

## 5. Data Flow

### 5.1 `POST /index` — 建索引

```
client ──POST /index──► main.py
                         │
                         ▼
                    loader.load_docs("docs/")     # 讀所有 *.md、按 "## " 切 section
                         │
                         ▼
                    list[Section]
                         │
                         ▼
                    bm25.build_index(sections)    # tokenize + 算 IDF
                         │
                         ▼
                    BM25Index 物件（in-memory）
                         │
                         ▼
                    store.save(index, ".kb/index.json")
                         │
                         ▼
                    回 200 {"files_indexed": 3, "sections_indexed": 9}
```

### 5.2 `POST /chat` — 問答

```
client ──POST /chat──► main.py
                        │
                        ▼
                   index 是否載入？
                   ┌─── no ──► 回 200 {"answer": "...not been indexed yet...", "sources": []}
                   │
                   yes
                   │
                   ▼
                   retrieval.search(query, k=3, threshold=T)
                   │
        ┌──────────┴──────────┐
        │                      │
  top score < T          top score >= T
        │                      │
        ▼                      ▼
  回 cannot-confirm      prompt.build(query, top3)
  （不呼 LLM）                   │
                                ▼
                          llm.ask(prompt)
                                │  response_format: JSON schema
                                │    {"answer": str, "sources": list[str]}
                                ▼
                          回 200 {answer, sources}
```

### 5.3 啟動流程

```
uvicorn 啟動 ──► FastAPI lifespan startup hook
                  │
                  ▼
             .kb/index.json 存在？
             ┌──── yes ────► store.load() → 存進 app.state.index
             │
             no
             │
             ▼
       不重建、不報錯。app.state.index = None
       後續 /chat 會回「請先 POST /index」
```

Production image 因為 build 時就 bake 好 index，永遠走 yes 分支。Local dev 可能走 no 分支來驗 PROMPT.md 的 "before indexing" 行為。

## 6. Error Handling

### 6.1 BM25 score threshold 決策

BM25 分數無界、會被文檔長度和語料規模影響。做法：用 PROMPT.md 的 3 個 curl test 當基準調參：

| Query | 預期 top-1 score | 用途 |
|-------|------------------|------|
| "How long do refunds take?" | 高 | grounded baseline |
| "Can I change my email address?" | 高 | grounded baseline |
| "Which restaurants are nearby?" | 低（"restaurants" 不在 vocabulary） | out-of-scope baseline |

實作時跑這 3 個 query、把分數印出來，threshold 設在 grounded 最低分跟 out-of-scope 之間。寫進 `app/config.py`（可被 env var `BM25_SCORE_THRESHOLD` 覆寫）。

### 6.2 異常路徑對照表

| 情境 | 處理 | HTTP |
|------|------|------|
| `OPENAI_API_KEY` 沒設 | App 啟動時直接 raise、不讓 server 起來 | — |
| `/chat` 在 `/index` 之前 | 回友善訊息「knowledge base not indexed yet」 | 200 |
| `/chat` query 是空字串 | FastAPI 自動 422 | 422 |
| `docs/` 不存在或空 | `/index` 回 200 但 `files_indexed: 0`、warn log | 200 |
| 某個 `.md` 沒有 `##` heading | 整檔當一個 section、heading 用檔名（去 `.md`） | — |
| OpenAI API 失敗（網路、429、500） | 503 + 簡短錯誤訊息，不漏 traceback | 503 |
| OpenAI 回的 JSON 不符 schema | structured output 模式不會發生；萬一發生則 500 | 500 |
| `.kb/index.json` 損壞 | log error、視為「沒有 index」、不要 crash | — |
| Threshold 過濾掉全部 | 回 `{"answer": "...cannot confirm...", "sources": []}` | 200 |

### 6.3 Logging

- Python `logging` module（標準庫）
- INFO：每次 `/index`（檔案數、section 數、耗時）、每次 `/chat`（query、top-3 sections、top score、是否走 fallback）
- ERROR：OpenAI API 失敗、index 載入失敗
- **絕不**印 API key 或完整 prompt（含 key 的 header）

## 7. Testing 策略

### 7.1 Unit tests（pytest）

| 測試檔 | 重點 case |
|--------|-----------|
| `test_loader.py` | 多 heading 切分、無 heading 退化、heading slug、空檔不 crash |
| `test_bm25.py` | tokenize（lowercase + 拆字 + 去 stopwords）、詞頻排序、未知詞 score=0 |
| `test_store.py` | save / load round-trip、壞 JSON 回 None |
| `test_retrieval.py` | top-K 排序、全部低於 threshold → cannot-confirm、k > 實際數量不報錯 |
| `test_prompt.py` | prompt 含每個 retrieved section 的 `[filename#heading]` tag、含 query、含 grounding instruction |

LLM 那一層不寫 unit test（外部 API 不穩定且花錢）—— 在 integration test 用 mock。

### 7.2 Integration test（`tests/test_api.py`）

用 FastAPI `TestClient`，把 OpenAI 那一層 mock 掉：

1. `/health` → 200
2. `/chat` 在 `/index` 之前 → 友善訊息
3. `/index` → 回 `{files_indexed, sections_indexed}`
4. `/chat` grounded query → 命中、含正確 citation
5. `/chat` out-of-scope → cannot-confirm、且 LLM 沒被呼叫
6. Index 持久化 round-trip

### 7.3 Manual smoke test

`scripts/smoke.sh` 包 PROMPT.md L60-105 的 7 個 curl，**真實打 OpenAI**。
unit + integration 全綠後最後跑這一關，驗證 prompt 工程在真實 LLM 上的效果。

## 8. 檔案結構

```
knowledge_base_qa_bot/
├── docs/                          # 既有 — sample KB（不動）
│   ├── refund_policy.md
│   ├── account_help.md
│   └── shipping_faq.md
│
├── challenge/                     # ★ Challenge Track 實作位置
│   ├── DESIGN.md                  # 本文件
│   ├── .env.example               # OPENAI_API_KEY=sk-... 等 template
│   ├── .env                       # 本機真實 key（被 .gitignore）
│   ├── .gitignore                 # 排除 .env、.kb/、__pycache__/、.venv/
│   ├── requirements.txt           # fastapi, uvicorn, rank-bm25, openai, python-dotenv, pytest
│   ├── README.md                  # 怎麼跑、怎麼測
│   ├── Dockerfile                 # multi-stage：builder 跑 indexing、final 跑 server
│   ├── .dockerignore              # 排除 .env、tests、.kb 等
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── types.py               # Section dataclass
│   │   ├── config.py              # 讀 .env + env、暴露所有設定值
│   │   ├── loader.py              # Markdown → Section list
│   │   ├── bm25.py                # BM25Index 封裝
│   │   ├── store.py               # save/load .kb/index.json
│   │   ├── retrieval.py           # search(query) → top-K + threshold
│   │   ├── prompt.py              # build LLM prompt
│   │   ├── llm.py                 # OpenAI client + structured output
│   │   ├── main.py                # FastAPI app + routes
│   │   └── build_index.py         # CLI script，Docker build stage 用
│   │
│   ├── tests/
│   │   ├── conftest.py            # fixtures（含 mock_openai）
│   │   ├── test_loader.py
│   │   ├── test_bm25.py
│   │   ├── test_store.py
│   │   ├── test_retrieval.py
│   │   ├── test_prompt.py
│   │   └── test_api.py            # integration
│   │
│   ├── scripts/
│   │   └── smoke.sh               # PROMPT.md 的 7 個 curl
│   │
│   └── .kb/                       # /index 之後產生（.gitignore）
│       └── index.json
│
├── scaffold/                      # 既有 — Guided Track（不動）
├── PROMPT.md                      # 既有
└── README.md                      # 既有
```

## 9. 開發順序

由內到外、TDD-friendly：

| # | 做什麼 | 為什麼這個順序 |
|---|--------|---------------|
| 0 | `requirements.txt`、`.env.example`、`.gitignore`、venv 建好 | 環境先穩 |
| 1 | `types.py`（Section dataclass） | 所有人會用到的型別、最小 |
| 2 | `loader.py` + test | 純函式、最好測 |
| 3 | `bm25.py` + test | 純運算、可重現 |
| 4 | `store.py` + test | I/O 但簡單、round-trip test 易寫 |
| 5 | `retrieval.py` + test | 組合 bm25 + threshold |
| 6 | `prompt.py` + test | 字串組裝 |
| 7 | `llm.py` + 手動 smoke | 第一次打真實 API、確認 structured output 能 work |
| 8 | `main.py` + `test_api.py` | 用 TestClient 串起來 |
| 9 | `config.py` + `.env` 整合 | 在 main 串好後做 |
| 10 | `build_index.py` + Dockerfile | 容器化 |
| 11 | `scripts/smoke.sh` 全綠 | 最終驗收 |

## 10. 部署與多環境

### 10.1 設定來源

`app/config.py` 集中讀取，先 `load_dotenv()`（本機才有 `.env`，K8s 無 `.env` 時 no-op），再 `os.environ`。App 不關心來源。

### 10.2 全部 env vars

| Env Var | 類別 | 用途 | 預設 |
|---------|------|------|------|
| `OPENAI_API_KEY` | **Secret** | OpenAI key | — |
| `OPENAI_MODEL` | Config | LLM 模型 | `gpt-4o-mini` |
| `BM25_SCORE_THRESHOLD` | Config | 過 fallback 的門檻；初值 `0.5`、實作時依 §6.1 跑基準調定 | `0.5` |
| `KB_DOCS_DIR` | Config | docs 目錄路徑 | `docs/` |
| `KB_INDEX_PATH` | Config | index 檔路徑 | `.kb/index.json` |
| `LOG_LEVEL` | Config | logging level | `INFO` |
| `ENV_NAME` | Config | local / qat / stg / prod，給 log 標記 | `local` |

### 10.3 K8s 對應

```yaml
spec:
  containers:
    - name: kb-qa-bot
      image: kb-qa-bot:1.0
      envFrom:
        - secretRef:    { name: kb-qa-bot-secrets }   # OPENAI_API_KEY
        - configMapRef: { name: kb-qa-bot-config }    # 其他全部
      # 不需要 volumeMounts —— docs 跟 index 都在 image 內
```

每個環境（QAT/STG/PROD）一份獨立的 Secret + ConfigMap，code 不動、image 不動，靠 manifest 切換。

### 10.4 Dockerfile（multi-stage）

**Build context** 設在 `knowledge_base_qa_bot/`（exercise 根目錄），這樣 `docs/` 跟 `challenge/` 都拷得到。指令：

```bash
# 從 knowledge_base_qa_bot/ 執行
docker build -f challenge/Dockerfile -t kb-qa-bot:1.0 .
```

```dockerfile
# Stage 1: builder
FROM python:3.12-slim AS builder
WORKDIR /build
COPY challenge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY challenge/app/ ./app/
COPY docs/ ./docs/                   # 從 build context (knowledge_base_qa_bot/) 直拷
RUN python -m app.build_index \
    --docs-dir docs/ \
    --output .kb/index.json

# Stage 2: runtime
FROM python:3.12-slim
WORKDIR /app
COPY challenge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /build/app/ ./app/
COPY --from=builder /build/docs/ ./docs/
COPY --from=builder /build/.kb/ ./.kb/
ENV KB_DOCS_DIR=/app/docs \
    KB_INDEX_PATH=/app/.kb/index.json
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`（放在 `knowledge_base_qa_bot/`）至少排除 `**/__pycache__`、`**/.env`、`**/.venv`、`**/tests`、`**/.kb`、`scaffold/`。

## 11. Audit 故事

- **`docs/` 變更**：走 git PR → review → merge → CI build → image tag → deploy。`git log docs/` 完整 trail：作者、時間、diff、commit msg、PR link、reviewer。
- **設定變更**：改 K8s ConfigMap / Secret 透過 GitOps repo PR，同樣有 git trail。
- **`/index` 呼叫**：不寫業務資料、只重建 index，audit 收益低。INFO log 留紀錄即可。
- **不開放 runtime mutation**：沒有 POST/PUT/DELETE docs 的 endpoint。要改 FAQ 一律走 git。

這個選擇直接套用 code 的工程紀律到知識庫上，audit 故事最強、最便宜 —— 也呼應 Karpathy 風格「把知識當 code」的精神。

## 12. Definition of Done

- [ ] PROMPT.md L60-105 的 7 個 curl test 全部通過
- [ ] `pytest tests/ -v` 全綠
- [ ] `.kb/index.json` 可以 `cat`、人能讀
- [ ] 重啟 server 後不需要重 index（startup hook 自動載入）
- [ ] Out-of-scope query 不呼 LLM、直接回 cannot-confirm
- [ ] 沒有 API key 或敏感資料進版控（`git log -p` 檢查）
- [ ] Dockerfile build 成功、image 起來能 serve `/health`

## 13. Out of Scope

以下為 PROMPT.md stretch goals 或 future work，**本次設計不涵蓋**：

- Streaming（`POST /chat/stream` + SSE）
- Browser UI
- Multi-format import（`.txt` / `.html` → `.md`）
- CLI / MCP 介面
- `wiki/index.md` 生成
- Answer filing 回寫 `wiki/`
- Conversation memory
- Paraphrase comparison（Markdown KB vs Vector RAG）
- Runtime mutation endpoint（POST docs、admin UI）
- 100k+ files 規模（換 Elasticsearch / Tantivy / SQLite FTS5）

完成 core 後可以逐項加。
