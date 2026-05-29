# AGENTS.md — KB Q&A Bot (Challenge Track)

> Agent 進入點。人類請看 `README.md`；這份是給 coding agent 的速查（指路 + 硬約束 + 驗證命令）。

## 這是什麼

Markdown 知識庫的 grounded Q&A bot：FastAPI + BM25 檢索 + OpenAI 結構化輸出。
讀 `../docs/*.md` 建索引，`/chat` 只根據檢索到的 section 回答並附 `filename#heading` 引用。

## 文件導覽

| 檔案 | 看它做什麼 |
|------|-----------|
| `DESIGN.md` | 架構、設計決策（Q1-Q8）、部署、audit。**改行為前先讀** |
| `PLAN.md` | 12 個 task 的 TDD step-by-step |
| `DONE.md` | DoD checklist + edge-case hardening 記錄 |
| `app/` | 實作（loader → bm25 → store → retrieval → prompt → llm → main） |
| `tests/` | pytest（unit + integration，全程 mock OpenAI） |

## 硬約束（不可違反）

1. **Grounded only** — 答案只能來自檢索到的 section，不可臆測。
2. **必附 citation** — `filename#heading` 格式，由 OpenAI structured output (`LLMResponse`) 強制。
3. **弱檢索誠實拒答** — top-1 BM25 score < threshold → 回 cannot-confirm、**不呼 LLM**（省成本、避免幻覺）。
4. **No secret in git** — `.env` 已 gitignore；絕不 commit API key。
5. **index.json 必須人可讀** — plain JSON（Karpathy 風格），不可改成 pickle 等黑箱格式。

## 技術棧

- Python **3.13**（見 `.python-version`；Dockerfile 同步對齊）
- runtime 依賴：`requirements.txt`／dev+test：`requirements-dev.txt`
- 工具設定集中在 `pyproject.toml`（ruff + mypy）

## 驗證命令

```bash
make check          # 完整驗證：ruff lint + format-check + mypy + pytest
```

沒有 make（純 Windows）時逐項跑：

```bash
python -m ruff check app/ tests/          # lint
python -m ruff format --check app/ tests/ # 格式
python -m mypy app/                        # 型別（只查 app/）
python -m pytest tests/                    # 測試（不打外部 API）
bash scripts/smoke.sh                      # 真實打 OpenAI（需 OPENAI_API_KEY，會花錢）
```

提交前 `make check` 要全綠。
