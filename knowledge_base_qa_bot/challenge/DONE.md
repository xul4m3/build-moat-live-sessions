# Done Checklist — KB Q&A Bot Challenge Track

Status as of 2026-05-28 on branch `feat/kb-qa-bot`.

## Definition of Done (per DESIGN.md §12)

- [x] `pytest tests/ -v` 全綠 — 55 tests pass
  - types (3)
  - loader (11)
  - bm25 (9)
  - store (7)
  - retrieval (4)
  - prompt (6)
  - llm (4)
  - config (4)
  - api (7)

- [x] `.kb/index.json` 可以 `cat`、人能讀 — store.save 用 indent=2, ensure_ascii=False，JSON 結構簡單可掃描

- [x] 重啟 server 後不需要重 index — lifespan startup hook 自動載入既有 .kb/index.json（test_startup_loads_existing_index 驗證）

- [x] Out-of-scope query 不呼 LLM、直接回 cannot-confirm — test_chat_out_of_scope_query_returns_cannot_confirm_without_calling_llm 驗證 mock_ask.assert_not_called()

- [x] 沒有 API key 或敏感資料進版控 — `.env` 在 .gitignore；API key 絕不會被 git add 進去

- [x] Dockerfile build 成功、image 起來能 serve `/health` — Task 10 已驗：docker build -t kb-qa-bot:dev . 成功、docker run 起來回應 /health 200

- [ ] PROMPT.md L60-105 的 7 個 curl test 全部通過 — **needs manual run with real OPENAI_API_KEY** via `bash scripts/smoke.sh`

## Outstanding for user

1. Set `OPENAI_API_KEY=sk-...` in `.env` (or export to shell)
2. Start server from `challenge/` directory:
   ```bash
   uvicorn app.main:app --reload
   ```
3. In a new terminal, run smoke tests (requires Git Bash on Windows, or WSL):
   ```bash
   cd challenge/
   bash scripts/smoke.sh
   ```
4. Verify each of the 7 outputs matches PROMPT.md expectations:

   **Step 1: /health**
   ```
   → 200, {"status":"ok"}
   ```

   **Step 2: /chat before /index**
   ```
   → 200, answer contains "not been indexed yet" or similar
   ```

   **Step 3: /index**
   ```
   → 200, {"files_indexed":3,"sections_indexed":9}
   ```

   **Step 4: cat .kb/index.json**
   ```
   → visible JSON with "sections" and "tokens" keys
   → readable structure (no minification)
   → can spot individual doc filenames and headings
   ```

   **Step 5: /chat grounded refund query**
   ```
   → 200, answer discusses refunds
   → sources includes "refund_policy.md#refund-timeline"
   ```

   **Step 6: /chat grounded email query**
   ```
   → 200, answer discusses email
   → sources includes "account_help.md#change-email-address"
   ```

   **Step 7: /chat out-of-scope restaurants query**
   ```
   → 200, answer says cannot-confirm
   → sources is empty []
   ```

### Tuning If Needed

If a grounded query falls into the cannot-confirm path (Step 5 or 6), the BM25_SCORE_THRESHOLD may be too high. Try:
```bash
# In .env:
BM25_SCORE_THRESHOLD=0.1
# Restart server and re-run smoke.sh
```

The default 0.5 was tuned against the sample corpus; your threshold may differ.

## Test Count Summary

```
$ pytest tests/ -v
============================= 55 passed in ~3.35s =============================
```

Breakdown:
- test_api.py: 7 (including integration tests)
- test_bm25.py: 9 (including edge cases like stopword-only queries)
- test_config.py: 4 (env parsing, defaults, validation)
- test_llm.py: 4 (OpenAI mock, structured output)
- test_loader.py: 11 (slugify, parse, edge cases)
- test_prompt.py: 6 (citation tags, grounding, order)
- test_retrieval.py: 4 (threshold, fallback, empty index)
- test_store.py: 7 (round-trip, corruption, format)
- test_types.py: 3 (Section, RetrievalResult, citations)

## Code Quality Checklist

- [x] No bare `except` clauses — all exceptions caught explicitly
- [x] No hardcoded secrets — all config via env vars
- [x] No debug print statements left in code
- [x] Type hints on all function signatures
- [x] Docstrings on all modules and public functions
- [x] Unit tests before integration tests before manual smoke
- [x] Error messages are user-friendly, not raw stack traces
- [x] BM25 index is JSON (not pickle), human-readable

## Commit History

Branch: `feat/kb-qa-bot` (off main)

Expected commits on this branch:
- Task 0: Bootstrap
- Task 1-9: Core implementation
- Task 10: Dockerfile + build_index.py
- Task 11a: smoke.sh verification script
- Task 11b: DONE.md checklist (this file)

Run `git log feat/kb-qa-bot --oneline ^main` from repo root to see full history.

## Notes for Reviewers

1. **Smoke test is manual** — It requires a real OPENAI_API_KEY and will incur API costs. We did not auto-run it in CI.

2. **Index JSON is human-readable** — By design (Karpathy style). You can `cat .kb/index.json` and understand the structure.

3. **Threshold tuning** — BM25 scores are corpus-dependent. The default 0.5 was chosen empirically; adjust via BM25_SCORE_THRESHOLD env var if needed.

4. **No external storage** — Index lives in .kb/index.json, loaded into RAM on startup. For 100k+ files, would need to switch to Elasticsearch or similar (out of scope).

5. **Mock coverage** — All tests mock OpenAI; only smoke.sh touches the real API.
