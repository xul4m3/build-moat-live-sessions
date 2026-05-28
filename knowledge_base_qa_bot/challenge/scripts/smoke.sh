#!/usr/bin/env bash
# PROMPT.md L60-105 の 7 個 curl 驗證。
# 用法: export OPENAI_API_KEY=sk-... && uvicorn app.main:app --reload && bash scripts/smoke.sh
# 注意: 這個腳本會真實呼叫 OpenAI API，需要有效的 API key 並會產生計費。
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"

echo "=== 1. /health ==="
curl -sS "$BASE/health"
echo
echo

echo "=== 2. /chat before /index (應該回 not indexed) ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "How long do refunds take?"}'
echo
echo

echo "=== 3. /index ==="
curl -sS -X POST "$BASE/index"
echo
echo

echo "=== 4. cat .kb/index.json (前 30 行) ==="
if [ -f .kb/index.json ]; then
    head -n 30 .kb/index.json
else
    echo "warning: .kb/index.json not found (是否從錯誤的 cwd 啟動 uvicorn?)"
fi
echo
echo

echo "=== 5. /chat grounded: refunds ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "How long do refunds take?"}'
echo
echo

echo "=== 6. /chat grounded: email ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "Can I change my email address?"}'
echo
echo

echo "=== 7. /chat out-of-scope: restaurants ==="
curl -sS -X POST "$BASE/chat" \
    -H "Content-Type: application/json" \
    -d '{"query": "Which restaurants are nearby?"}'
echo

echo
echo "=== Smoke test done. Review output above against PROMPT.md expectations. ==="
