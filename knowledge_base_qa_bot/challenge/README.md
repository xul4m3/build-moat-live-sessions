# KB Q&A Bot — Challenge Track

從零實作的 Markdown KB + BM25 Q&A bot。設計細節見 `DESIGN.md`、施工順序見 `PLAN.md`。

## Setup

```bash
cd challenge
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # 含 runtime + 測試依賴；production image 只裝 requirements.txt
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
