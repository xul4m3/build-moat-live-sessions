# === Stage 1: builder ===
# 用 slim 版減少 image 體積；builder 只在 build 階段跑、最後不會留在 runtime
# 版本對齊 .python-version（3.13）—— 避免本機 / CI / Docker 跑不同 Python 版本
FROM python:3.13-slim AS builder

WORKDIR /build

# 先裝依賴。先 copy requirements.txt 才 copy code 是 Docker layer cache 慣例：
# 改 code 不會讓 pip install 整個 layer 重跑、build 加速
# Builder 階段不需要 dev deps（pytest/httpx）—— 只跑 build_index 這支 CLI
COPY challenge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + KB 都帶進來（build context 是 knowledge_base_qa_bot/，所以兩個路徑都看得到）
COPY challenge/app/ ./app/
COPY docs/ ./docs/

# Build 階段就把 index 算好、bake 進 image，runtime 啟動就 ready
RUN python -m app.build_index \
    --docs-dir docs/ \
    --output .kb/index.json

# === Stage 2: runtime ===
FROM python:3.13-slim

WORKDIR /app

# 只裝 runtime 依賴（pytest/httpx 放在 requirements-dev.txt、不進 image）
COPY challenge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 從 builder 拿三樣產物（COPY --from=builder 是 multi-stage 慣用語）
COPY --from=builder /build/app/  ./app/
COPY --from=builder /build/docs/ ./docs/
COPY --from=builder /build/.kb/  ./.kb/

# 切到非 root user。container 對外提供 HTTP，最小化 attack surface 是基本實務。
# --system 建系統帳號（無密碼、不能 login）；--no-create-home 不需要 home dir
RUN adduser --system --no-create-home --group appuser
USER appuser

# Runtime 預設參數；K8s manifest 可覆寫
ENV KB_DOCS_DIR=/app/docs \
    KB_INDEX_PATH=/app/.kb/index.json \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
