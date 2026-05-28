"""FastAPI integration test。

用 fastapi.testclient.TestClient 模擬 HTTP 呼叫；OpenAI client 用 monkeypatch mock。

TestClient 是 FastAPI 提供的同步測試用戶端：
- 底層用 httpx，不需要真的起 server
- 每次用 `with TestClient(app)` 進入都會觸發 lifespan（startup/shutdown）
- 這樣 test 就能完整驗證 startup hook 的行為

monkeypatch 是 pytest 內建 fixture：
- 可以暫時改 env var（setenv/delenv）、替換函式（setattr）
- test 結束後自動復原，不會汙染其他 test
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_factory(monkeypatch, sample_docs_dir, tmp_path):
    """工廠 fixture：建一個 FastAPI app instance，並 mock 掉外部依賴。

    回傳一個 callable，呼叫它拿到 (TestClient, mock_llm_ask)。

    為什麼用「工廠」而不是直接 fixture 回 TestClient？
    - 有的 test 需要 initial_index=True（先建好 index 再測 /chat）
    - 有的 test 需要在 fixture 外再設 env var（例如 threshold 999）
    - 工廠讓每個 test 按需組態，彈性更高
    """
    def _build(initial_index: bool = False):
        # 1. 設好 env var，讓 load_config() 拿到測試用的值
        # BM25_SCORE_THRESHOLD 刻意不在這裡設，讓 test 在呼 _build() 前自己設定：
        # - 要驗 grounded path 的 test：先 setenv("BM25_SCORE_THRESHOLD", "-1.0")
        # - 要驗 fallback path 的 test：先 setenv("BM25_SCORE_THRESHOLD", "999.0")
        # 這樣 _build() 內不會覆蓋掉 test 外層的設定，符合「fixture 不干預 test 意圖」原則。
        #
        # 注意：rank_bm25 的 BM25Okapi 在極小語料（2 個 section）下，
        # 所有 term 的 IDF = 0，所以 get_scores() 一律回 0.0。
        # grounded path 的 test 用 threshold=-1.0 繞過這個邊際情形；
        # 生產環境的 docs/ 語料夠大，score 不會是 0，用預設 0.5 即可。
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("KB_DOCS_DIR", str(sample_docs_dir))
        monkeypatch.setenv("KB_INDEX_PATH", str(tmp_path / ".kb" / "index.json"))

        # 2. mock 掉 LLMClient.ask；要在 import main 之前 patch
        # monkeypatch.setattr("app.llm.LLMClient.ask", mock_ask) 的意思：
        # 把 app.llm 模組裡 LLMClient class 的 ask method 換成 MagicMock。
        # 因為 patch 的是 class 上的 method（不是 instance），所以所有 instance 都受影響。
        mock_ask = MagicMock()
        monkeypatch.setattr("app.llm.LLMClient.ask", mock_ask)

        # 3. import 並建 app
        # importlib.reload(main) 的必要性：
        # main.py 的 module-level 程式碼（例如 _config = load_config()）只在
        # 第一次 import 時執行一次。若 test 改了 env var 再 import，Python 會
        # 從快取（sys.modules）拿已有的 module，load_config() 不會重跑。
        # reload() 強制重新執行整個 module，讓 env 改動生效。
        from importlib import reload
        from app import main
        reload(main)  # 確保拿到最新 env 變數

        # 4. 如果 test 想要預先有 index，先呼一次 /index
        # TestClient 做為 context manager 使用時才會觸發 lifespan；
        # 這裡直接建構也可以（TestClient 預設在 __init__ 就啟動 lifespan）。
        client = TestClient(main.app)
        if initial_index:
            client.post("/index")
        return client, mock_ask

    return _build


def test_health_returns_ok(app_factory):
    client, _ = app_factory()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_endpoint_builds_and_persists(app_factory, tmp_path):
    """/index 應該讀 KB_DOCS_DIR 下的 .md、建 BM25 index、把 index.json 寫到磁碟。

    conftest.py 的 sample_docs_dir 寫了兩個 .md（refunds.md + account.md），
    各有一個 ## section，所以 files_indexed=2、sections_indexed=2。
    """
    client, _ = app_factory()

    r = client.post("/index")
    assert r.status_code == 200

    body = r.json()
    assert body["files_indexed"] == 2          # conftest 寫 2 個檔
    assert body["sections_indexed"] == 2       # 各 1 個 ## section

    # index 檔被寫出來（用 tmp_path 確認路徑跟 app_factory 設的 KB_INDEX_PATH 一致）
    index_path = tmp_path / ".kb" / "index.json"
    assert index_path.exists()


# LLMResponse 在此 import，test 可以 mock_ask.return_value = LLMResponse(...)
# 這樣不需要知道 main.py 內部用的是 dict 還是 pydantic，只要遵守 LLMResponse schema 就好
from app.llm import LLMResponse


def test_chat_before_index_returns_friendly_message(app_factory):
    """/index 之前呼 /chat -> 友善訊息、不要 500。

    這個 test 驗證：server 剛啟動、還沒 index 時，/chat 應該回 200 + 說明訊息，
    而不是噴 NullPointerException 或 500。
    """
    client, mock_ask = app_factory()  # 不呼 /index

    r = client.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 200
    body = r.json()
    # 訊息裡要說 "not been indexed" 或 "not indexed"（大小寫不限）
    assert "not been indexed" in body["answer"].lower() \
        or "not indexed" in body["answer"].lower()
    assert body["sources"] == []
    # LLM 不該被呼叫，因為根本沒有 index 可以檢索
    mock_ask.assert_not_called()


def test_chat_grounded_query_calls_llm_and_returns_citation(app_factory, monkeypatch):
    """正常 query -> 命中 retrieval -> 呼 LLM -> 回 answer + sources。

    這是 happy path：
    1. /index 先建好 index
    2. query 跟 index 內容有關 -> BM25 score 超過 threshold
    3. 呼 LLM，LLM 回 answer + citation
    4. /chat 把 LLM 回傳原樣給客戶端

    threshold 設 -1.0 因為測試語料只有 2 個 section，rank_bm25 的 Okapi 在
    這種極小語料下 IDF=0，所有 score 都是 0.0。
    生產環境語料夠大，score > 0；test 用 -1.0 模擬「任何 score 都算命中」。
    """
    # 先設 threshold，讓 reload(main) 時拿到這個值
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "-1.0")
    client, mock_ask = app_factory(initial_index=True)

    # 讓 mock_ask 回一個真實的 LLMResponse 物件
    mock_ask.return_value = LLMResponse(
        answer="Approved refunds are processed within 5-7 business days.",
        sources=["refunds.md#refund-timeline"],
    )

    r = client.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 200
    body = r.json()
    assert "5-7 business days" in body["answer"]
    assert "refunds.md#refund-timeline" in body["sources"]
    # LLM 被呼叫一次（grounded path）
    mock_ask.assert_called_once()


def test_chat_out_of_scope_query_returns_cannot_confirm_without_calling_llm(
    app_factory, monkeypatch
):
    """完全沒命中的 query -> fallback、不呼 LLM。

    threshold 設到 999，任何 BM25 score 都過不了，觸發 fallback 路徑。
    fallback 的設計原則：不確定的時候誠實說不知道，而不是胡說。
    """
    # 把 threshold 拉高，讓任何 query 都過不了
    # 注意：要在 app_factory() 之前 setenv，因為 factory 裡 reload(main) 時會重讀 env
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "999.0")
    client, mock_ask = app_factory(initial_index=True)

    r = client.post("/chat", json={"query": "Which restaurants are nearby?"})
    assert r.status_code == 200
    body = r.json()
    assert "cannot confirm" in body["answer"].lower()
    assert body["sources"] == []
    # fallback 路徑不該呼 LLM（省成本、避免幻覺）
    mock_ask.assert_not_called()
