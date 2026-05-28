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
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("BM25_SCORE_THRESHOLD", "0.1")  # 寬鬆好驗 grounded path
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
