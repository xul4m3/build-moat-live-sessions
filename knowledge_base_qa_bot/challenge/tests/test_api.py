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

# LLMResponse：test 用它組 mock_ask.return_value，模擬 LLM 結構化輸出的回傳型別
from app.llm import LLMResponse


@pytest.fixture
def app_factory(monkeypatch, sample_docs_dir, tmp_path):
    """工廠 fixture：建一個 FastAPI app instance，並 mock 掉外部依賴。

    回傳一個 callable，呼叫它拿到 (TestClient, mock_llm_ask)。

    為什麼用「工廠」而不是直接 fixture 回 TestClient？
    - 有的 test 需要 initial_index=True（先建好 index 再測 /chat）
    - 有的 test 需要在 fixture 外再設 env var（例如 threshold 999）
    - 工廠讓每個 test 按需組態，彈性更高

    Teardown：fixture 結尾 yield 之後會把每個被建立的 client 顯式 __exit__()，
    確保 lifespan 的 shutdown hook 被呼到、anyio portal 不會 leak。
    """
    created_clients: list[TestClient] = []

    def _build(
        initial_index: bool = False, docs_dir: Path | None = None, pre_seed_index: str | None = None
    ):
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
        #
        # docs_dir：預設用 conftest 的 sample_docs_dir；test 可覆寫成空目錄 / 不存在目錄。
        # pre_seed_index：若給字串，在建 client（觸發 lifespan）前先寫進 KB_INDEX_PATH，
        #                 用來模擬「啟動時已有既有 / 壞掉的 index.json」。
        index_path = tmp_path / ".kb" / "index.json"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        chosen_docs = docs_dir if docs_dir is not None else sample_docs_dir
        monkeypatch.setenv("KB_DOCS_DIR", str(chosen_docs))
        monkeypatch.setenv("KB_INDEX_PATH", str(index_path))

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

        # 3b. 若 test 要求，在 lifespan 觸發前預先寫入 index 檔
        # （reload 不會碰檔案、lifespan 在下面的 __enter__() 才讀，所以時機剛好）
        if pre_seed_index is not None:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(pre_seed_index, encoding="utf-8")

        # 4. 建立 TestClient 並觸發 lifespan（startup hook）
        # FastAPI 的 TestClient 只有在 context manager 模式下（with TestClient(app)）
        # 才會觸發 lifespan。這裡手動呼 __enter__() 讓 startup hook 執行，
        # 這樣 lifespan 裡的「載入既有 index」邏輯才會生效。
        # 對應的 __exit__() 由 fixture teardown 段（yield 之後）負責呼，
        # 不能依賴 GC —— ExitStack 沒有 __del__、靠 GC 不會關掉 lifespan。
        client = TestClient(main.app)
        client.__enter__()
        created_clients.append(client)
        if initial_index:
            client.post("/index")
        return client, mock_ask

    yield _build

    # Teardown：每個被建出的 client 都顯式關掉，跑 lifespan shutdown 段。
    for c in created_clients:
        c.__exit__(None, None, None)


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
    assert body["files_indexed"] == 2  # conftest 寫 2 個檔
    assert body["sections_indexed"] == 2  # 各 1 個 ## section

    # index 檔被寫出來（用 tmp_path 確認路徑跟 app_factory 設的 KB_INDEX_PATH 一致）
    index_path = tmp_path / ".kb" / "index.json"
    assert index_path.exists()


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
    assert "not been indexed" in body["answer"].lower() or "not indexed" in body["answer"].lower()
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


def test_startup_loads_existing_index(app_factory, sample_docs_dir, tmp_path, monkeypatch):
    """建 index、銷掉 app、重新起一個 app -> 不需要再呼 /index。

    這驗證 lifespan startup hook 真的有把 .kb/index.json 載進來：
    - client1 呼 /index，把 index 寫到 tmp_path/.kb/index.json
    - client2 reload(main) + 新建 TestClient，lifespan 執行時讀到 index.json
    - client2 不呼 /index，直接問 /chat -> 應該也能正常回答（不是 "not indexed" 訊息）
    """
    # 第一次：建 index
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "-1.0")
    client1, _ = app_factory(initial_index=True)
    assert (tmp_path / ".kb" / "index.json").exists()

    # 第二次：用同樣的 KB_INDEX_PATH 起一個新 client，但不呼 /index
    # app_factory 會 reload(main)，lifespan 重跑，startup hook 讀到既有 index
    client2, mock_ask = app_factory()

    # mock LLM 防止真的打 API
    mock_ask.return_value = LLMResponse(answer="x", sources=["refunds.md#refund-timeline"])

    r = client2.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 200
    # 重點：沒呼 /index 還是能正常回應 -> 證明 startup 載入成功
    assert "not been indexed" not in r.json()["answer"].lower()


def test_chat_llm_failure_returns_503(app_factory, monkeypatch):
    """LLM 呼叫拋例外（網路 / 認證 / rate limit）-> 503，不要漏 traceback。

    對應 DESIGN.md §6.2「OpenAI API 失敗（網路、429、500）→ 503 + 簡短錯誤訊息」。
    我們模擬 openai.APIConnectionError；caller 應該拿到 503 而非 500 或 trace。
    """
    import openai

    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "-1.0")
    client, mock_ask = app_factory(initial_index=True)

    # APIConnectionError 需要 request 物件；在 mock 場景傳 None 即可（不會被使用）
    mock_ask.side_effect = openai.APIConnectionError(request=None)  # type: ignore[arg-type]

    r = client.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 503
    body = r.json()
    # FastAPI HTTPException 的 detail 會出現在 response body
    assert "unavailable" in body["detail"].lower()


def test_chat_llm_refusal_returns_503(app_factory, monkeypatch):
    """LLMRefusalError（app 自訂例外，跟 openai.OpenAIError 不同類）-> 也要回 503。

    main.py 的 except tuple 同時抓 (openai.OpenAIError, LLMRefusalError)，
    但之前只測過 openai 那條。這個 test 守住 refusal 分支不會退化成 500。
    """
    from app.llm import LLMRefusalError

    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "-1.0")
    client, mock_ask = app_factory(initial_index=True)
    mock_ask.side_effect = LLMRefusalError("model refused")

    r = client.post("/chat", json={"query": "How long do refunds take?"})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"].lower()


def test_index_empty_docs_dir_returns_zero_counts(app_factory, tmp_path):
    """/index 對空目錄 -> 200 + files/sections = 0（BM25Index.build([]) 不該 crash）。"""
    empty = tmp_path / "empty_docs"
    empty.mkdir()
    client, _ = app_factory(docs_dir=empty)

    r = client.post("/index")
    assert r.status_code == 200
    assert r.json() == {"files_indexed": 0, "sections_indexed": 0}


def test_index_nonexistent_docs_dir_returns_zero_counts(app_factory, tmp_path):
    """/index 對不存在的目錄 -> load_docs 回 []，200 + count 0（靜默、不報錯）。"""
    missing = tmp_path / "nope"  # 不 mkdir
    client, _ = app_factory(docs_dir=missing)

    r = client.post("/index")
    assert r.status_code == 200
    assert r.json()["files_indexed"] == 0


def test_startup_with_corrupt_index_falls_back_to_not_indexed(app_factory):
    """啟動時 index.json 是壞掉的 JSON -> store.load 回 None、不 crash、
    /chat 回「尚未 index」訊息（recovery path）。"""
    client, mock_ask = app_factory(pre_seed_index="{ this is not valid json")

    r = client.post("/chat", json={"query": "anything"})
    assert r.status_code == 200
    answer = r.json()["answer"].lower()
    assert "not been indexed" in answer or "not indexed" in answer
    mock_ask.assert_not_called()
