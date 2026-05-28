"""驗證 config 從 env var 讀取、提供預設值、缺必要欄位時 raise。"""
import pytest

from app.config import Config, load_config


def test_load_config_uses_env_values(monkeypatch):
    """env var 有值 -> 用 env 的值。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "1.5")
    monkeypatch.setenv("KB_DOCS_DIR", "/custom/docs")
    monkeypatch.setenv("KB_INDEX_PATH", "/custom/index.json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENV_NAME", "qat")

    cfg = load_config()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_model == "gpt-4o"
    assert cfg.bm25_score_threshold == 1.5
    assert cfg.kb_docs_dir.as_posix() == "/custom/docs"
    assert cfg.kb_index_path.as_posix() == "/custom/index.json"
    assert cfg.log_level == "DEBUG"
    assert cfg.env_name == "qat"


def test_load_config_uses_defaults(monkeypatch):
    """除了 OPENAI_API_KEY 必填，其他都該有合理預設值。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # 確保其他 env 是空的
    for k in ("OPENAI_MODEL", "BM25_SCORE_THRESHOLD", "KB_DOCS_DIR",
              "KB_INDEX_PATH", "LOG_LEVEL", "ENV_NAME"):
        monkeypatch.delenv(k, raising=False)

    cfg = load_config()
    assert cfg.openai_model == "gpt-4o-mini"
    assert cfg.bm25_score_threshold == 0.5
    assert cfg.log_level == "INFO"
    assert cfg.env_name == "local"


def test_load_config_missing_api_key_raises(monkeypatch):
    """OPENAI_API_KEY 沒設 -> 直接 raise、不要默默讓 server 起來。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        load_config()
