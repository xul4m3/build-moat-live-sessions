"""驗證 config 從 env var 讀取、提供預設值、缺必要欄位時 raise。"""
from pathlib import Path

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
    # 直接 Path 比較跨平台、不用管 str() 在 Windows / POSIX 的差異
    assert cfg.kb_docs_dir == Path("/custom/docs")
    assert cfg.kb_index_path == Path("/custom/index.json")
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
    # 預設 path 也要驗，否則改錯 default 不會被 test 抓到
    assert cfg.kb_docs_dir == Path("../docs")
    assert cfg.kb_index_path == Path(".kb/index.json")


def test_load_config_invalid_threshold_raises_with_var_name(monkeypatch):
    """非法 BM25_SCORE_THRESHOLD -> 錯誤訊息要點名是哪個 var，方便 ops 排查。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("BM25_SCORE_THRESHOLD", "not-a-float")
    with pytest.raises(ValueError, match="BM25_SCORE_THRESHOLD"):
        load_config()


def test_load_config_missing_api_key_raises(monkeypatch):
    """OPENAI_API_KEY 沒設 -> 直接 raise、不要默默讓 server 起來。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        load_config()
