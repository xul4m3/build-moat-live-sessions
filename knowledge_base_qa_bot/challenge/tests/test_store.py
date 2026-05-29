"""驗證 index 序列化：save → load 拿回等價物件。"""
from pathlib import Path

import pytest

from app.bm25 import BM25Index
from app.store import save, load
from app.types import Section


def _sample_index() -> BM25Index:
    sections = [
        Section("a.md", "Alpha", "alpha", "the alpha section body"),
        Section("b.md", "Beta", "beta", "the beta section body"),
    ]
    return BM25Index.build(sections)


def test_save_then_load_roundtrips(tmp_path: Path):
    """save 後 load 回來，sections 跟 tokens 都該一致。"""
    idx = _sample_index()
    path = tmp_path / ".kb" / "index.json"

    save(idx, path)

    assert path.exists(), "save 應該建好 .kb/ 目錄並寫檔"

    loaded = load(path)
    assert loaded is not None
    assert loaded.sections == idx.sections
    assert loaded.tokens == idx.tokens


def test_load_missing_file_returns_none(tmp_path: Path):
    """檔不存在 -> 回 None、不要 raise。"""
    assert load(tmp_path / "nope.json") is None


def test_load_corrupt_json_returns_none(tmp_path: Path):
    """JSON 解析失敗 -> 回 None、不要 crash。"""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert load(bad) is None


def test_load_missing_keys_returns_none(tmp_path: Path):
    """JSON 是合法但缺欄位 -> 回 None。"""
    bad = tmp_path / "incomplete.json"
    bad.write_text('{"sections": []}', encoding="utf-8")  # 缺 tokens
    assert load(bad) is None

    bad2 = tmp_path / "incomplete2.json"
    bad2.write_text('{"tokens": []}', encoding="utf-8")  # 缺 sections
    assert load(bad2) is None


def test_load_wrong_tokens_shape_returns_none(tmp_path: Path):
    """tokens 應為 list[list[str]]；扁平 list[str] -> 回 None。

    若不擋下這個 case，BM25Okapi 會把每個字串當 iterable 拆成單字元 token，
    所有查詢 score=0、silent failure。
    """
    bad = tmp_path / "wrong_tokens.json"
    bad.write_text(
        '{"sections": [], "tokens": ["alpha", "beta"]}',  # 應該是 [["alpha"], ["beta"]]
        encoding="utf-8",
    )
    assert load(bad) is None


def test_load_path_is_directory_returns_none(tmp_path: Path):
    """path 指向目錄而非檔案 -> read_text 會 raise OSError、應該被吃掉回 None。"""
    a_dir = tmp_path / "im_a_directory"
    a_dir.mkdir()
    assert load(a_dir) is None


def test_saved_json_is_human_readable(tmp_path: Path):
    """Karpathy 風格的核心：.kb/index.json 要 cat 起來看得懂。"""
    idx = _sample_index()
    path = tmp_path / "index.json"
    save(idx, path)

    raw = path.read_text(encoding="utf-8")
    assert "alpha" in raw                   # 看得到 section heading slug
    assert "the alpha section body" in raw  # 看得到原文
    assert "\n" in raw                      # 有 indent、不是一坨
