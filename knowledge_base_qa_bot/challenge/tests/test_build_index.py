"""驗證 build_index CLI：成功寫 index、空 corpus 回 1、缺參數 argparse 退出。

build_index 是 Docker build 階段跑的 CLI（bake index 進 image）。
壞了會讓 docker build 直接失敗、沒有 runtime fallback，所以值得有 test 守住。
"""
from pathlib import Path

import pytest

from app.build_index import main
from app.store import load


def test_main_success_writes_index_and_returns_zero(sample_docs_dir: Path, tmp_path: Path):
    """正常 docs 目錄 -> 寫出可載入的 index、回 exit code 0。"""
    out = tmp_path / "out" / "index.json"  # 故意用不存在的巢狀目錄，順便驗 save 會 mkdir
    rc = main(["--docs-dir", str(sample_docs_dir), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    # 寫出來的 index 要能被 store.load 讀回（不是壞檔）
    idx = load(out)
    assert idx is not None
    assert len(idx.sections) == 2  # conftest 的 sample_docs_dir 有 2 個 section


def test_main_empty_corpus_returns_one(tmp_path: Path, capsys):
    """空目錄（沒有 .md）-> 回 1 + stderr 警告、不寫出 index（讓 docker build 失敗）。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "index.json"
    rc = main(["--docs-dir", str(empty), "--output", str(out)])
    assert rc == 1
    assert not out.exists()  # 沒有 section 就不該寫檔
    # capsys 是 pytest 內建 fixture：抓 stdout / stderr
    err = capsys.readouterr().err
    assert "no .md files" in err.lower()


def test_main_nonexistent_docs_dir_returns_one(tmp_path: Path):
    """--docs-dir 指向不存在的目錄 -> load_docs 回 []，走 empty corpus path、回 1。"""
    missing = tmp_path / "does-not-exist"
    out = tmp_path / "index.json"
    rc = main(["--docs-dir", str(missing), "--output", str(out)])
    assert rc == 1


def test_main_missing_required_arg_exits_2(tmp_path: Path):
    """缺 required 參數（--docs-dir）-> argparse 呼 sys.exit(2)。

    argparse 解析失敗會 raise SystemExit；pytest.raises 接住、檢查 exit code。
    """
    out = tmp_path / "index.json"
    with pytest.raises(SystemExit) as exc_info:
        main(["--output", str(out)])  # 故意不給 --docs-dir
    assert exc_info.value.code == 2
