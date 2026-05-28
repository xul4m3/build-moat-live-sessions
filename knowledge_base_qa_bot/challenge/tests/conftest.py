"""
pytest 的全域 fixture 集中地。conftest.py 是 pytest 的特殊檔名：
同層或上層目錄裡所有 test 都會自動拿到這裡定義的 fixture，不用 import。
"""
import pytest
from pathlib import Path


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """建一個暫存 docs/ 目錄、寫兩個樣本 .md 進去，回傳路徑。

    tmp_path 是 pytest 內建 fixture：每個 test 拿到一個獨立的暫存資料夾，
    test 結束自動清掉，不會污染 working tree。
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text(
        "# Refund Policy\n\n"
        "## Refund Timeline\n\n"
        "Approved refunds are processed within 5-7 business days.\n",
        encoding="utf-8",
    )
    (docs / "account.md").write_text(
        "# Account\n\n"
        "## Reset Password\n\n"
        "Password reset link expires after 30 minutes.\n",
        encoding="utf-8",
    )
    return docs
