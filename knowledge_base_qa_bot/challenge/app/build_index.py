"""
Docker build 階段執行的 CLI：讀 docs/、建 BM25 index、寫到指定路徑。

跟 FastAPI 沒關係 —— 純粹「load_docs + BM25Index.build + save」的 shell。
Dockerfile 在 builder stage 跑這個、把產物 COPY 進 runtime image。

用法:
    python -m app.build_index --docs-dir docs/ --output .kb/index.json
"""
import argparse
import sys
from pathlib import Path

from app.bm25 import BM25Index
from app.loader import load_docs
from app.store import save


def main(argv: list[str] | None = None) -> int:
    """CLI entry。回傳 exit code（0 成功、非 0 失敗）。

    argparse 是 Python 標準庫的命令列參數解析工具：
    - add_argument 註冊一個參數，type=Path 自動轉成 Path 物件
    - required=True 強制使用者一定要傳
    - 解析失敗（缺參數、格式錯誤）argparse 自動印錯誤訊息、exit code 2

    argv=None 時 argparse 會從 sys.argv 拿；test 時可以傳 list[str] 直接呼。
    """
    parser = argparse.ArgumentParser(description="Build BM25 index from docs/")
    parser.add_argument("--docs-dir", type=Path, required=True,
                        help="Directory containing .md files to index")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for index.json")
    args = parser.parse_args(argv)

    sections = load_docs(args.docs_dir)
    if not sections:
        # 寫 stderr 而非 stdout：給 Docker build log 一個明顯的警告，
        # 同時 exit code 1 讓 Docker build 因此失敗（避免 ship 空 image）
        print(f"warning: no .md files found in {args.docs_dir}",
              file=sys.stderr)
        return 1

    index = BM25Index.build(sections)
    save(index, args.output)

    # set comprehension 去重後算檔案數
    files = len({s.filename for s in sections})
    print(f"indexed {files} files, {len(sections)} sections → {args.output}")
    return 0


# Python 慣用法：if __name__ == "__main__" 表示「直接執行此檔時才跑」，
# 被當作 module import 時不會自動跑（避免副作用）。
if __name__ == "__main__":
    sys.exit(main())
