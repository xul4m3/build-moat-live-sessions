# KB Q&A Bot — 驗證命令統一入口。
#
# Windows 注意：make 需要 Git Bash / WSL / scoop install make。
# 沒有 make 時直接跑下面每個 target 內的命令即可（見 AGENTS.md）。
#
# 用 python -m 形式呼叫工具，確保用的是 venv 裡的版本、不是全域。

PY := python

.PHONY: help install lint format typecheck test check

help: ## 列出所有 target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## 裝 runtime + dev 依賴
	$(PY) -m pip install -r requirements-dev.txt

lint: ## ruff lint（不改檔）
	$(PY) -m ruff check app/ tests/

format: ## ruff format（改檔）
	$(PY) -m ruff format app/ tests/

typecheck: ## mypy 型別檢查（只查 app/）
	$(PY) -m mypy app/

test: ## pytest（unit + integration，不打外部 API）
	$(PY) -m pytest tests/ -x

check: ## 完整驗證：lint + format-check + typecheck + test
	$(PY) -m ruff check app/ tests/
	$(PY) -m ruff format --check app/ tests/
	$(PY) -m mypy app/
	$(PY) -m pytest tests/
