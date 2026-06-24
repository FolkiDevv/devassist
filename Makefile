.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install dev test test-all lint format check clean

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Установить пакет (runtime) в editable-режиме
	$(PYTHON) -m pip install -e .

dev: ## Установить пакет + dev-зависимости (pytest, ruff)
	$(PYTHON) -m pip install -e . -r requirements-dev.txt

test: ## Быстрые офлайн-тесты (без сети)
	$(PYTHON) -m pytest -m "not live"

test-all: ## Все тесты, включая live (нужен GIGACHAT_ACCESS_KEY)
	$(PYTHON) -m pytest

lint: ## Проверить стиль и ошибки (ruff)
	$(PYTHON) -m ruff check .

format: ## Автоформатирование и автофиксы (ruff)
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

check: lint test ## Линт + офлайн-тесты (запускайте перед пушем)

clean: ## Удалить кеши и артефакты сборки
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
