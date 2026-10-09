"""Общие фикстуры для тестов devassist."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devassist.config import Config, load_environment  # noqa: E402
from devassist.tools.base import ToolContext  # noqa: E402


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Пустой временный «проект»."""
    return tmp_path


@pytest.fixture
def config(project: Path) -> Config:
    return Config(project_root=project, access_key="dummy")


@pytest.fixture
def ctx(config: Config) -> ToolContext:
    return ToolContext(config=config)


@pytest.fixture(scope="session")
def live_config() -> Config:
    """Конфиг для live-тестов (окружение + .env репозитория, OAuth или mTLS).

    Пропускает тест, если реквизиты GigaChat не заданы.
    """
    env = load_environment(REPO_ROOT)
    model = env.get("DEVASSIST_TEST_MODEL") or "GigaChat-2-Max"
    cfg = Config.load(project_root=REPO_ROOT, model=model)
    if cfg.auth_mode == "none":
        pytest.skip("Нет реквизитов GigaChat (ключ или cert+key) — live-тесты пропущены")
    return cfg
