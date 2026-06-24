"""Общие фикстуры для тестов devassist."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devassist.config import Config  # noqa: E402
from devassist.tools.base import ToolContext  # noqa: E402


def resolve_access_key() -> str | None:
    """Ключ GigaChat для live-тестов.

    Берём из окружения; если не задан — из эталонного скрипта test_gigachat.py,
    чтобы тесты были воспроизводимы одной командой без ручной настройки.
    """
    key = os.environ.get("GIGACHAT_ACCESS_KEY")
    if key:
        return key
    try:
        import test_gigachat  # type: ignore

        return getattr(test_gigachat, "ACCESS_KEY", None) or None
    except Exception:
        return None


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
    """Конфиг для live-тестов; пропускает тест, если ключ недоступен."""
    key = resolve_access_key()
    if not key:
        pytest.skip("Нет GIGACHAT_ACCESS_KEY — live-тесты пропущены")
    model = os.environ.get("DEVASSIST_TEST_MODEL", "GigaChat-2-Max")
    return Config(access_key=key, model=model, project_root=Path.cwd())
