"""Модель безопасности devassist.

Две основные гарантии:
  1. **Песочница по пути** — все файловые операции ограничены корнем проекта;
     попытки выйти за его пределы (../, абсолютные пути наружу, симлинки)
     отклоняются.
  2. **Классификация риска операции** — каждый вызов инструмента имеет уровень
     риска (SAFE / WRITE / DANGEROUS). Мутирующие и потенциально опасные
     операции требуют подтверждения пользователя (если не включён auto_approve).
"""

from __future__ import annotations

import re
from enum import IntEnum
from pathlib import Path


class RiskLevel(IntEnum):
    SAFE = 0       # чтение, поиск — без подтверждения
    WRITE = 1      # изменение ФС (запись/редактирование/коммит)
    DANGEROUS = 2  # потенциально разрушительные команды


class SandboxError(RuntimeError):
    """Попытка выйти за пределы песочницы проекта."""


def resolve_in_root(root: Path, path: str | Path) -> Path:
    """Приводит ``path`` к абсолютному и проверяет, что он внутри ``root``.

    Возвращает разрешённый Path. Бросает SandboxError, если путь ведёт наружу.
    """
    root = root.resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    # resolve() раскрывает .. и симлинки; strict=False — файла может ещё не быть.
    resolved = p.resolve()
    if resolved != root and root not in resolved.parents:
        raise SandboxError(
            f"Путь '{path}' выходит за пределы корня проекта ({root}). "
            "Операция запрещена."
        )
    return resolved


# Шаблоны заведомо опасных shell-конструкций.
_DANGEROUS_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*-?[rf]",   # rm -rf и вариации
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r":\(\)\s*\{",                       # fork-бомба
    r"\bchmod\s+-R\b",
    r"\bchown\s+-R\b",
    r"\bgit\s+push\b.*--force",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b",
    r"\bshutdown\b|\breboot\b",
    r"\bcurl\b|\bwget\b",                # сетевые загрузки запрещены политикой
    r"\bnc\b|\bnetcat\b",
]
_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)


def classify_shell_command(command: str) -> RiskLevel:
    """Оценивает риск shell-команды."""
    if _DANGEROUS_RE.search(command):
        return RiskLevel.DANGEROUS
    return RiskLevel.WRITE  # любая shell-команда может что-то менять
