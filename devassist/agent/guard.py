"""Ограничители агентного цикла.

Решают, когда принудительно остановить ход: исчерпан лимит шагов или модель
застряла (серия неудачных вызовов инструментов). Детектор зацикливания на
повторяющихся вызовах встраивается в :meth:`LoopGuard.after_tool` — он видит
каждый вызов с аргументами и его исход.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from devassist.llm.types import FunctionCall

StopKind = Literal["max_steps", "tool_failures"]


@dataclass(frozen=True)
class StopReason:
    kind: StopKind
    message: str


class LoopGuard:
    """Состояние ограничителей на один ход агента."""

    def __init__(self, *, max_steps: int, max_failures: int):
        self._max_steps = max_steps
        self._max_failures = max_failures
        self._steps = 0
        self._failures = 0

    @property
    def steps(self) -> int:
        return self._steps

    def before_step(self) -> StopReason | None:
        """Вызывается перед обращением к модели; учитывает шаг."""
        if self._steps >= self._max_steps:
            return StopReason(
                "max_steps",
                f"Достигнут лимит шагов агента ({self._max_steps}). "
                "Задача может быть не завершена.",
            )
        self._steps += 1
        return None

    def after_tool(self, call: FunctionCall, ok: bool) -> StopReason | None:
        """Вызывается после каждого вызова инструмента."""
        self._failures = 0 if ok else self._failures + 1
        if self._failures >= self._max_failures:
            return StopReason(
                "tool_failures",
                f"Прервано: {self._failures} неудачных вызовов инструментов подряд. "
                "Похоже, агент застрял — уточните задачу или попробуйте другую модель.",
            )
        return None
