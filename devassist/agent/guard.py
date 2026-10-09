"""Ограничители агентного цикла.

Решают, когда принудительно остановить ход:

* исчерпан лимит шагов;
* серия неудачных вызовов инструментов (модель застряла на ошибках);
* зацикливание — модель повторяет один и тот же вызов (имя + аргументы), хотя
  между повторами ничего не менялось, и результат заведомо будет тем же. На
  ``max_repeats``-м повторе модель получает предупреждение вместе с результатом,
  следующий повтор не выполняется, ход останавливается.

«Ничего не менялось» — между двумя одинаковыми вызовами не было успешной
изменяющей операции *другого* вызова. Собственное изменение повтор не
сбрасывает: ``pytest`` без правок между запусками — зацикливание, правка →
``pytest`` — нет.

Отказ пользователя не считается ошибкой инструмента, но запоминается:
повтор отклонённого в этом ходе вызова отклоняется без нового вопроса.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from devassist.llm.types import FunctionCall

StopKind = Literal["max_steps", "tool_failures", "tool_repeats"]


@dataclass(frozen=True)
class StopReason:
    kind: StopKind
    message: str


@dataclass(frozen=True)
class ToolOutcome:
    """Исход вызова инструмента для ограничителей."""

    ok: bool
    changed: bool = False  # успешная изменяющая операция (риск ≥ WRITE)
    rejected: bool = False  # отклонено пользователем
    # Неуспех, который не говорит о застревании: команда отработала с ненулевым кодом
    # (grep без совпадений, упавшие тесты). Серию ошибок не меняет.
    soft: bool = False


@dataclass(frozen=True)
class CallCheck:
    """Решение ограничителя перед выполнением вызова."""

    repeats: int  # который раз подряд (без изменений между ними) делается этот вызов
    warning: str | None = None  # предупреждение для модели — дописывается к результату
    stop: StopReason | None = None  # вызов не выполнять, ход остановить
    rejected_before: bool = False  # пользователь уже отклонил этот вызов в этом ходе


def call_key(call: FunctionCall) -> str:
    """Канонический ключ вызова: имя + JSON аргументов с упорядоченными ключами."""
    args = json.dumps(
        call.arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return f"{call.name}:{args}"


class LoopGuard:
    """Состояние ограничителей на один ход агента."""

    def __init__(self, *, max_steps: int, max_failures: int, max_repeats: int = 3):
        self._max_steps = max_steps
        self._max_failures = max_failures
        self._max_repeats = max_repeats
        self._steps = 0
        self._failures = 0
        self._changes = 0  # успешных изменяющих операций за ход
        self._seen: dict[str, int] = {}  # ключ → значение _changes сразу после вызова
        self._repeats: dict[str, int] = {}  # ключ → повторов подряд без изменений
        self._rejected: set[str] = set()

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

    def before_tool(self, call: FunctionCall) -> CallCheck:
        """Вызывается перед выполнением инструмента; учитывает повтор."""
        key = call_key(call)
        unchanged = self._seen.get(key) == self._changes
        repeats = self._repeats.get(key, 0) + 1 if unchanged else 1
        self._repeats[key] = repeats
        rejected = key in self._rejected

        if repeats > self._max_repeats:
            return CallCheck(
                repeats,
                stop=StopReason(
                    "tool_repeats",
                    f"Прервано: агент зациклился — {repeats}-й одинаковый вызов "
                    f"{call.name} подряд без изменений между ними. "
                    "Уточните задачу или попробуйте другую модель.",
                ),
                rejected_before=rejected,
            )
        warning = None
        if repeats == self._max_repeats:
            warning = (
                f"ВНИМАНИЕ: ты уже {repeats}-й раз вызываешь {call.name} с теми же "
                "аргументами, а между вызовами ничего не менялось — результат тот же. "
                "Не повторяй этот вызов: используй уже полученный результат, смени "
                "подход или ответь пользователю. Следующий такой повтор остановит работу."
            )
        return CallCheck(repeats, warning=warning, rejected_before=rejected)

    def after_tool(self, call: FunctionCall, outcome: ToolOutcome) -> StopReason | None:
        """Вызывается после каждого выполненного (или отклонённого) вызова."""
        key = call_key(call)
        if outcome.rejected:
            self._rejected.add(key)  # серия ошибок не меняется: модель не застряла
        elif not outcome.soft:
            self._failures = 0 if outcome.ok else self._failures + 1
        if outcome.changed:
            self._changes += 1
        self._seen[key] = self._changes
        if self._failures >= self._max_failures:
            return StopReason(
                "tool_failures",
                f"Прервано: {self._failures} неудачных вызовов инструментов подряд. "
                "Похоже, агент застрял — уточните задачу или попробуйте другую модель.",
            )
        return None
