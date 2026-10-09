"""Протокол событий агента — единственная связь ядра с пользовательским интерфейсом.

Агент не знает, как именно отображается его работа: он сообщает о событиях
(начало стрима, вызов инструмента, запрос подтверждения, итог хода), а UI
решает, как их показать. Это позволяет заменить терминальный интерфейс (live-
рендеринг Markdown, статус-строка, TUI), не трогая агентный цикл, и тестировать
цикл без rich.

:class:`AgentEvents` — базовый класс с пустыми реализациями: интерфейс
переопределяет только нужные методы. ``confirm`` по умолчанию возвращает
``False`` — интерфейс, «забывший» его реализовать, ничего не подтвердит молча.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from devassist.tools.base import Display, ToolResult

NoticeLevel = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class ToolCallInfo:
    """Вызов инструмента в том виде, в каком его показывают пользователю."""

    name: str
    summary: str = ""


@dataclass
class TurnStats:
    """Итоги одного хода агента."""

    steps: int = 0
    tool_calls: int = 0
    # Сумма по всем обращениям хода — столько токенов оплачено.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Размер контекста после последнего обращения (промпт + ответ).
    context_tokens: int = 0
    # Почему ход прерван ограничителем (None — модель завершила ход сама).
    stop_reason: str | None = None

    @property
    def billed_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class AgentEvents:
    """Получатель событий агента. Все методы — no-op; ``confirm`` → False."""

    # --- ответ модели ---
    def on_stream_start(self) -> None:
        """Запрос к модели отправлен (потоковый режим); ждём первый токен."""

    def on_stream_delta(self, text: str) -> None:
        """Очередной кусок текста модели."""

    def on_stream_end(self) -> None:
        """Потоковый ответ закончен (успешно или с ошибкой)."""

    def on_assistant_text(self, text: str) -> None:
        """Ответ модели целиком (непотоковый режим)."""

    # --- инструменты ---
    def on_tool_call(self, call: ToolCallInfo) -> None:
        """Модель вызвала инструмент."""

    def confirm(self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool) -> bool:
        """Спросить пользователя, выполнять ли изменяющую/опасную операцию."""
        return False

    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        """Результат инструмента. ``previewed`` — превью уже показано при подтверждении."""

    # --- служебное ---
    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        """Сообщение агента пользователю (лимиты, остановки, предупреждения)."""

    def on_turn_end(self, stats: TurnStats) -> None:
        """Ход завершён."""
