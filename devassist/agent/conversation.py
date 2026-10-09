"""История диалога агента.

Чистая структура данных без побочных эффектов: не читает файловую систему и
не знает о системном промпте. Сериализуется в версионированный словарь — на
этом строится сохранение и восстановление чатов. Хранит расход токенов
последнего обращения (``last_usage``) — по нему будет срабатывать сжатие.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from devassist.llm.types import FunctionCall, Message, Usage

INTERRUPTED_NOTE = "Выполнение прервано пользователем (Ctrl+C). Результата нет."


class Conversation:
    FORMAT_VERSION = 1

    def __init__(self, messages: Iterable[Message] = (), *, last_usage: Usage | None = None):
        self._messages: list[Message] = list(messages)
        self._last_usage = last_usage

    # ------------------------------------------------------------------ #
    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def last_usage(self) -> Usage | None:
        """Расход токенов последнего обращения к модели (None — ещё не было)."""
        return self._last_usage

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------ #
    def add_user(self, text: str) -> None:
        self._messages.append(Message(role="user", content=text))

    def add_assistant(self, message: Message, usage: Usage | None = None) -> None:
        self._messages.append(message)
        if usage is not None:
            self._last_usage = usage

    def add_function_result(self, name: str, content: str) -> None:
        self._messages.append(Message(role="function", name=name, content=content))

    # ------------------------------------------------------------------ #
    def pending_call(self) -> FunctionCall | None:
        """Вызов инструмента, на который ещё нет ответа (последнее сообщение)."""
        if self._messages and self._messages[-1].role == "assistant":
            return self._messages[-1].function_call
        return None

    def repair(self, note: str = INTERRUPTED_NOTE) -> bool:
        """Закрывает «висящий» вызов инструмента ответом-заглушкой.

        Без этого следующий запрос к API содержал бы function_call без
        результата. Возвращает True, если история была исправлена.
        """
        call = self.pending_call()
        if call is None:
            return False
        self.add_function_result(call.name, note)
        return True

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.FORMAT_VERSION,
            "messages": [m.model_dump(exclude_none=True) for m in self._messages],
            "last_usage": self._last_usage.model_dump() if self._last_usage else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Conversation:
        version = data.get("version")
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"Неподдерживаемая версия формата диалога: {version!r}")
        messages = [Message.model_validate(m) for m in data.get("messages") or []]
        usage = data.get("last_usage")
        return cls(messages, last_usage=Usage.model_validate(usage) if usage else None)
