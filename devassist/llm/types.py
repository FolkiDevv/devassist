"""Провайдеро-независимые типы сообщений и инструментов.

Эти типы — внутреннее представление. Конкретный провайдер (GigaChat)
сериализует их в свой формат запроса и парсит ответ обратно.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "function"]


class FunctionCall(BaseModel):
    """Запрос модели на вызов инструмента."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Универсальное сообщение диалога.

    - system/user/assistant: обычный текст в ``content``.
    - assistant с function_call: модель просит вызвать инструмент.
    - function: результат выполнения инструмента (``name`` + ``content``).
    """

    role: Role
    content: str = ""
    name: Optional[str] = None  # имя функции для role="function"
    function_call: Optional[FunctionCall] = None
    # Непрозрачный идентификатор состояния функций GigaChat — нужно
    # возвращать вместе с assistant-сообщением, чтобы сохранить контекст.
    functions_state_id: Optional[str] = None


class ToolSpec(BaseModel):
    """Описание инструмента для модели (имя, описание, JSON-schema параметров)."""

    name: str
    description: str
    parameters: Dict[str, Any]


class AssistantTurn(BaseModel):
    """Результат одного обращения к модели."""

    message: Message
    finish_reason: str = "stop"
    usage: Dict[str, Any] = Field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return self.message.function_call is not None
