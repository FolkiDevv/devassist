"""Состояние диалога агента и управление контекстом.

Хранит историю сообщений и системный промпт. При разрастании истории
выполняет простое усечение середины (сохраняя системный промпт и последние
сообщения), чтобы укладываться в контекст модели.
"""

from __future__ import annotations

from pathlib import Path

from devassist.agent.prompts import SYSTEM_PROMPT
from devassist.context import build_project_context
from devassist.llm.types import Message


class Session:
    def __init__(self, project_root: Path, *, max_messages: int = 80):
        context = build_project_context(project_root)
        system = f"{SYSTEM_PROMPT}\n\n=== КОНТЕКСТ ПРОЕКТА ===\n{context}"
        self._system = Message(role="system", content=system)
        self._history: list[Message] = []
        self._max_messages = max_messages

    # ------------------------------------------------------------------ #
    def add_user(self, text: str) -> None:
        self._history.append(Message(role="user", content=text))

    def add_assistant(self, message: Message) -> None:
        self._history.append(message)

    def add_function_result(self, name: str, content: str) -> None:
        self._history.append(Message(role="function", name=name, content=content))

    # ------------------------------------------------------------------ #
    def messages(self) -> list[Message]:
        """Сообщения для отправки модели (системный + усечённая история)."""
        history = self._history
        if len(history) > self._max_messages:
            # Сохраняем последние max_messages сообщений. Чтобы не оборвать
            # связку assistant(function_call)->function, сдвигаемся при нужде.
            history = history[-self._max_messages :]
            if history and history[0].role == "function":
                history = history[1:]
        return [self._system, *history]

    @property
    def system_prompt(self) -> str:
        return self._system.content

    def __len__(self) -> int:
        return len(self._history)
