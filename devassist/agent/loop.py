"""Агентный цикл: план → действие → наблюдение → коррекция.

Один «ход» (run_turn) принимает запрос пользователя и крутит цикл:
обращение к модели → если модель просит инструмент, выполняем его (с
подтверждением для изменяющих/опасных операций) и возвращаем результат
обратно в модель → повторяем, пока модель не выдаст финальный текстовый
ответ либо не сработает ограничитель.

Ядро не зависит от UI: всё, что видит пользователь, передаётся через
:class:`~devassist.agent.events.AgentEvents`.
"""

from __future__ import annotations

from pydantic import BaseModel

from devassist.agent.events import AgentEvents, ToolCallInfo, TurnStats
from devassist.agent.session import Session
from devassist.config import Config
from devassist.llm.base import LLMProvider
from devassist.llm.types import AssistantTurn, Message, ToolSpec
from devassist.project.workspace import Workspace
from devassist.security import RiskLevel
from devassist.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: Config,
        events: AgentEvents | None = None,
        *,
        workspace: Workspace | None = None,
        session: Session | None = None,
    ):
        self._provider = provider
        self._registry = registry
        self._cfg = config
        self._events = events or AgentEvents()
        self._workspace = workspace or Workspace(config.project_root)
        self._ctx = ToolContext(workspace=self._workspace)
        self._session = session or Session(config.project_root)

    @property
    def session(self) -> Session:
        return self._session

    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> str:
        """Обрабатывает один запрос пользователя до финального ответа."""
        self._session.add_user(user_input)
        specs = self._registry.specs()
        stats = TurnStats()
        final_text = ""
        consecutive_failures = 0

        for _step in range(self._cfg.max_steps):
            stats.steps += 1
            turn = self._next_turn(specs)
            msg = turn.message
            self._session.add_assistant(msg)
            stats.prompt_tokens += turn.usage.prompt_tokens
            stats.completion_tokens += turn.usage.completion_tokens
            stats.context_tokens = turn.usage.prompt_tokens + turn.usage.completion_tokens

            if not turn.wants_tool:
                final_text = msg.content
                break

            # --- модель просит инструмент ---
            stats.tool_calls += 1
            if self._execute_tool_call(msg):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= self._cfg.max_tool_failures:
                    final_text = (
                        f"Прервано: {consecutive_failures} неудачных вызовов "
                        "инструментов подряд. Похоже, агент застрял — уточните "
                        "задачу или попробуйте другую модель."
                    )
                    stats.stop_reason = "tool_failures"
                    self._events.on_notice(final_text, level="error")
                    break
        else:
            final_text = (
                "Достигнут лимит шагов агента "
                f"({self._cfg.max_steps}). Задача может быть не завершена."
            )
            stats.stop_reason = "max_steps"
            self._events.on_notice(final_text, level="error")

        self._events.on_turn_end(stats)
        return final_text

    # ------------------------------------------------------------------ #
    def _next_turn(self, specs: list[ToolSpec]) -> AssistantTurn:
        """Один проход модели с выводом текста (потоковым или цельным)."""
        messages = self._session.messages()
        if self._cfg.stream:
            self._events.on_stream_start()
            try:
                return self._provider.stream(
                    messages,
                    tools=specs,
                    temperature=self._cfg.temperature,
                    on_delta=self._events.on_stream_delta,
                )
            finally:
                self._events.on_stream_end()

        turn = self._provider.complete(messages, tools=specs, temperature=self._cfg.temperature)
        if turn.message.content.strip():
            self._events.on_assistant_text(turn.message.content)
        return turn

    # ------------------------------------------------------------------ #
    def _execute_tool_call(self, msg: Message) -> bool:
        """Выполняет запрошенный моделью инструмент. Возвращает True при успехе."""
        assert msg.function_call is not None
        name = msg.function_call.name
        tool = self._registry.get(name)

        if tool is None:
            return self._fail(
                ToolCallInfo(name, "(неизвестный инструмент)"),
                f"инструмент '{name}' не существует.",
            )

        # 1) Валидация параметров
        try:
            params = tool.parse(msg.function_call.arguments)
        except Exception as e:  # ошибка схемы — возвращаем модели
            return self._fail(
                ToolCallInfo(name, "(неверные аргументы)"), f"валидация аргументов: {e}"
            )

        call = ToolCallInfo(name, self._describe(tool, params))
        self._events.on_tool_call(call)

        # 2) Подтверждение изменяющих/опасных операций
        previewed = False
        try:
            risk = tool.risk(params, self._ctx)
            if risk >= RiskLevel.WRITE and not self._cfg.auto_approve:
                # Превью заодно проверяет выполнимость: если оно падает, операция
                # не запускается и подтверждение не запрашивается.
                preview = tool.preview(params, self._ctx)
                dangerous = risk >= RiskLevel.DANGEROUS
                if not self._events.confirm(call, preview, dangerous=dangerous):
                    return self._fail(
                        call,
                        "отклонено пользователем",
                        model_text=(
                            "Пользователь ОТКЛОНИЛ выполнение этой операции. "
                            "Не повторяй её; предложи альтернативу или уточни план."
                        ),
                    )
                previewed = preview is not None
        except ToolError as e:
            return self._fail(call, str(e))
        except Exception as e:  # неожиданная ошибка — не роняем агента
            return self._fail(call, f"внутренняя ошибка при подготовке: {e}")

        # 3) Выполнение
        try:
            result: ToolResult = tool.run(params, self._ctx)
        except ToolError as e:
            return self._fail(call, str(e))
        except Exception as e:  # неожиданная ошибка — не роняем агента
            return self._fail(call, f"внутренняя ошибка выполнения: {e}")

        self._events.on_tool_result(call, result, previewed=previewed)
        self._session.add_function_result(name, result.as_function_content())
        return result.ok

    def _fail(self, call: ToolCallInfo, error: str, *, model_text: str | None = None) -> bool:
        """Неудачный вызов: показать пользователю и сообщить модели."""
        summary = error.splitlines()[0] if error else "ошибка"
        result = ToolResult(content=error, ok=False, summary=summary)
        self._events.on_tool_result(call, result, previewed=False)
        self._session.add_function_result(call.name, model_text or f"ОШИБКА: {error}")
        return False

    @staticmethod
    def _describe(tool: Tool, params: BaseModel) -> str:
        try:
            return tool.describe(params)
        except Exception:
            return ""
