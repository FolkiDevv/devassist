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

from devassist.agent.context_window import estimate_tokens, fit_history
from devassist.agent.conversation import Conversation
from devassist.agent.events import AgentEvents, ToolCallInfo, TurnStats
from devassist.agent.guard import LoopGuard
from devassist.agent.prompts import build_system_prompt
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
        conversation: Conversation | None = None,
    ):
        """Конструктор не обращается к файловой системе и сети."""
        self._provider = provider
        self._registry = registry
        self._cfg = config
        self._events = events or AgentEvents()
        self._workspace = workspace or Workspace(config.project_root)
        self._ctx = ToolContext(workspace=self._workspace)
        self._conversation = conversation or Conversation()
        self._model = config.model
        self._system_prompt: str | None = None  # строится лениво, сбрасывается в reset()

    # ------------------------------------------------------------------ #
    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, name: str) -> None:
        """Сменить модель (действует со следующего обращения)."""
        name = name.strip()
        if not name:
            raise ValueError("Имя модели не может быть пустым")
        self._model = name

    def reset(self) -> None:
        """Начать новый диалог; контекст проекта будет собран заново."""
        self._conversation = Conversation()
        self._system_prompt = None

    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> str:
        """Обрабатывает один запрос пользователя до финального ответа.

        При любом прерывании (Ctrl+C, ошибка LLM) история приводится в
        согласованное состояние, исключение пробрасывается дальше.
        """
        try:
            return self._run_turn(user_input)
        except BaseException:
            self._conversation.repair()
            raise

    def _run_turn(self, user_input: str) -> str:
        self._conversation.add_user(user_input)
        specs = self._registry.specs()
        guard = LoopGuard(max_steps=self._cfg.max_steps, max_failures=self._cfg.max_tool_failures)
        stats = TurnStats()
        final_text = ""

        while True:
            stop = guard.before_step()
            if stop is None:
                stats.steps = guard.steps
                turn = self._next_turn(specs)
                msg = turn.message
                self._conversation.add_assistant(msg, turn.usage)
                stats.prompt_tokens += turn.usage.prompt_tokens
                stats.completion_tokens += turn.usage.completion_tokens
                stats.context_tokens = turn.usage.prompt_tokens + turn.usage.completion_tokens

                if not turn.wants_tool:
                    final_text = msg.content
                    break

                # --- модель просит инструмент ---
                assert msg.function_call is not None
                stats.tool_calls += 1
                ok = self._execute_tool_call(msg)
                stop = guard.after_tool(msg.function_call, ok)
            if stop is not None:
                final_text = stop.message
                stats.stop_reason = stop.kind
                self._events.on_notice(stop.message, level="error")
                break

        self._events.on_turn_end(stats)
        return final_text

    # ------------------------------------------------------------------ #
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(self._workspace)
        return self._system_prompt

    def _build_request(self) -> list[Message]:
        """Сообщения для модели: системный промпт + история в пределах бюджета.

        Единственная точка сборки запроса — сюда встраивается сжатие контекста.
        """
        system = Message(role="system", content=self.system_prompt())
        budget = max(self._cfg.context_budget_tokens - estimate_tokens([system]), 1_000)
        return [system, *fit_history(self._conversation.messages, budget)]

    def _next_turn(self, specs: list[ToolSpec]) -> AssistantTurn:
        """Один проход модели с выводом текста (потоковым или цельным)."""
        messages = self._build_request()
        if self._cfg.stream:
            self._events.on_stream_start()
            try:
                return self._provider.stream(
                    messages,
                    tools=specs,
                    model=self._model,
                    temperature=self._cfg.temperature,
                    on_delta=self._events.on_stream_delta,
                )
            finally:
                self._events.on_stream_end()

        turn = self._provider.complete(
            messages, tools=specs, model=self._model, temperature=self._cfg.temperature
        )
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
        self._conversation.add_function_result(name, result.as_function_content())
        return result.ok

    def _fail(self, call: ToolCallInfo, error: str, *, model_text: str | None = None) -> bool:
        """Неудачный вызов: показать пользователю и сообщить модели."""
        summary = error.splitlines()[0] if error else "ошибка"
        result = ToolResult(content=error, ok=False, summary=summary)
        self._events.on_tool_result(call, result, previewed=False)
        self._conversation.add_function_result(call.name, model_text or f"ОШИБКА: {error}")
        return False

    @staticmethod
    def _describe(tool: Tool, params: BaseModel) -> str:
        try:
            return tool.describe(params)
        except Exception:
            return ""
