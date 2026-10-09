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

import time

from pydantic import BaseModel

from devassist.agent.context_window import (
    DEFAULT_CONTEXT_WINDOW,
    budget_for_window,
    estimate_specs_tokens,
    estimate_tokens,
    fit_history,
)
from devassist.agent.conversation import Conversation
from devassist.agent.events import AgentEvents, ToolCallInfo, TurnStats
from devassist.agent.guard import CallCheck, LoopGuard, ToolOutcome
from devassist.agent.prompts import build_system_prompt
from devassist.config import Config
from devassist.llm.base import LLMProvider
from devassist.llm.model_windows import ModelWindows
from devassist.llm.types import AssistantTurn, Message, ToolSpec
from devassist.project.workspace import Workspace
from devassist.security import RiskLevel
from devassist.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult

REJECTED_NOTE = (
    "Пользователь ОТКЛОНИЛ выполнение этой операции. "
    "Не повторяй её; предложи альтернативу или уточни план."
)
REJECTED_AGAIN_NOTE = (
    "Пользователь уже ОТКЛОНИЛ эту операцию в этом ходе, повторно она не предлагается. "
    "Не повторяй её; предложи альтернативу или уточни план."
)
LOOP_STOP_NOTE = (
    "Не выполнено: повтор того же вызова без изменений. Ход остановлен из-за зацикливания."
)


def _with_note(content: str, note: str | None) -> str:
    return f"{content}\n\n{note}" if note else content


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
        windows: ModelWindows | None = None,
    ):
        """Конструктор не обращается к файловой системе и сети.

        ``windows`` — замеренные окна моделей (по умолчанию пусто: окно
        :data:`~devassist.agent.context_window.DEFAULT_CONTEXT_WINDOW`).
        """
        self._provider = provider
        self._registry = registry
        self._cfg = config
        # `is None`, а не `or`: пустой Conversation ложен (__len__ == 0).
        self._events = AgentEvents() if events is None else events
        self._workspace = Workspace(config.project_root) if workspace is None else workspace
        self._ctx = ToolContext(workspace=self._workspace, ask_user=self._events.ask_user)
        self._conversation = Conversation() if conversation is None else conversation
        self._model = config.model
        self._windows = ModelWindows() if windows is None else windows
        self._system_prompt: str | None = None  # строится лениво, сбрасывается в reset()
        self._billed_tokens = 0  # потрачено за сессию (reset() не сбрасывает)

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

    @property
    def config(self) -> Config:
        return self._cfg

    @property
    def context_tokens(self) -> int:
        """Размер контекста по последнему обращению к модели (0 — диалог пуст)."""
        usage = self._conversation.last_usage
        return usage.prompt_tokens + usage.completion_tokens if usage else 0

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def windows(self) -> ModelWindows:
        return self._windows

    @property
    def context_window(self) -> int | None:
        """Замеренное окно текущей модели (None — не замерено)."""
        return self._windows.get(self._model)

    @property
    def context_budget(self) -> int:
        """Бюджет запроса (оценка в токенах): явный из конфига или от окна модели."""
        if self._cfg.context_budget_tokens is not None:
            return self._cfg.context_budget_tokens
        return budget_for_window(self.context_window or DEFAULT_CONTEXT_WINDOW)

    @property
    def billed_tokens(self) -> int:
        """Токены, оплаченные за сессию (сумма по всем обращениям, включая прерванные ходы)."""
        return self._billed_tokens

    def set_model(self, name: str) -> None:
        """Сменить модель (действует со следующего обращения)."""
        name = name.strip()
        if not name:
            raise ValueError("Имя модели не может быть пустым")
        self._model = name

    def reset(self, conversation: Conversation | None = None) -> None:
        """Начать новый диалог или продолжить сохранённый (``conversation``).

        Контекст проекта будет собран заново; потраченные токены сессии не сбрасываются.
        """
        self._conversation = Conversation() if conversation is None else conversation
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
        guard = LoopGuard(
            max_steps=self._cfg.max_steps,
            max_failures=self._cfg.max_tool_failures,
            max_repeats=self._cfg.max_tool_repeats,
        )
        stats = TurnStats()
        started = time.monotonic()
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
                self._billed_tokens += turn.usage.prompt_tokens + turn.usage.completion_tokens
                stats.context_tokens = turn.usage.prompt_tokens + turn.usage.completion_tokens

                if not turn.wants_tool:
                    final_text = msg.content
                    break

                # --- модель просит инструмент ---
                call = msg.function_call
                assert call is not None
                check = guard.before_tool(call)
                if check.stop is not None:
                    # Вызов не выполняется; история остаётся согласованной.
                    self._conversation.repair(LOOP_STOP_NOTE)
                    stop = check.stop
                else:
                    if check.warning:
                        self._events.on_notice(
                            f"Модель {check.repeats}-й раз повторяет вызов {call.name} "
                            "без изменений — предупреждена; следующий повтор остановит ход.",
                            level="warn",
                        )
                    stats.tool_calls += 1
                    outcome = self._execute_tool_call(msg, check)
                    stop = guard.after_tool(call, outcome)
            if stop is not None:
                final_text = stop.message
                stats.stop_reason = stop.kind
                self._events.on_notice(stop.message, level="error")
                break

        stats.duration_s = time.monotonic() - started
        self._events.on_turn_end(stats)
        return final_text

    # ------------------------------------------------------------------ #
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(self._workspace)
        return self._system_prompt

    def _build_request(self, specs: list[ToolSpec]) -> list[Message]:
        """Сообщения для модели: системный промпт + история в пределах бюджета.

        Единственная точка сборки запроса — сюда встраивается сжатие контекста.
        """
        system = Message(role="system", content=self.system_prompt())
        # Схемы инструментов уходят в каждом запросе и занимают то же окно.
        used = estimate_tokens([system]) + estimate_specs_tokens(specs)
        budget = max(self.context_budget - used, 1_000)
        return [system, *fit_history(self._conversation.messages, budget)]

    def _next_turn(self, specs: list[ToolSpec]) -> AssistantTurn:
        """Один проход модели с выводом текста (потоковым или цельным)."""
        messages = self._build_request(specs)
        self._events.on_stream_start()
        try:
            if self._cfg.stream:
                return self._provider.stream(
                    messages,
                    tools=specs,
                    model=self._model,
                    temperature=self._cfg.temperature,
                    on_delta=self._events.on_stream_delta,
                )
            turn = self._provider.complete(
                messages, tools=specs, model=self._model, temperature=self._cfg.temperature
            )
        finally:
            self._events.on_stream_end()

        if turn.message.content.strip():
            self._events.on_assistant_text(turn.message.content)
        return turn

    # ------------------------------------------------------------------ #
    def _execute_tool_call(self, msg: Message, check: CallCheck) -> ToolOutcome:
        """Выполняет запрошенный моделью инструмент.

        ``check.warning`` (предупреждение ограничителя) дописывается к результату,
        который получит модель.
        """
        assert msg.function_call is not None
        name = msg.function_call.name
        tool = self._registry.get(name)
        note = check.warning

        if tool is None:
            return self._fail(
                ToolCallInfo(name, "(неизвестный инструмент)"),
                f"инструмент '{name}' не существует.",
                note=note,
            )

        # 1) Валидация параметров
        try:
            params = tool.parse(msg.function_call.arguments)
        except Exception as e:  # ошибка схемы — возвращаем модели
            return self._fail(
                ToolCallInfo(name, "(неверные аргументы)"),
                f"валидация аргументов: {e}",
                note=note,
            )

        call = ToolCallInfo(name, self._describe(tool, params))
        self._events.on_tool_call(call)

        # 2) Подтверждение изменяющих/опасных операций
        previewed = False
        risk = RiskLevel.SAFE
        try:
            risk = tool.risk(params, self._ctx)
            if risk >= RiskLevel.WRITE and not self._cfg.auto_approve:
                if check.rejected_before:
                    # Тот же вызов уже отклонён в этом ходе — не переспрашиваем.
                    self._fail(
                        call,
                        "уже отклонено пользователем",
                        model_text=REJECTED_AGAIN_NOTE,
                        note=note,
                    )
                    return ToolOutcome(ok=False, rejected=True)
                # Превью заодно проверяет выполнимость: если оно падает, операция
                # не запускается и подтверждение не запрашивается.
                preview = tool.preview(params, self._ctx)
                dangerous = risk >= RiskLevel.DANGEROUS
                if not self._events.confirm(call, preview, dangerous=dangerous):
                    self._fail(call, "отклонено пользователем", model_text=REJECTED_NOTE, note=note)
                    return ToolOutcome(ok=False, rejected=True)
                previewed = preview is not None
        except ToolError as e:
            return self._fail(call, str(e), note=note)
        except Exception as e:  # неожиданная ошибка — не роняем агента
            return self._fail(call, f"внутренняя ошибка при подготовке: {e}", note=note)

        # 3) Выполнение
        result: ToolResult | None = None
        error = ""
        self._events.on_tool_start(call)
        try:
            result = tool.run(params, self._ctx)
        except ToolError as e:
            error = str(e)
        except Exception as e:  # неожиданная ошибка — не роняем агента
            error = f"внутренняя ошибка выполнения: {e}"
        finally:
            self._events.on_tool_end(call)
        if result is None:
            return self._fail(call, error or "инструмент не вернул результат", note=note)

        self._events.on_tool_result(call, result, previewed=previewed)
        self._conversation.add_function_result(name, _with_note(result.as_function_content(), note))
        return ToolOutcome(ok=result.ok, changed=result.ok and risk >= RiskLevel.WRITE)

    def _fail(
        self,
        call: ToolCallInfo,
        error: str,
        *,
        model_text: str | None = None,
        note: str | None = None,
    ) -> ToolOutcome:
        """Неудачный вызов: показать пользователю и сообщить модели."""
        summary = error.splitlines()[0] if error else "ошибка"
        result = ToolResult(content=error, ok=False, summary=summary)
        self._events.on_tool_result(call, result, previewed=False)
        content = model_text or f"ОШИБКА: {error}"
        self._conversation.add_function_result(call.name, _with_note(content, note))
        return ToolOutcome(ok=False)

    @staticmethod
    def _describe(tool: Tool, params: BaseModel) -> str:
        try:
            return tool.describe(params)
        except Exception:
            return ""
