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
from collections.abc import Callable

from pydantic import BaseModel

from devassist.agent.compaction import CompactionError, plan_compaction, summarize
from devassist.agent.context_window import (
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_WINDOW,
    MIN_HISTORY_TOKENS,
    budget_for_window,
    estimate_specs_tokens,
    estimate_text_tokens,
    estimate_tokens,
    fit_history,
)
from devassist.agent.conversation import Conversation, Summary
from devassist.agent.events import (
    AgentEvents,
    Approval,
    CompactResult,
    ToolCallInfo,
    TurnStats,
)
from devassist.agent.guard import CallCheck, LoopGuard, ToolOutcome, call_key
from devassist.agent.prompts import (
    build_system_prompt,
    mode_prompt,
    nested_instructions_prompt,
    summary_prompt,
)
from devassist.config import Config
from devassist.llm.base import LLMError, LLMProvider
from devassist.llm.model_windows import ModelWindows
from devassist.llm.types import AssistantTurn, Message, ToolSpec, Usage
from devassist.permissions import Decision, PermissionMode, ToolKind, decide, next_mode
from devassist.project.instructions import NestedInstructions
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
STOP_SUMMARY_NOTE = (
    "=== ХОД ОСТАНОВЛЕН ===\n{reason}\nИнструменты больше не вызывай. Кратко ответь "
    "пользователю: что уже сделано, что не получилось и почему, что осталось и что "
    "предлагаешь дальше."
)
INTERRUPTED_ANSWER_NOTE = "\n\n(ответ прерван пользователем)"
BROKEN_ANSWER_NOTE = "\n\n(ответ оборван из-за ошибки)"
PLAN_BLOCKED_NOTE = (
    "Не выполнено: включён режим планирования, изменения запрещены. Не пытайся "
    "вносить их: опиши нужные изменения в плане и передай его пользователю "
    "инструментом exit_plan_mode."
)


# Сжатие контекста — доли бюджета истории (бюджет запроса за вычетом системного
# промпта и схем инструментов).
COMPACT_KEEP_SHARE = 0.25  # свежие сообщения, остающиеся дословно
COMPACT_MIN_SHARE = 0.2  # меньше — сжимать не стоит запроса к модели
SUMMARY_SHARE = 0.15  # предел краткого содержания
MAX_SUMMARY_CHARS = 12_000
MIN_SUMMARY_CHARS = 1_000

# Калибровка оценки токенов по фактическому usage: реальные токены / оценка.
MIN_TOKEN_SCALE = 0.5
MAX_TOKEN_SCALE = 1.5


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
        self._ctx = ToolContext(
            workspace=self._workspace,
            ask_user=self._events.ask_user,
            get_mode=lambda: self._mode,
            set_mode=self.set_mode,
        )
        self._conversation = Conversation() if conversation is None else conversation
        self._model = config.model
        # Режим разрешений — состояние сессии, как модель: reset() его не меняет.
        # Может смениться посреди хода (Shift+Tab из потока клавиш) — читается при
        # каждом вызове инструмента и каждом обращении к модели.
        self._mode = config.mode
        self._windows = ModelWindows() if windows is None else windows
        self._system_prompt: str | None = None  # строится лениво, сбрасывается в reset()
        # Инструкции подкаталогов, с которыми агент работал в этом диалоге.
        self._nested = NestedInstructions(self._workspace.root)
        self._billed_tokens = 0  # потрачено за сессию (reset() не сбрасывает)
        # Оценка контекста после сжатия — пока модель не сообщит настоящий размер.
        self._compacted_tokens = 0
        self._last_turn: TurnStats | None = None
        # Реальные токены / оценка по модели: оценка ~3 символа на токен грубая, а
        # бюджеты считаются в её единицах (калибруется по prompt_tokens ответов).
        self._token_scale: dict[str, float] = {}
        # Вызовы, которые пользователь разрешил «всегда» (до конца сессии; reset() не
        # сбрасывает — как и режим).
        self._session_allowed: set[str] = set()
        self._last_estimate = 0  # оценка последнего отправленного запроса

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
    def mode(self) -> PermissionMode:
        return self._mode

    @property
    def context_tokens(self) -> int:
        """Размер контекста по последнему обращению к модели (0 — диалог пуст).

        После сжатия (и при продолжении сжатого чата) — оценка: настоящий размер
        станет известен со следующим запросом. Считается один раз — свойство
        читается при каждой перерисовке статус-строки.
        """
        usage = self._conversation.last_usage
        if usage is not None:
            return usage.prompt_tokens + usage.completion_tokens
        if self._conversation.summary is None:
            return 0
        if not self._compacted_tokens:
            specs = self._registry.specs()
            self._compacted_tokens = self._to_real(
                self._overhead_tokens(specs) + self._history_tokens(self._conversation)
            )
        return self._compacted_tokens

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
        """Бюджет запроса: явный из конфига (в единицах оценки) или от окна модели."""
        if self._cfg.context_budget_tokens is not None:
            return self._cfg.context_budget_tokens
        return budget_for_window(self.context_window or DEFAULT_CONTEXT_WINDOW)

    @property
    def token_scale(self) -> float:
        """Реальные токены на единицу оценки для текущей модели (1.0 — ещё не известно)."""
        return self._token_scale.get(self._model, 1.0)

    def _estimate_budget(self) -> int:
        """Бюджет запроса в единицах оценки, в которых меряется история.

        Бюджет от окна модели — в реальных токенах: делится на калибровку, иначе
        грубая оценка (~3 символа на токен) срабатывала бы раньше времени и часть
        окна пропадала. Явный ``DEVASSIST_CONTEXT_TOKENS`` уже задан в единицах оценки.
        """
        if self._cfg.context_budget_tokens is not None:
            return self._cfg.context_budget_tokens
        return int(self.context_budget / self.token_scale)

    def _to_real(self, estimate: int) -> int:
        return int(estimate * self.token_scale)

    def _calibrate(self, prompt_tokens: int) -> None:
        """Уточнить калибровку по фактическому размеру отправленного запроса.

        Рост принимается сразу (осторожность), снижение — наполовину: один запрос с
        необычным текстом не должен резко раздвинуть бюджет.
        """
        if prompt_tokens <= 0 or self._last_estimate <= 0:
            return
        ratio = min(max(prompt_tokens / self._last_estimate, MIN_TOKEN_SCALE), MAX_TOKEN_SCALE)
        old = self._token_scale.get(self._model)
        if old is not None and ratio < old:
            ratio = (old + ratio) / 2
        self._token_scale[self._model] = ratio

    @property
    def last_turn(self) -> TurnStats | None:
        """Итоги последнего хода (None — ход прерван исключением или ещё не было)."""
        return self._last_turn

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

    def set_mode(self, mode: PermissionMode) -> None:
        """Сменить режим разрешений (действует со следующего вызова инструмента)."""
        self._mode = PermissionMode(mode)

    def cycle_mode(self) -> PermissionMode:
        """Следующий режим по кругу (Shift+Tab); возвращает новый режим."""
        self._mode = next_mode(self._mode)
        return self._mode

    def reset(self, conversation: Conversation | None = None) -> None:
        """Начать новый диалог или продолжить сохранённый (``conversation``).

        Контекст проекта будет собран заново; потраченные токены сессии не сбрасываются.
        """
        self._conversation = Conversation() if conversation is None else conversation
        self._system_prompt = None
        self._nested.reset()
        self._compacted_tokens = 0

    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> str:
        """Обрабатывает один запрос пользователя до финального ответа.

        При любом прерывании (Ctrl+C, ошибка LLM) история приводится в
        согласованное состояние, исключение пробрасывается дальше.
        """
        self._last_turn = None
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
        compact_failed = False  # не удалось — до конца хода не пытаемся снова

        while True:
            stop = guard.before_step()
            if stop is None:
                stats.steps = guard.steps
                if self._cfg.auto_compact and not compact_failed:
                    compact_failed = not self._auto_compact(specs, stats)
                turn = self._next_turn(specs)
                msg = turn.message
                self._conversation.add_assistant(msg, turn.usage)
                self._account(turn.usage, stats)

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
                stats.stop_reason = stop.kind
                self._events.on_notice(stop.message, level="error")
                final_text = self._wrap_up(specs, stop.message, stats) or stop.message
                break

        stats.duration_s = time.monotonic() - started
        self._last_turn = stats
        self._events.on_turn_end(stats)
        return final_text

    def _account(self, usage: Usage, stats: TurnStats) -> None:
        """Учесть расход обращения: статистика хода, сессия, калибровка оценки."""
        stats.prompt_tokens += usage.prompt_tokens
        stats.completion_tokens += usage.completion_tokens
        self._billed_tokens += usage.prompt_tokens + usage.completion_tokens
        stats.context_tokens = usage.prompt_tokens + usage.completion_tokens
        self._calibrate(usage.prompt_tokens)

    def _wrap_up(self, specs: list[ToolSpec], reason: str, stats: TurnStats) -> str:
        """Итог модели после остановки ограничителем: что сделано и что дальше.

        Инструменты в запросе остаются (иначе API не примет историю с вызовами), но
        вызов из ответа не выполняется — в историю идёт только текст. Сбой обращения
        к модели не мешает: остаётся сообщение ограничителя.
        """
        try:
            turn = self._next_turn(specs, note=STOP_SUMMARY_NOTE.format(reason=reason))
        except LLMError:
            return ""
        self._account(turn.usage, stats)
        text = turn.message.content.strip()
        if text:
            self._conversation.add_assistant(
                Message(role="assistant", content=turn.message.content), turn.usage
            )
        return text

    # ------------------------------------------------------------------ #
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(self._workspace)
        return self._system_prompt

    def _system_text(self) -> str:
        """Системное сообщение без краткого содержания: промпт, инструкции
        подкаталогов, правила режима."""
        parts = [
            self.system_prompt(),
            nested_instructions_prompt(self._nested.files),
            mode_prompt(self._mode),
        ]
        return "\n\n".join(part for part in parts if part)

    def _overhead_tokens(self, specs: list[ToolSpec]) -> int:
        """Системное сообщение и схемы инструментов (уходят в каждом запросе)."""
        system = Message(role="system", content=self._system_text())
        return estimate_tokens([system]) + estimate_specs_tokens(specs)

    def _history_budget(self, specs: list[ToolSpec]) -> int:
        """Бюджет истории: бюджет запроса за вычетом системного сообщения и схем."""
        return max(self._estimate_budget() - self._overhead_tokens(specs), MIN_HISTORY_TOKENS)

    @staticmethod
    def _history_tokens(conversation: Conversation) -> int:
        """История, которую видит модель: краткое содержание + сообщения после него."""
        size = estimate_tokens(conversation.context_messages())
        if conversation.summary is not None:
            size += estimate_text_tokens(summary_prompt(conversation.summary.text))
        return size

    def _build_request(self, specs: list[ToolSpec], note: str = "") -> list[Message]:
        """Сообщения для модели: системный промпт + история в пределах бюджета.

        Единственная точка сборки запроса. Краткое содержание сжатого начала диалога
        дописывается к системному сообщению (GigaChat принимает одно системное
        сообщение — первым); история — сообщения после него, а если и они не
        укладываются в бюджет, старые отбрасываются (:func:`fit_history`).
        """
        system_text = self._system_text()
        summary = self._conversation.summary
        if summary is not None:
            system_text = f"{system_text}\n\n{summary_prompt(summary.text)}"
        if note:
            system_text = f"{system_text}\n\n{note}"
        system = Message(role="system", content=system_text)
        # Схемы инструментов уходят в каждом запросе и занимают то же окно.
        used = estimate_tokens([system]) + estimate_specs_tokens(specs)
        budget = max(self._estimate_budget() - used, MIN_HISTORY_TOKENS)
        history = fit_history(self._conversation.context_messages(), budget)
        self._last_estimate = used + estimate_tokens(history)
        return [system, *history]

    # ------------------------------------------------------------------ #
    def compact(self, instructions: str = "") -> CompactResult | None:
        """Сжать весь диалог в краткое содержание (команда ``/compact``).

        ``instructions`` — пожелания пользователя к резюме. None — сжимать нечего.
        Ошибка модели — :class:`~devassist.llm.base.LLMError`; диалог тогда не меняется.
        """
        specs = self._registry.specs()
        return self._compact(self._history_budget(specs), specs, instructions=instructions)

    def _auto_compact(self, specs: list[ToolSpec], stats: TurnStats) -> bool:
        """Сжать историю, если она подошла к порогу. False — сжать не удалось."""
        budget = self._history_budget(specs)
        if self._history_tokens(self._conversation) < self._cfg.compact_threshold * budget:
            return True

        def count(usage: Usage) -> None:
            stats.prompt_tokens += usage.prompt_tokens
            stats.completion_tokens += usage.completion_tokens

        try:
            self._compact(budget, specs, auto=True, on_usage=count)
        except LLMError as e:
            self._events.on_notice(
                f"не удалось сжать контекст: {e} — старые сообщения будут отбрасываться",
                level="warn",
            )
            return False
        return True

    def _compact(
        self,
        budget: int,
        specs: list[ToolSpec],
        *,
        auto: bool = False,
        instructions: str = "",
        on_usage: Callable[[Usage], None] | None = None,
    ) -> CompactResult | None:
        """Свернуть начало диалога в краткое содержание (``budget`` — бюджет истории).

        Автоматически (``auto``) свежие сообщения остаются дословно, а сжатие
        пропускается, если сворачивать почти нечего; по команде сворачивается всё.
        Диалог меняется только после успешного ответа модели.
        """
        conversation = self._conversation
        if auto:
            plan = plan_compaction(
                conversation,
                keep_tokens=int(budget * COMPACT_KEEP_SHARE),
                min_tokens=int(budget * COMPACT_MIN_SHARE),
            )
        else:
            plan = plan_compaction(conversation, keep_tokens=0)
        if plan is None:
            return None

        def count(usage: Usage) -> None:
            self._billed_tokens += usage.prompt_tokens + usage.completion_tokens
            if on_usage is not None:
                on_usage(usage)

        overhead = self._overhead_tokens(specs)
        before = self._history_tokens(conversation)
        max_chars = int(budget * SUMMARY_SHARE) * CHARS_PER_TOKEN
        result: CompactResult | None = None
        self._events.on_compact_start(auto=auto)
        try:
            previous = conversation.summary.text if conversation.summary is not None else None
            text = summarize(
                self._provider,
                plan.messages,
                model=self._model,
                temperature=self._cfg.temperature,
                input_budget=self._estimate_budget(),
                max_chars=min(max(max_chars, MIN_SUMMARY_CHARS), MAX_SUMMARY_CHARS),
                previous=previous,
                instructions=instructions,
                on_usage=count,
            )
            summary = Summary(text=text, upto=plan.upto)
            after = self._history_tokens(Conversation(conversation.messages, summary=summary))
            if after >= before:
                raise CompactionError("краткое содержание вышло не короче самой истории")
            conversation.set_summary(summary)
            self._compacted_tokens = self._to_real(overhead + after)
            result = CompactResult(
                before_tokens=self._to_real(overhead + before),
                after_tokens=self._compacted_tokens,
                messages=len(plan.messages),
                auto=auto,
            )
        finally:
            self._events.on_compact_end(result)
        return result

    def _next_turn(self, specs: list[ToolSpec], note: str = "") -> AssistantTurn:
        """Один проход модели с выводом текста (потоковым или цельным).

        ``note`` — дополнение к системному сообщению только для этого запроса.
        Оборванный поток (Esc, сбой сети) оставляет в истории уже показанный текст с
        пометкой — модель знает, что её ответ прерван, а следующий запрос не идёт
        двумя репликами пользователя подряд.
        """
        messages = self._build_request(specs, note)
        streamed: list[str] = []

        def on_delta(text: str) -> None:
            streamed.append(text)
            self._events.on_stream_delta(text)

        self._events.on_stream_start()
        try:
            if self._cfg.stream:
                return self._provider.stream(
                    messages,
                    tools=specs,
                    model=self._model,
                    temperature=self._cfg.temperature,
                    on_delta=on_delta,
                )
            turn = self._provider.complete(
                messages, tools=specs, model=self._model, temperature=self._cfg.temperature
            )
        except BaseException as e:
            partial = "".join(streamed).rstrip()
            if partial.strip():
                tail = (
                    INTERRUPTED_ANSWER_NOTE
                    if isinstance(e, KeyboardInterrupt)
                    else (BROKEN_ANSWER_NOTE)
                )
                self._conversation.add_assistant(Message(role="assistant", content=partial + tail))
            raise
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
            unknown = ToolCallInfo(name, "(неизвестный инструмент)")
            self._events.on_tool_call(unknown)
            return self._fail(unknown, f"инструмент '{name}' не существует.", note=note)

        # 1) Валидация параметров
        try:
            params = tool.parse(msg.function_call.arguments)
        except Exception as e:  # ошибка схемы — возвращаем модели
            invalid = ToolCallInfo(name, "(неверные аргументы)")
            self._events.on_tool_call(invalid)
            return self._fail(invalid, f"валидация аргументов: {e}", note=note)

        call = ToolCallInfo(name, self._describe(tool, params), tool.kind)
        self._events.on_tool_call(call)
        key = call_key(msg.function_call)

        # 2) Разрешение по режиму: выполнить, спросить или заблокировать
        previewed = False
        risk = RiskLevel.SAFE
        try:
            risk = tool.risk(params, self._ctx)
            decision = decide(
                self._mode,
                risk,
                tool.kind,
                auto_approve=self._cfg.auto_approve,
                yes_all=self._cfg.yes_all,
            )
            if decision is Decision.ASK and key in self._session_allowed:
                decision = Decision.ALLOW  # пользователь разрешил этот вызов «всегда»
            if decision is Decision.BLOCK:
                return self._fail(
                    call,
                    "заблокировано: режим планирования",
                    model_text=PLAN_BLOCKED_NOTE,
                    note=note,
                )
            if decision is Decision.ASK:
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
                answer = self._events.confirm(call, preview, dangerous=dangerous)
                if isinstance(answer, bool):
                    answer = Approval.YES if answer else Approval.NO
                if answer is Approval.ALWAYS and not dangerous:
                    self._allow_always(call, key)
                elif answer is not Approval.YES:
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
        if result.ok:
            self._attach_instructions(tool, params)
        return ToolOutcome(
            ok=result.ok, changed=result.ok and risk >= RiskLevel.WRITE, soft=result.soft
        )

    def _allow_always(self, call: ToolCallInfo, key: str) -> None:
        """«Да, и не спрашивать»: правки — режим авто-правок, прочее — этот вызов."""
        if call.kind is ToolKind.EDIT:
            if self._mode is PermissionMode.MANUAL:
                self.set_mode(PermissionMode.ACCEPT_EDITS)
                self._events.on_notice(
                    "режим «авто-правки»: дальнейшие правки файлов — без вопросов "
                    "(Shift+Tab — сменить)"
                )
            return
        self._session_allowed.add(key)
        self._events.on_notice(
            f"{call.name} с этими аргументами больше не требует подтверждения до конца сессии"
        )

    def _attach_instructions(self, tool: Tool, params: BaseModel) -> None:
        """Подключить инструкции подкаталогов, которых коснулся вызов.

        Они уходят в системном сообщении следующих запросов — поэтому переживают
        обрезку и сжатие истории и не дублируются.
        """
        try:
            found = self._nested.add_paths(tool.paths(params))
        except Exception:  # инструкции не повод прерывать работу
            return
        if found:
            names = ", ".join(doc.name for doc in found)
            self._events.on_notice(f"подключены инструкции подкаталога: {names}")

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
