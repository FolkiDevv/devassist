"""Тестовые двойники: провайдер с заготовленными ходами и записывающий UI."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from devassist.agent.events import AgentEvents, ToolCallInfo, TurnStats
from devassist.llm.base import LLMProvider, PromptTooLong
from devassist.llm.types import AssistantTurn, FunctionCall, Message, Usage
from devassist.tools.base import Display, Tool, ToolContext, ToolResult


class ScriptedProvider(LLMProvider):
    """Провайдер-заглушка: выдаёт заранее заданную последовательность ходов.

    Элемент сценария — AssistantTurn либо исключение (будет брошено). Каждый
    запрос записывается в ``requests`` (сообщения + параметры вызова).
    """

    def __init__(self, turns: Iterable[AssistantTurn | BaseException] = ()):
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, messages, tools=None, **kwargs) -> AssistantTurn:
        self.requests.append({"messages": list(messages), "tools": tools, **kwargs})
        if self._turns:
            item = self._turns.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return text_turn("конец")

    def stream(self, messages, tools=None, *, on_delta=None, **kwargs) -> AssistantTurn:
        turn = self.complete(messages, tools, **kwargs)
        if on_delta and turn.message.content:
            on_delta(turn.message.content)
        return turn


class WindowProvider(ScriptedProvider):
    """Провайдер с окном ``window`` токенов для замера (токен ≈ ``chars_per_token`` символа).

    Отказ — :class:`PromptTooLong` со статусом ``status``; ``hint`` — текст отказа
    называет лимит (как vLLM). Каждая проба пишется в ``measured`` (модель, токены).
    """

    supports_measure = True

    def __init__(
        self,
        window: int,
        turns: Iterable[AssistantTurn | BaseException] = (),
        *,
        chars_per_token: float = 3.3,
        hint: bool = False,
        status: int | None = 422,
        error: BaseException | None = None,
        overhead: int = 5,
    ):
        super().__init__(turns)
        self.window = window
        self.chars_per_token = chars_per_token
        self.hint = hint
        self.status = status
        self.error = error
        self.overhead = overhead  # служебные токены сообщения
        self.measured: list[tuple[str | None, int]] = []

    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        if self.error is not None:
            raise self.error
        tokens = int(len(text) / self.chars_per_token) + self.overhead
        self.measured.append((model, tokens))
        if tokens + 1 > self.window:
            detail = (
                f"This model's maximum context length is {self.window} tokens. "
                f"However, you requested {tokens + 1} tokens"
                if self.hint
                else "Request too large"
            )
            raise PromptTooLong(self.status, detail)
        return tokens


def text_turn(text: str, usage: Usage | None = None) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), usage=usage or Usage())


def tool_turn(name: str, args: dict, usage: Usage | None = None) -> AssistantTurn:
    return AssistantTurn(
        message=Message(
            role="assistant",
            content="",
            function_call=FunctionCall(name=name, arguments=args),
        ),
        finish_reason="function_call",
        usage=usage or Usage(),
    )


class RecordingEvents(AgentEvents):
    """UI-двойник: запоминает события, на подтверждение отвечает ``confirm_answer``."""

    def __init__(self, confirm_answer: bool = True, answers: list | None = None):
        self.confirm_answer = confirm_answer
        self.answers = answers  # ответы на ask_user (None — отказ)
        self.questions: list = []
        self.events: list[tuple[str, Any]] = []
        self.confirms: list[tuple[ToolCallInfo, Display | None, bool]] = []
        self.results: list[tuple[ToolCallInfo, ToolResult, bool]] = []
        self.stats: list[TurnStats] = []
        self.notices: list[tuple[str, str]] = []

    def on_stream_start(self) -> None:
        self.events.append(("stream_start", None))

    def on_stream_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def on_stream_end(self) -> None:
        self.events.append(("stream_end", None))

    def on_assistant_text(self, text: str) -> None:
        self.events.append(("text", text))

    def on_tool_call(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_call", call))

    def confirm(self, call, preview, *, dangerous):
        self.confirms.append((call, preview, dangerous))
        return self.confirm_answer

    def on_tool_start(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_start", call))

    def on_tool_end(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_end", call))

    def on_tool_result(self, call, result, *, previewed):
        self.events.append(("tool_result", call))
        self.results.append((call, result, previewed))

    def ask_user(self, questions):
        self.questions.append(list(questions))
        return self.answers

    def on_notice(self, text, *, level="info"):
        self.notices.append((level, text))

    def on_turn_end(self, stats: TurnStats) -> None:
        self.stats.append(stats)


class FakeParams(BaseModel):
    path: str = "x"


class FakeTool(Tool):
    """Настраиваемый WRITE-инструмент для тестов агентного цикла."""

    name = "fake_write"
    description = "тестовый инструмент"
    Params = FakeParams

    def __init__(
        self,
        *,
        preview: Callable[[], Display | None] | None = None,
        run: Callable[[], ToolResult] | None = None,
    ):
        self._preview = preview or (lambda: Display("+x", kind="diff", title="x"))
        self._run = run or (lambda: ToolResult(content="ok", summary="ok"))
        self.runs = 0

    def risk(self, params, ctx: ToolContext):
        from devassist.security import RiskLevel

        return RiskLevel.WRITE

    def preview(self, params, ctx):
        return self._preview()

    def run(self, params, ctx):
        self.runs += 1
        return self._run()
