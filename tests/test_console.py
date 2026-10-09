"""Тесты терминального UI: очистка управляющих символов, рендер событий."""

from __future__ import annotations

import io
import re

import pytest

from devassist.agent.events import ToolCallInfo, TurnStats
from devassist.tools.base import Display, ToolResult
from devassist.ui.console import Console, sanitize
from devassist.ui.format import clip_lines, format_tokens, plural


def _console():
    buf = io.StringIO()
    return Console(no_color=True, file=buf), buf


def test_sanitize_strips_escape_sequences():
    assert sanitize("a\x1b[31mred\x1b[0m\tb\nc\x07\x9b") == "a[31mred[0m\tb\nc"


def test_stream_and_output_are_sanitized():
    ui, buf = _console()
    ui.on_stream_start()
    ui.on_stream_delta("hi \x1b]0;pwned\x07there")
    ui.on_stream_end()
    ui.on_tool_result(
        ToolCallInfo("run_shell", "ls"),
        ToolResult(content="", summary="ok", display=Display("\x1b[2Jcleared", title="$ ls")),
        previewed=False,
    )
    out = buf.getvalue()
    assert "\x1b" not in out and "\x07" not in out
    assert "there" in out and "cleared" in out


def test_previewed_diff_not_repeated():
    ui, buf = _console()
    result = ToolResult(
        content="", summary="изменён a.py", display=Display("+UNIQUE", "diff", "a.py")
    )
    ui.on_tool_result(ToolCallInfo("edit_file", "a.py"), result, previewed=True)
    assert "UNIQUE" not in buf.getvalue()
    ui.on_tool_result(ToolCallInfo("edit_file", "a.py"), result, previewed=False)
    assert "UNIQUE" in buf.getvalue()


def test_turn_stats_line():
    ui, buf = _console()
    ui.on_turn_end(
        TurnStats(
            steps=2,
            tool_calls=1,
            prompt_tokens=200,
            completion_tokens=20,
            context_tokens=12_345,
            duration_s=3.21,
        )
    )
    out = buf.getvalue()
    assert "2 шага" in out and "1 инструмент " in out and "3.2 с" in out
    assert "контекст ~12.3k" in out and "потрачено 220" in out


# ------------------------------ живой режим ------------------------------ #
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _live_console(width: int = 60):
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=width, auto_refresh=False), buf


def _plain(buf: io.StringIO) -> str:
    return _ANSI_RE.sub("", buf.getvalue())


def test_streamed_markdown_is_formatted():
    ui, buf = _live_console()
    ui.on_stream_start()
    for chunk in ["# Заго", "ловок\n\nЭто **жир", "ный** текст\n", "\n- пункт\n", "\nконец"]:
        ui.on_stream_delta(chunk)
    ui.on_stream_end()
    out = _plain(buf)
    assert "Заголовок" in out and "жирный текст" in out and "конец" in out
    assert "**" not in out and "# " not in out
    assert "\x1b[1m" in buf.getvalue()  # жирный — настоящим стилем терминала
    assert ui._live is None


def test_live_area_stopped_after_interrupt(tmp_path):
    from fakes import ScriptedProvider, text_turn

    from devassist.agent.loop import Agent
    from devassist.config import Config
    from devassist.tools.base import build_default_registry

    class Interrupting(ScriptedProvider):
        def stream(self, messages, tools=None, *, on_delta=None, **kwargs):
            on_delta("Начало ответа\n\nещё")
            raise KeyboardInterrupt

    ui, buf = _live_console()
    cfg = Config(access_key="x", project_root=tmp_path, stream=True)
    agent = Agent(Interrupting([text_turn("x")]), build_default_registry(), cfg, ui)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("привет")
    assert ui._live is None
    assert "Начало ответа" in _plain(buf) and "ещё" in _plain(buf)


def test_tool_spinner_lifecycle():
    ui, _ = _live_console()
    call = ToolCallInfo("run_shell", "sleep 1")
    ui.on_tool_start(call)
    assert ui._live is not None
    ui.on_tool_end(call)
    assert ui._live is None
    ui.stop_live()  # идемпотентно


def test_single_live_area():
    ui, _ = _live_console()
    ui.on_stream_start()
    first = ui._live
    ui.on_tool_start(ToolCallInfo("x"))
    assert ui._live is not first and not first.active
    ui.stop_live()


def test_no_live_area_without_terminal():
    ui, buf = _console()
    ui.on_stream_start()
    ui.on_tool_start(ToolCallInfo("x"))
    assert ui._live is None
    ui.on_stream_end()
    assert buf.getvalue() == ""


def test_long_tool_output_is_clipped():
    ui, buf = _console()
    text = "\n".join(f"line{i}" for i in range(100))
    ui.on_tool_result(
        ToolCallInfo("run_shell", "seq"),
        ToolResult(content=text, summary="ok", display=Display(text, title="$ seq")),
        previewed=False,
    )
    out = buf.getvalue()
    assert "line0" in out and "line99" in out and "line50" not in out
    assert "пропущено 84 строки" in out


def test_confirm_question_includes_summary(monkeypatch):
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: (ui._c.print(prompt), "д")[1])
    assert ui.confirm(ToolCallInfo("edit_file", "a.py"), None, dangerous=False) is True
    assert "Применить edit_file (a.py)?" in buf.getvalue()
    monkeypatch.setattr(ui._c, "input", lambda prompt: (ui._c.print(prompt), "")[1])
    assert ui.confirm(ToolCallInfo("run_shell", "rm -rf x"), None, dangerous=True) is False
    assert "ОПАСНО" in buf.getvalue()


def test_format_helpers():
    assert [format_tokens(n) for n in (950, 12_345, 120_000, 1_234_567)] == [
        "950",
        "12.3k",
        "120k",
        "1.2M",
    ]
    assert [plural(n, "шаг", "шага", "шагов") for n in (1, 2, 5, 11, 21, 112)] == [
        "шаг",
        "шага",
        "шагов",
        "шагов",
        "шаг",
        "шагов",
    ]
    assert clip_lines("a\nb") == "a\nb"
    assert len(clip_lines("x" * 10_000)) <= 300
    long_lines = "\n".join(f"{i:03d} " + "x" * 290 for i in range(40)) + "\nИТОГ: 3 failed"
    clipped = clip_lines(long_lines)
    assert len(clipped) <= 4000
    assert clipped.startswith("000") and clipped.endswith("ИТОГ: 3 failed")


def test_animation_thread_stops():
    buf = io.StringIO()
    ui = Console(file=buf, force_terminal=True, width=60)  # с потоком анимации
    ui.on_stream_start()
    area = ui._live
    ui.on_stream_delta("текст\n\nещё\n")
    ui.on_stream_end()
    assert area._thread is not None and not area._thread.is_alive()
    assert "текст" in _plain(buf)
