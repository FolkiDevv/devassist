"""Тесты терминального UI: очистка управляющих символов, рендер событий."""

from __future__ import annotations

import io

from devassist.agent.events import ToolCallInfo, TurnStats
from devassist.tools.base import Display, ToolResult
from devassist.ui.console import Console, sanitize


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
            steps=2, tool_calls=1, prompt_tokens=200, completion_tokens=20, context_tokens=120
        )
    )
    out = buf.getvalue()
    assert "2 шага" in out and "контекст ~120" in out and "потрачено 220" in out
