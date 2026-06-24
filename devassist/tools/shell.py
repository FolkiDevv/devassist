"""Выполнение shell-команд с захватом вывода."""

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from devassist.devassist.security import RiskLevel, classify_shell_command
from devassist.devassist.tools.base import Tool, ToolContext, ToolResult

_MAX_OUTPUT = 30_000


class RunShellParams(BaseModel):
    command: str = Field(description="Команда для выполнения через /bin/sh")
    timeout: int = Field(default=120, description="Таймаут в секундах")


class RunShellTool(Tool):
    name = "run_shell"
    description = (
        "Выполняет shell-команду в корне проекта и возвращает stdout/stderr и код "
        "возврата. Используйте для сборки, запуска тестов, утилит. "
        "Сетевые команды (curl/wget) запрещены политикой безопасности."
    )
    Params = RunShellParams

    def risk(self, params: RunShellParams, ctx: ToolContext) -> RiskLevel:
        return classify_shell_command(params.command)

    def preview(self, params: RunShellParams, ctx: ToolContext) -> str:
        return f"$ {params.command}"

    def run(self, params: RunShellParams, ctx: ToolContext) -> ToolResult:
        try:
            proc = subprocess.run(
                params.command,
                shell=True,
                cwd=str(ctx.root),
                capture_output=True,
                text=True,
                timeout=params.timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"Команда превысила таймаут ({params.timeout}с).",
                ok=False,
                summary="таймаут команды",
            )

        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = ""
        if out:
            combined += out
        if err:
            combined += ("\n" if combined else "") + "[stderr]\n" + err
        if len(combined) > _MAX_OUTPUT:
            combined = combined[:_MAX_OUTPUT] + "\n...(вывод обрезан)"

        content = f"exit code: {proc.returncode}\n{combined or '(пустой вывод)'}"
        return ToolResult(
            content=content,
            ok=proc.returncode == 0,
            summary=f"$ {params.command[:60]} → код {proc.returncode}",
            display=combined,  # пусто → UI не покажет блок вывода
        )
