"""Инструмент для работы с git (безопасное подмножество операций)."""

from __future__ import annotations

import subprocess
from typing import List, Optional

from pydantic import BaseModel, Field

from devassist.devassist.security import RiskLevel
from devassist.devassist.tools.base import Tool, ToolContext, ToolError, ToolResult

# Разрешённые подкоманды. Деструктивные (reset --hard, clean, push --force)
# намеренно не входят — их при необходимости вызывают через run_shell с
# подтверждением.
_READ_ONLY = {"status", "diff", "log", "show", "branch", "stash-list"}
_WRITE = {"add", "commit", "checkout", "switch", "restore", "stash"}


class GitParams(BaseModel):
    subcommand: str = Field(
        description=(
            "Подкоманда git: status, diff, log, show, branch, add, commit, "
            "checkout, switch, restore, stash"
        )
    )
    args: List[str] = Field(
        default_factory=list,
        description="Дополнительные аргументы, например ['-m', 'сообщение'] для commit",
    )


class GitTool(Tool):
    name = "git"
    description = (
        "Выполняет операции git в репозитории проекта. Поддерживаются: "
        "status, diff, log, show, branch (только чтение); add, commit, checkout, "
        "switch, restore, stash (изменяющие). Деструктивные операции недоступны."
    )
    Params = GitParams

    def risk(self, params: GitParams, ctx: ToolContext) -> RiskLevel:
        return RiskLevel.SAFE if params.subcommand in _READ_ONLY else RiskLevel.WRITE

    def _check(self, params: GitParams) -> None:
        sub = params.subcommand
        if sub not in _READ_ONLY and sub not in _WRITE:
            raise ToolError(
                f"Подкоманда git '{sub}' не разрешена. Доступно: "
                f"{sorted(_READ_ONLY | _WRITE)}."
            )

    def preview(self, params: GitParams, ctx: ToolContext) -> Optional[str]:
        if self.risk(params, ctx) >= RiskLevel.WRITE:
            return f"$ git {params.subcommand} {' '.join(params.args)}"
        return None

    def run(self, params: GitParams, ctx: ToolContext) -> ToolResult:
        self._check(params)
        cmd = ["git", params.subcommand, *params.args]
        # компактный лог по умолчанию
        if params.subcommand == "log" and not params.args:
            cmd = ["git", "log", "--oneline", "-n", "20"]
        try:
            proc = subprocess.run(
                cmd, cwd=str(ctx.root), capture_output=True, text=True, timeout=60
            )
        except FileNotFoundError:
            raise ToolError("git не установлен или недоступен в PATH.")
        except subprocess.TimeoutExpired:
            return ToolResult(content="git: таймаут", ok=False, summary="git таймаут")

        out = (proc.stdout or "") + (
            ("\n[stderr]\n" + proc.stderr) if proc.stderr else ""
        )
        out = out.strip() or "(нет вывода)"
        if len(out) > 30_000:
            out = out[:30_000] + "\n...(обрезано)"
        return ToolResult(
            content=f"exit code: {proc.returncode}\n{out}",
            ok=proc.returncode == 0,
            summary=f"git {params.subcommand} → код {proc.returncode}",
            display=out,
        )
