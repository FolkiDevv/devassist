"""Агентный цикл: план → действие → наблюдение → коррекция.

Один «ход» (run_turn) принимает запрос пользователя и крутит цикл:
обращение к модели → если модель просит инструмент, выполняем его (с
подтверждением для изменяющих/опасных операций) и возвращаем результат
обратно в модель → повторяем, пока модель не выдаст финальный текстовый
ответ либо не исчерпается лимит шагов.
"""

from __future__ import annotations

from typing import Optional

from devassist.devassist.agent.session import Session
from devassist.devassist.config import Config
from devassist.devassist.llm.base import LLMProvider
from devassist.devassist.llm.types import Message
from devassist.devassist.security import RiskLevel
from devassist.devassist.tools.base import ToolContext, ToolError, ToolResult
from devassist.devassist.tools.base import ToolRegistry  # type: ignore
from devassist.devassist.ui.console import Console


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: Config,
        ui: Console,
        session: Optional[Session] = None,
    ):
        self._provider = provider
        self._registry = registry
        self._cfg = config
        self._ui = ui
        self._ctx = ToolContext(config=config)
        self._session = session or Session(config.project_root)

    @property
    def session(self) -> Session:
        return self._session

    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> str:
        """Обрабатывает один запрос пользователя до финального ответа."""
        self._session.add_user(user_input)
        specs = self._registry.specs()
        final_text = ""
        steps = 0
        tools_used = 0
        total_tokens = 0
        consecutive_failures = 0

        for step in range(self._cfg.max_steps):
            steps += 1
            turn = self._next_turn(specs)
            msg = turn.message
            self._session.add_assistant(msg)
            total_tokens += int(turn.usage.get("total_tokens", 0) or 0)

            if not turn.wants_tool:
                final_text = msg.content
                break

            # --- модель просит инструмент ---
            tools_used += 1
            ok = self._execute_tool_call(msg)
            if ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= self._cfg.max_tool_failures:
                    final_text = (
                        f"Прервано: {consecutive_failures} неудачных вызовов "
                        "инструментов подряд. Похоже, агент застрял — уточните "
                        "задачу или попробуйте другую модель."
                    )
                    self._ui.error(final_text)
                    break
        else:
            final_text = (
                "Достигнут лимит шагов агента "
                f"({self._cfg.max_steps}). Задача может быть не завершена."
            )
            self._ui.error(final_text)

        self._ui.turn_stats(steps=steps, tokens=total_tokens, tools=tools_used)
        return final_text

    # ------------------------------------------------------------------ #
    def _next_turn(self, specs):
        """Один проход модели с выводом текста (потоковым или цельным)."""
        if self._cfg.stream:
            self._ui.begin_stream()
            try:
                turn = self._provider.stream(
                    self._session.messages(),
                    tools=specs,
                    temperature=self._cfg.temperature,
                    on_delta=self._ui.stream_write,
                )
            finally:
                self._ui.end_stream()
            return turn

        turn = self._provider.complete(
            self._session.messages(),
            tools=specs,
            temperature=self._cfg.temperature,
        )
        # Показываем текст модели (рассуждения/план), если есть.
        if turn.message.content.strip():
            self._ui.assistant(turn.message.content)
        return turn

    # ------------------------------------------------------------------ #
    def _execute_tool_call(self, msg: Message) -> bool:
        """Выполняет запрошенный моделью инструмент. Возвращает True при успехе."""
        assert msg.function_call is not None
        name = msg.function_call.name
        raw_args = msg.function_call.arguments
        tool = self._registry.get(name)

        if tool is None:
            self._ui.tool_call(name, "(неизвестный инструмент)")
            self._session.add_function_result(
                name, f"ОШИБКА: инструмент '{name}' не существует."
            )
            return False

        # 1) Валидация параметров
        try:
            params = tool.parse(raw_args)
        except Exception as e:  # ошибка схемы — возвращаем модели
            self._ui.tool_call(name, "(неверные аргументы)")
            self._session.add_function_result(
                name, f"ОШИБКА валидации аргументов: {e}"
            )
            return False

        # 2) Краткая сводка вызова
        summary = self._call_summary(name, params)
        self._ui.tool_call(name, summary)

        # 3) Подтверждение для изменяющих/опасных операций
        risk = tool.risk(params, self._ctx)
        if risk >= RiskLevel.WRITE and not self._cfg.auto_approve:
            preview = None
            try:
                preview = tool.preview(params, self._ctx)
            except ToolError:
                # превью не удалось (например, файл не найден) — пусть run() вернёт ошибку
                return self._run_and_record(tool, params, name)
            if preview:
                if name in ("write_file", "edit_file"):
                    self._ui.diff(preview, title=getattr(params, "path", "изменения"))
                else:
                    self._ui.output_block(preview)
            dangerous = risk >= RiskLevel.DANGEROUS
            question = (
                f"Выполнить опасную операцию '{name}'?"
                if dangerous
                else f"Применить '{name}'?"
            )
            if not self._ui.confirm(question, dangerous=dangerous):
                self._ui.tool_result("отклонено пользователем", ok=False)
                self._session.add_function_result(
                    name,
                    "Пользователь ОТКЛОНИЛ выполнение этой операции. "
                    "Не повторяй её; предложи альтернативу или уточни план.",
                )
                return False

        # 4) Выполнение
        return self._run_and_record(tool, params, name)

    def _run_and_record(self, tool, params, name: str) -> bool:
        try:
            result: ToolResult = tool.run(params, self._ctx)
        except ToolError as e:
            self._ui.tool_result(str(e), ok=False)
            self._session.add_function_result(name, f"ОШИБКА: {e}")
            return False
        except Exception as e:  # неожиданная ошибка — не роняем агента
            self._ui.tool_result(f"внутренняя ошибка: {e}", ok=False)
            self._session.add_function_result(name, f"ОШИБКА выполнения: {e}")
            return False

        # Показ результата пользователю
        self._ui.tool_result(result.summary or "готово", ok=result.ok)
        if result.display and name in ("write_file", "edit_file"):
            # дифф уже показывали в превью при подтверждении; повторно не дублируем,
            # но в auto_approve режиме покажем здесь
            if self._cfg.auto_approve:
                title = getattr(params, "path", None) or "изменения"
                self._ui.diff(result.display, title=str(title))
        elif result.display and name in ("run_shell", "git", "search_content",
                                         "read_file", "list_dir", "find_files"):
            self._ui.output_block(result.display[:2000], title=self._output_title(name, params))

        self._session.add_function_result(name, result.as_function_content())
        return result.ok

    @staticmethod
    def _output_title(name: str, params) -> str:
        d = params.model_dump()
        if name == "run_shell":
            return f"$ {str(d.get('command',''))[:60]}"
        if name == "git":
            return f"git {d.get('subcommand','')}"
        if name == "search_content":
            return f"поиск: /{d.get('pattern','')}/"
        if name in ("read_file", "list_dir"):
            return str(d.get("path", ""))
        if name == "find_files":
            return f"файлы: {d.get('pattern','')}"
        return "вывод"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _call_summary(name: str, params) -> str:
        d = params.model_dump()
        if "path" in d:
            return str(d["path"])
        if "command" in d:
            return str(d["command"])[:70]
        if "pattern" in d:
            return f"/{d['pattern']}/"
        if "subcommand" in d:
            return f"{d['subcommand']} {' '.join(d.get('args', []))}".strip()
        return ""
