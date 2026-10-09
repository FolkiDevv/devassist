"""Интерактивный режим (REPL). Строка ввода — :mod:`devassist.cli.prompt`.

Семантика клавиш:
  * Ctrl+C на приглашении — сбросить ввод (не выход);
  * Ctrl+C или Esc во время ответа — прервать текущий ход агента;
  * Esc Esc на приглашении — очистить ввод;
  * Ctrl+D (EOF) или /exit — выход.
"""

from __future__ import annotations

from devassist import __version__
from devassist.agent.loop import Agent
from devassist.cli.commands import CommandContext, CommandRegistry, is_repl_command
from devassist.cli.indexing import ensure_index
from devassist.cli.interrupt import EscInterrupt
from devassist.cli.prompt import InputReader, StatusInfo, make_input_reader
from devassist.llm.base import LLMError
from devassist.ui.console import Console


def status_of(agent: Agent) -> StatusInfo:
    """Данные для статус-строки (вычисляются при каждой отрисовке приглашения)."""
    return StatusInfo(
        model=agent.model,
        context_tokens=agent.context_tokens,
        context_budget=agent.config.context_budget_tokens,
        billed_tokens=agent.billed_tokens,
        auto_approve=agent.config.auto_approve,
    )


def esc_interrupt_for(ui: Console, interrupt: EscInterrupt | None = None) -> EscInterrupt:
    """Прерывание ходов клавишей Esc (если stdin — терминал) + подсказка в индикаторе."""
    interrupt = EscInterrupt() if interrupt is None else interrupt
    if interrupt.enabled:
        ui.set_interrupt_keys("Esc — прервать", interrupt.paused)
    return interrupt


def run_repl(
    agent: Agent,
    ui: Console,
    commands: CommandRegistry,
    *,
    read_input: InputReader | None = None,
    interrupt: EscInterrupt | None = None,
    index_on_start: bool = True,
) -> int:
    """``interrupt`` подменяется в тестах; по умолчанию Esc слушается только вместе с
    настоящей строкой ввода (``read_input is None``). ``index_on_start`` — до первого
    ввода построить индекс проекта, если его нет (:func:`~devassist.cli.indexing.ensure_index`)."""
    if interrupt is None:
        interrupt = EscInterrupt(enabled=None if read_input is None else False)
    esc = esc_interrupt_for(ui, interrupt)
    read = read_input or make_input_reader(
        commands=commands,
        workspace=agent.workspace,
        status=lambda: status_of(agent),
        no_color=ui.no_color,
    )
    ui.banner(
        version=__version__,
        model=agent.model,
        root=str(agent.workspace.root),
        hints=[(cmd.name, cmd.summary) for cmd in commands],
        auto_approve=agent.config.auto_approve,
    )
    # Ввод не принимается, пока строится индекс (Esc/Ctrl+C — отменить построение).
    if index_on_start:
        ensure_index(agent.workspace, ui, esc)
    ctx = CommandContext(agent=agent, ui=ui, commands=commands, interrupt=esc)
    typeahead = esc.take_typeahead()  # набранное во время хода — в следующую строку ввода

    while True:
        try:
            # Набранное во время хода подставляется один раз: Ctrl+C его сбрасывает.
            default, typeahead = typeahead, ""
            line = (read(default=default) if default else read()).strip()
        except EOFError:
            ui.system("\nдо встречи!")
            return 0
        except KeyboardInterrupt:
            ui.system("(Ctrl+D или /exit — выход)")
            continue
        if not line:
            continue

        if is_repl_command(line):
            if not commands.dispatch(line, ctx):
                return 0
            typeahead = esc.take_typeahead()
            continue

        try:
            ui.print()
            with esc:
                agent.run_turn(line)
            ui.print()
        except LLMError as e:
            ui.stop_live()
            ui.error(str(e))
        except KeyboardInterrupt:
            ui.stop_live()
            ui.system("\n(прервано)")
        except Exception as e:  # ошибка внутри хода не должна закрывать REPL
            ui.stop_live()
            ui.error(f"внутренняя ошибка: {type(e).__name__}: {e}")
        typeahead = esc.take_typeahead()
