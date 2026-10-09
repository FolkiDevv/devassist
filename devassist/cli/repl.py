"""Интерактивный режим (REPL). Строка ввода — :mod:`devassist.cli.prompt`.

Семантика клавиш:
  * Ctrl+C на приглашении — сбросить ввод (не выход);
  * Ctrl+C или Esc во время ответа — прервать текущий ход агента;
  * Esc Esc на приглашении — очистить ввод;
  * Ctrl+D (EOF) или /exit — выход.

Диалог сохраняется после каждого хода — и прерванного тоже (см. :func:`autosave`).
"""

from __future__ import annotations

from devassist import __version__
from devassist.agent.chat_store import ChatInfo, ChatRecorder
from devassist.agent.loop import Agent
from devassist.cli.commands import CommandContext, CommandRegistry, is_repl_command
from devassist.cli.indexing import ensure_index
from devassist.cli.interrupt import EscInterrupt
from devassist.cli.models import ModelCatalog, ensure_context_window
from devassist.cli.prompt import InputReader, StatusInfo, make_input_reader
from devassist.llm.base import LLMError
from devassist.ui.console import Console


def status_of(agent: Agent) -> StatusInfo:
    """Данные для статус-строки (вычисляются при каждой отрисовке приглашения)."""
    return StatusInfo(
        model=agent.model,
        context_tokens=agent.context_tokens,
        context_budget=agent.context_budget,
        billed_tokens=agent.billed_tokens,
        auto_approve=agent.config.auto_approve,
    )


def autosave(chats: ChatRecorder | None, agent: Agent, ui: Console) -> None:
    """Сохранить чат после хода. Ход к этому моменту уже согласован (``repair``)."""
    if chats is None:
        return
    warning = chats.save(agent.conversation, model=agent.model)
    if warning:
        ui.warn(warning)


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
    chats: ChatRecorder | None = None,
    resumed: ChatInfo | None = None,
    index_on_start: bool = True,
    models: ModelCatalog | None = None,
) -> int:
    """``interrupt`` подменяется в тестах; по умолчанию Esc слушается только вместе с
    настоящей строкой ввода (``read_input is None``). ``index_on_start`` — до первого
    ввода построить индекс проекта, если его нет (:func:`~devassist.cli.indexing.ensure_index`).

    ``chats`` — куда сохранять диалог (None — не сохранять); ``resumed`` — чат,
    продолженный при запуске (``--continue``/``--resume``): после баннера показывается
    его последний обмен.

    ``models`` — каталог моделей для ``/model``; по умолчанию с настоящей строкой
    ввода список загружается в фоне. Окно контекста незамеренной модели замеряется
    до первого ввода (:func:`~devassist.cli.models.ensure_context_window`)."""
    if interrupt is None:
        interrupt = EscInterrupt(enabled=None if read_input is None else False)
    esc = esc_interrupt_for(ui, interrupt)
    if models is None and read_input is None:
        models = ModelCatalog(agent.provider)
    catalog = models
    arg_choices = (
        {"/model": lambda: catalog.choices(agent.windows, agent.model)}
        if catalog is not None
        else {}
    )
    read = read_input or make_input_reader(
        commands=commands,
        workspace=agent.workspace,
        status=lambda: status_of(agent),
        no_color=ui.no_color,
        arg_choices=arg_choices,
    )
    ui.banner(
        version=__version__,
        model=agent.model,
        root=str(agent.workspace.root),
        hints=[(cmd.name, cmd.summary) for cmd in commands],
        auto_approve=agent.config.auto_approve,
    )
    if resumed is not None:
        ui.chat_resumed(resumed, agent.conversation.messages)
    # Ввод не принимается, пока строится индекс (Esc/Ctrl+C — отменить построение).
    if index_on_start:
        ensure_index(agent.workspace, ui, esc)
    ensure_context_window(agent, ui, interrupt=esc)
    if models is not None:
        models.start()  # после замера: не делить с ним соединение; ввод сети не ждёт
    ctx = CommandContext(
        agent=agent, ui=ui, commands=commands, chats=chats, interrupt=esc, models=models
    )
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
        autosave(chats, agent, ui)
        typeahead = esc.take_typeahead()
