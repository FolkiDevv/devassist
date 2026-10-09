"""Тесты CLI (без сети): разбор команд, реестр, REPL, одноразовый режим."""

from __future__ import annotations

import pytest
from fakes import ScriptedProvider, text_turn, tool_turn

import devassist.cli.app as app
from devassist.agent.chat_store import ChatRecorder, ChatStore
from devassist.agent.loop import Agent
from devassist.cli.commands import (
    CommandContext,
    SlashCommand,
    default_commands,
    is_repl_command,
)
from devassist.cli.repl import run_repl
from devassist.config import Config
from devassist.project.workspace import Workspace
from devassist.tools.base import build_default_registry
from devassist.ui.console import Console


@pytest.mark.parametrize(
    "line",
    ["/help", "/exit", "/q", "/clear", "/model GigaChat-2-Max", "/model"],
)
def test_recognised_commands(line):
    assert is_repl_command(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "/home/kestrel/repos/devassist",  # абсолютный путь — НЕ команда
        "/usr/bin/python3 запусти это",
        "/",  # просто слеш
        "проанализируй /home/kestrel/x",  # путь внутри запроса
        "посмотри код",
        "",
    ],
)
def test_non_commands_go_to_agent(line):
    assert is_repl_command(line) is False


# ------------------------------ реестр ------------------------------ #
@pytest.fixture
def cli_env(tmp_path, capsys):
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    provider = ScriptedProvider()
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=ui, commands=commands)
    return agent, ui, commands, ctx, provider


def _out(capsys) -> str:
    return capsys.readouterr().out


def test_help_lists_all_commands(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/help", ctx) is True
    out = _out(capsys)
    for cmd in commands:
        assert cmd.name in out
    assert "/quit" in out


def test_aliases_and_case_insensitive(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.get("/Q") is commands.get("/exit")
    assert commands.dispatch("/QUIT", ctx) is False


def test_unknown_command_reports_and_continues(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/nope", ctx) is True
    assert "неизвестная команда" in _out(capsys)


def test_duplicate_registration_rejected(cli_env):
    _, _, commands, _, _ = cli_env
    with pytest.raises(ValueError):
        commands.register(SlashCommand("/x", "x", lambda c, a: True, aliases=("/help",)))


def test_model_and_clear_commands(cli_env, capsys):
    agent, _, commands, ctx, _ = cli_env
    commands.dispatch("/model GigaChat-2-Max", ctx)
    assert agent.model == "GigaChat-2-Max"
    commands.dispatch("/model", ctx)
    assert "GigaChat-2-Max" in _out(capsys)
    agent.conversation.add_user("старое")
    commands.dispatch("/clear", ctx)
    assert len(agent.conversation) == 0


# ------------------------------- REPL ------------------------------- #
def _reader(*items):
    queue = list(items)

    def read():
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def test_repl_ctrl_c_at_prompt_does_not_exit(cli_env, capsys):
    agent, ui, commands, _, _ = cli_env
    read = _reader(KeyboardInterrupt(), "/exit")
    assert run_repl(agent, ui, commands, read_input=read) == 0
    out = _out(capsys)
    assert "Ctrl+D" in out and "/model" in out  # подсказка + баннер из реестра


def test_repl_survives_errors_in_turn(cli_env, capsys, monkeypatch):
    agent, ui, commands, _, provider = cli_env

    def broken(_line):
        raise RuntimeError("что-то сломалось")

    monkeypatch.setattr(agent, "run_turn", broken)
    read = _reader("сделай", "ещё раз", EOFError())
    assert run_repl(agent, ui, commands, read_input=read) == 0
    assert _out(capsys).count("что-то сломалось") == 2


# ---------------------------- one-shot e2e ---------------------------- #
@pytest.fixture
def oneshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACHAT_ACCESS_KEY", "dummy")
    for name in (
        "GIGACHAT_TIMEOUT",
        "DEVASSIST_TEMPERATURE",
        "DEVASSIST_CONTEXT_TOKENS",
        "DEVASSIST_SAVE_CHATS",
    ):
        monkeypatch.delenv(name, raising=False)

    def use(provider):
        monkeypatch.setattr(app, "_make_provider", lambda _cfg: provider)

    return use


def test_oneshot_end_to_end(tmp_path, oneshot, capsys):
    (tmp_path / "hello.txt").write_text("привет из файла", encoding="utf-8")
    provider = ScriptedProvider(
        [tool_turn("read_file", {"path": "hello.txt"}), text_turn("В файле приветствие.")]
    )
    oneshot(provider)
    code = app.main(["-C", str(tmp_path), "-p", "что в hello.txt?", "--no-color", "--no-stream"])
    assert code == 0
    out = capsys.readouterr().out
    assert "В файле приветствие." in out and "read_file" in out
    assert "привет из файла" in provider.requests[-1]["messages"][-1].content
    # чат сохранён — его можно продолжить через -c
    (saved,) = ChatStore(Workspace(tmp_path)).recent()
    assert saved.title == "что в hello.txt?" and saved.requests == 1


def test_oneshot_ctrl_c_exit_code(tmp_path, oneshot):
    oneshot(ScriptedProvider([KeyboardInterrupt()]))
    assert app.main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 130


def test_oneshot_llm_error_exit_code(tmp_path, oneshot):
    from devassist.llm.base import LLMError

    oneshot(ScriptedProvider([LLMError("сеть упала")]))
    assert app.main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 2


def test_missing_project_dir(tmp_path, oneshot):
    assert app.main(["-C", str(tmp_path / "nope"), "-p", "x", "--no-color"]) == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["--version"])
    assert exc.value.code == 0
    assert "devassist" in capsys.readouterr().out


def test_repl_status_and_auto_approve_banner(cli_env, capsys):
    from devassist.cli.repl import status_of

    agent, ui, commands, _, _ = cli_env  # cli_env собран с auto_approve=True
    status = status_of(agent)
    assert status.model == agent.model and status.auto_approve is True
    assert status.context_budget == agent.config.context_budget_tokens
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "авто-подтверждение" in _out(capsys)


def test_repl_prefills_typeahead_and_uses_esc(cli_env, capsys):
    import contextlib

    agent, ui, commands, _, provider = cli_env

    class FakeEsc:
        enabled = True
        entered = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        typeahead = ["", "набрано во время хода"]  # после построения индекса, после хода

        def take_typeahead(self):
            return self.typeahead.pop(0) if self.typeahead else ""

    calls = []
    answers = ["привет", "/exit"]

    def read(**kwargs):
        calls.append(kwargs)
        return answers.pop(0)

    esc = FakeEsc()
    assert run_repl(agent, ui, commands, read_input=read, interrupt=esc) == 0
    assert esc.entered == 2  # построение индекса и ход — внутри перехвата Esc, /exit — нет
    assert calls == [{}, {"default": "набрано во время хода"}]
    assert ui._interrupt_hint == "Esc — прервать"


def test_ctrl_c_discards_prefilled_typeahead(cli_env):
    import contextlib

    agent, ui, commands, _, _ = cli_env

    class FakeEsc:
        enabled = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        typeahead = ["", "набрано"]

        def take_typeahead(self):
            return self.typeahead.pop(0) if self.typeahead else ""

    calls = []
    answers = ["привет", KeyboardInterrupt(), "/exit"]

    def read(**kwargs):
        calls.append(kwargs)
        item = answers.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    run_repl(agent, ui, commands, read_input=read, interrupt=FakeEsc())
    assert calls == [{}, {"default": "набрано"}, {}]


def test_plain_reader_keeps_typeahead(monkeypatch):
    from devassist.cli import prompt

    monkeypatch.setattr("builtins.input", lambda p: " и ещё")
    assert prompt._read_plain("набрано") == "набрано и ещё"


# ---------------------------- сохранение чатов ---------------------------- #
def _main(tmp_path, *args: str) -> int:
    return app.main(["-C", str(tmp_path), "--no-color", "--no-stream", *args])


def _store(tmp_path) -> ChatStore:
    return ChatStore(Workspace(tmp_path))


def _history(provider) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in provider.requests[-1]["messages"][1:]]


def test_oneshot_no_save(tmp_path, oneshot, monkeypatch):
    oneshot(ScriptedProvider([text_turn("ok")]))
    assert _main(tmp_path, "-p", "x", "--no-save") == 0
    monkeypatch.setenv("DEVASSIST_SAVE_CHATS", "0")
    assert _main(tmp_path, "-p", "x") == 0
    assert not (tmp_path / ".devassist").exists()


def test_oneshot_continue_appends_to_latest_chat(tmp_path, oneshot):
    oneshot(ScriptedProvider([text_turn("первый ответ")]))
    assert _main(tmp_path, "-p", "первый") == 0
    provider = ScriptedProvider([text_turn("второй ответ")])
    oneshot(provider)
    assert _main(tmp_path, "-c", "-p", "второй") == 0
    assert _history(provider) == [
        ("user", "первый"),
        ("assistant", "первый ответ"),
        ("user", "второй"),
    ]
    (chat,) = _store(tmp_path).recent()
    assert chat.title == "первый" and chat.requests == 2 and chat.preview == "второй ответ"


def test_continue_without_chats_starts_new(tmp_path, oneshot, capsys):
    oneshot(ScriptedProvider([text_turn("ok")]))
    assert _main(tmp_path, "-c", "-p", "x") == 0
    assert "сохранённых чатов нет" in capsys.readouterr().out


def test_resume_by_id_and_unknown_id(tmp_path, oneshot, capsys):
    oneshot(ScriptedProvider([text_turn("a")]))
    _main(tmp_path, "-p", "первый")
    (chat,) = _store(tmp_path).recent()
    provider = ScriptedProvider([text_turn("b")])
    oneshot(provider)
    assert _main(tmp_path, "--resume", chat.id[:15], "-p", "дальше") == 0
    assert _history(provider)[0] == ("user", "первый")
    assert _main(tmp_path, "-r", "1999", "-p", "x") == 1
    assert "не найден" in capsys.readouterr().out


def test_resume_without_id_opens_picker(tmp_path, oneshot, monkeypatch):
    oneshot(ScriptedProvider([text_turn("a"), text_turn("b")]))
    _main(tmp_path, "-p", "первый")
    _main(tmp_path, "--no-save", "-p", "мимо")  # не сохраняется — в списке один чат
    shown = []

    def pick(self, chats, current_id=""):
        shown.append(chats)
        return chats[0]

    monkeypatch.setattr(Console, "pick_chat", pick)
    provider = ScriptedProvider([text_turn("c")])
    oneshot(provider)
    assert _main(tmp_path, "-r", "-p", "дальше") == 0
    assert [c.title for c in shown[0]] == ["первый"]
    assert _history(provider)[0] == ("user", "первый")
    # отмена выбора — новый чат
    monkeypatch.setattr(Console, "pick_chat", lambda self, chats, current_id="": None)
    provider = ScriptedProvider([text_turn("d")])
    oneshot(provider)
    assert _main(tmp_path, "-r", "-p", "новый") == 0
    assert _history(provider) == [("user", "новый")]
    assert len(_store(tmp_path).recent()) == 2


def test_continue_and_resume_are_exclusive(tmp_path, oneshot):
    with pytest.raises(SystemExit):
        _main(tmp_path, "-c", "-r", "x")


def test_interrupted_oneshot_is_saved_consistently(tmp_path, oneshot):
    oneshot(ScriptedProvider([tool_turn("read_file", {"path": "a.txt"}), KeyboardInterrupt()]))
    assert _main(tmp_path, "-p", "прочитай") == 130
    store = _store(tmp_path)
    (chat,) = store.recent()
    conversation = store.load(chat.id).conversation
    assert conversation.pending_call() is None
    assert [m.role for m in conversation.messages] == ["user", "assistant", "function"]


@pytest.fixture
def repl_env(cli_env):
    agent, ui, commands, ctx, provider = cli_env
    ctx.chats = ChatRecorder(ChatStore(agent.workspace))
    return agent, ui, commands, ctx, provider


def test_repl_autosaves_after_each_turn(repl_env):
    agent, ui, commands, ctx, provider = repl_env
    provider._turns = [text_turn("ответ 1"), KeyboardInterrupt(), text_turn("ответ 3")]
    read = _reader("первый", "второй", "третий", "/exit")
    assert run_repl(agent, ui, commands, read_input=read, chats=ctx.chats) == 0
    (chat,) = ctx.chats.store.recent()
    assert chat.id == ctx.chats.chat_id and chat.requests == 3 and chat.preview == "ответ 3"


def test_repl_save_error_warns_once(repl_env, capsys, monkeypatch):
    agent, ui, commands, ctx, provider = repl_env

    def boom(*a, **kw):
        raise OSError("диск только для чтения")

    monkeypatch.setattr(ctx.chats.store, "save", boom)
    read = _reader("раз", "два", "/exit")
    assert run_repl(agent, ui, commands, read_input=read, chats=ctx.chats) == 0
    assert _out(capsys).count("диск только для чтения") == 1


def test_clear_starts_new_chat_and_resume_returns(repl_env, capsys):
    agent, _, commands, ctx, provider = repl_env
    provider._turns = [text_turn("старый ответ"), text_turn("новый ответ")]
    agent.run_turn("старый вопрос")
    ctx.chats.save(agent.conversation)
    old_id = ctx.chats.chat_id
    commands.dispatch("/clear", ctx)
    assert ctx.chats.chat_id != old_id and len(agent.conversation) == 0

    commands.dispatch(f"/resume {old_id}", ctx)
    assert ctx.chats.chat_id == old_id
    out = _out(capsys)
    assert "продолжаем чат «старый вопрос»" in out and "старый ответ" in out
    agent.run_turn("ещё")
    assert _history(provider)[0] == ("user", "старый вопрос")


def test_resume_picker_and_errors(repl_env, capsys, monkeypatch):
    agent, ui, commands, ctx, _ = repl_env
    commands.dispatch("/resume", ctx)
    assert "сохранённых чатов пока нет" in _out(capsys)

    agent.conversation.add_user("сохранённый")
    ctx.chats.save(agent.conversation)
    saved_id = ctx.chats.chat_id
    commands.dispatch("/clear", ctx)
    calls = []

    def pick(chats, current_id=""):
        calls.append((chats, current_id))
        return None  # отмена — ничего не меняется

    monkeypatch.setattr(ui, "pick_chat", pick)
    commands.dispatch("/chats", ctx)  # алиас
    assert [c.id for c in calls[0][0]] == [saved_id] and calls[0][1] == ctx.chats.chat_id
    assert len(agent.conversation) == 0

    monkeypatch.setattr(ui, "pick_chat", lambda chats, current_id="": chats[0])
    commands.dispatch("/resume", ctx)
    assert agent.conversation.messages[0].content == "сохранённый"

    commands.dispatch("/resume nope", ctx)
    assert "не найден" in _out(capsys)
    ctx.chats = None
    commands.dispatch("/resume", ctx)
    assert "недоступны" in _out(capsys)


def test_repl_shows_resumed_chat_after_banner(repl_env, capsys):
    agent, ui, commands, ctx, _ = repl_env
    agent.conversation.add_user("прошлый вопрос")
    info = ctx.chats.store.save("20260101-000000-abcd", agent.conversation)
    run_repl(agent, ui, commands, read_input=_reader("/exit"), chats=ctx.chats, resumed=info)
    out = _out(capsys)
    assert out.index("модель") < out.index("продолжаем чат «прошлый вопрос»")


# ------------------------------ индекс проекта ------------------------------ #
def _index_complete(agent) -> bool:
    from devassist.project.index import ProjectIndex

    with ProjectIndex(agent.workspace) as ix:
        return ix.is_complete()


def test_repl_builds_index_before_first_input(cli_env, capsys):
    agent, ui, commands, _, _ = cli_env
    (agent.workspace.root / "mod.py").write_text("def f():\n    pass\n", encoding="utf-8")
    seen = []

    def read(**kwargs):
        seen.append(_index_complete(agent))  # к первому вводу индекс уже готов
        return "/exit"

    assert run_repl(agent, ui, commands, read_input=read) == 0
    assert seen == [True]
    out = _out(capsys)
    assert "индексирую проект…" in out and "индекс проекта: 1 файл, 1 определение" in out

    # индекс готов — при следующем запуске не строится заново
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "индексирую" not in _out(capsys)


def test_repl_without_index_on_start(cli_env):
    agent, ui, commands, _, _ = cli_env
    run_repl(agent, ui, commands, read_input=_reader("/exit"), index_on_start=False)
    assert not (agent.workspace.root / ".devassist").exists()


def test_cancelled_index_build_lets_user_continue(cli_env, capsys, monkeypatch):
    from devassist.project import index as index_mod

    agent, ui, commands, _, _ = cli_env
    for i in range(3):
        (agent.workspace.root / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
    real_refresh = index_mod.ProjectIndex.refresh

    def interrupted(self, base=None, *, progress=None):
        def stop(n):
            if n == 2:
                raise KeyboardInterrupt  # так приходит Esc/Ctrl+C
            progress(n)

        return real_refresh(self, base, progress=stop)

    monkeypatch.setattr(index_mod.ProjectIndex, "refresh", interrupted)
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "построение индекса прервано (просмотрено: 1 файл" in _out(capsys)
    assert not _index_complete(agent)

    # следующий запуск достраивает индекс
    monkeypatch.setattr(index_mod.ProjectIndex, "refresh", real_refresh)
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "индекс проекта: 3 файла" in _out(capsys)


def test_index_build_error_does_not_block_repl(cli_env, capsys, monkeypatch):
    from devassist.project import index as index_mod

    agent, ui, commands, _, _ = cli_env

    def broken(self):
        raise OSError("только чтение")

    monkeypatch.setattr(index_mod.ProjectIndex, "open", broken)
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "индекс проекта недоступен: только чтение" in _out(capsys)


def test_index_command(cli_env, capsys):
    agent, ui, commands, ctx, _ = cli_env
    (agent.workspace.root / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
    assert commands.dispatch("/index", ctx) is True
    out = _out(capsys)
    assert "индекс проекта: 1 файл, 1 определение" in out
    assert "языки: python 1" in out and "добавлено 1" in out
    commands.dispatch("/index", ctx)
    assert "изменений нет" in _out(capsys)
    commands.dispatch("/index rebuild", ctx)
    assert "добавлено 1" in _out(capsys)
    commands.dispatch("/index что-то", ctx)
    assert "использование: /index [rebuild]" in _out(capsys)
