"""Тесты загрузки конфигурации: приоритет источников, отсутствие побочных эффектов."""

from __future__ import annotations

import os

import pytest

from devassist.cli import main
from devassist.config import Config, ConfigError, load_environment, read_dotenv


def _env_file(path, text):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text(text, encoding="utf-8")


def test_read_dotenv_parses_quotes_comments_export(tmp_path):
    (tmp_path / ".env").write_text(
        "# comment\nA=1\nexport B=\"two\"\nC='three'\nbroken line\n", encoding="utf-8"
    )
    assert read_dotenv(tmp_path / ".env") == {"A": "1", "B": "two", "C": "three"}
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_priority_environ_over_root_over_cwd(tmp_path):
    root, cwd = tmp_path / "proj", tmp_path / "cwd"
    _env_file(root, "GIGACHAT_MODEL=from-root\nGIGACHAT_SCOPE=root-scope\n")
    _env_file(cwd, "GIGACHAT_MODEL=from-cwd\nGIGACHAT_SCOPE=cwd-scope\nGIGACHAT_TIMEOUT=7\n")

    env = load_environment(root, environ={"GIGACHAT_SCOPE": "env-scope"}, cwd=cwd)
    assert env["GIGACHAT_MODEL"] == "from-root"  # root .env важнее cwd .env
    assert env["GIGACHAT_SCOPE"] == "env-scope"  # окружение важнее файлов
    assert env["GIGACHAT_TIMEOUT"] == "7"  # cwd .env — последний источник

    cfg = Config.load(project_root=root, environ={}, cwd=cwd, model="from-cli")
    assert cfg.model == "from-cli"  # аргумент CLI важнее всего
    assert cfg.timeout == 7


def test_load_does_not_touch_os_environ(tmp_path, monkeypatch):
    monkeypatch.delenv("GIGACHAT_ACCESS_KEY", raising=False)
    _env_file(tmp_path, "GIGACHAT_ACCESS_KEY=secret-from-dotenv\n")
    cfg = Config.load(project_root=tmp_path, environ={}, cwd=tmp_path)
    assert cfg.access_key == "secret-from-dotenv"
    assert "GIGACHAT_ACCESS_KEY" not in os.environ


@pytest.mark.parametrize(
    "env",
    [
        {"GIGACHAT_TIMEOUT": "abc"},
        {"GIGACHAT_TIMEOUT": "0"},
        {"DEVASSIST_TEMPERATURE": "warm"},
        {"DEVASSIST_TEMPERATURE": "5"},
        {"DEVASSIST_MODE": "yolo"},
    ],
)
def test_bad_values_raise_config_error(tmp_path, env):
    with pytest.raises(ConfigError):
        Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)


def test_main_reports_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACHAT_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACHAT_TIMEOUT", "abc")
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "GIGACHAT_TIMEOUT" in capsys.readouterr().out


def test_main_without_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in ("GIGACHAT_ACCESS_KEY", "GIGACHAT_CERT", "GIGACHAT_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "реквизиты" in capsys.readouterr().out


def test_save_chats_flag_and_env(tmp_path):
    def load(env, **kw):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path, **kw).save_chats

    assert load({}) is True
    assert load({"DEVASSIST_SAVE_CHATS": "0"}) is False
    assert load({"DEVASSIST_SAVE_CHATS": "1"}, save_chats=False) is False  # --no-save сильнее


def test_mode_flag_and_env(tmp_path):
    from devassist.permissions import PermissionMode

    def load(env, **kw):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path, **kw).mode

    assert load({}) is PermissionMode.MANUAL
    assert load({"DEVASSIST_MODE": " Plan "}) is PermissionMode.PLAN
    # --mode сильнее окружения
    assert load({"DEVASSIST_MODE": "plan"}, mode=PermissionMode.ACCEPT_EDITS) is (
        PermissionMode.ACCEPT_EDITS
    )
    with pytest.raises(ConfigError, match="DEVASSIST_MODE"):
        load({"DEVASSIST_MODE": "auto"})
