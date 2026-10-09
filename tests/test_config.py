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
        {"DEVASSIST_COMPACT_THRESHOLD": "5"},
        {"DEVASSIST_COMPACT_THRESHOLD": "99"},
        {"DEVASSIST_COMPACT_THRESHOLD": "0.8"},
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


def test_compaction_settings(tmp_path):
    def load(env):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    cfg = load({})
    assert cfg.auto_compact is True and cfg.compact_threshold == 0.8
    cfg = load({"DEVASSIST_AUTO_COMPACT": "0", "DEVASSIST_COMPACT_THRESHOLD": "65"})
    assert cfg.auto_compact is False and cfg.compact_threshold == 0.65


def test_yes_all_implies_auto_approve(tmp_path):
    cfg = Config.load(project_root=tmp_path, yes_all=True, environ={}, cwd=tmp_path)
    assert cfg.auto_approve and cfg.yes_all
    cfg = Config.load(project_root=tmp_path, auto_approve=True, environ={}, cwd=tmp_path)
    assert cfg.auto_approve and not cfg.yes_all


def test_ca_bundle_enables_verification(tmp_path):
    import ssl

    import certifi

    env = {"GIGACHAT_ACCESS_KEY": "k", "GIGACHAT_CA_BUNDLE": certifi.where()}
    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert cfg.verify_ssl and cfg.ca_bundle == certifi.where()
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED
    # явный GIGACHAT_VERIFY_SSL=0 важнее
    cfg = Config.load(
        project_root=tmp_path, environ={**env, "GIGACHAT_VERIFY_SSL": "0"}, cwd=tmp_path
    )
    assert cfg.build_ssl_verify() is False
    with pytest.raises(ConfigError, match="GIGACHAT_CA_BUNDLE"):
        Config.load(
            project_root=tmp_path,
            environ={"GIGACHAT_CA_BUNDLE": str(tmp_path / "нет.pem")},
            cwd=tmp_path,
        )


def _self_signed(tmp_path):
    import shutil
    import subprocess

    if shutil.which("openssl") is None:
        pytest.skip("нет openssl")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=test",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def test_mtls_verification_loads_trust_anchors(tmp_path):
    """Голый SSLContext без корневых сертификатов не проверил бы ни один сервер."""
    import ssl

    cert, key = _self_signed(tmp_path)
    base = {"GIGACHAT_CERT": str(cert), "GIGACHAT_KEY": str(key)}
    cfg = Config.load(
        project_root=tmp_path, environ={**base, "GIGACHAT_VERIFY_SSL": "1"}, cwd=tmp_path
    )
    context = cfg.build_ssl_verify()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.cert_store_stats()["x509_ca"] > 0 or context.get_ca_certs() != []
    cfg = Config.load(
        project_root=tmp_path, environ={**base, "GIGACHAT_CA_BUNDLE": str(cert)}, cwd=tmp_path
    )
    context = cfg.build_ssl_verify()
    assert context.verify_mode == ssl.CERT_REQUIRED and len(context.get_ca_certs()) == 1
