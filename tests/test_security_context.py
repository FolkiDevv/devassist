"""Тесты песочницы, классификации риска и контекста проекта."""

from __future__ import annotations

import pytest

from devassist.devassist.context import build_file_tree, build_project_context, read_memory
from devassist.devassist.security import (
    RiskLevel,
    SandboxError,
    classify_shell_command,
    resolve_in_root,
)


def test_resolve_in_root_ok(tmp_path):
    (tmp_path / "sub").mkdir()
    p = resolve_in_root(tmp_path, "sub/x.txt")
    assert str(p).startswith(str(tmp_path))


def test_resolve_in_root_escape(tmp_path):
    with pytest.raises(SandboxError):
        resolve_in_root(tmp_path, "../../etc/passwd")
    with pytest.raises(SandboxError):
        resolve_in_root(tmp_path, "/etc/passwd")


def test_classify_shell():
    assert classify_shell_command("ls") == RiskLevel.WRITE
    assert classify_shell_command("rm -rf foo") == RiskLevel.DANGEROUS
    assert classify_shell_command("sudo apt update") == RiskLevel.DANGEROUS
    assert classify_shell_command("curl http://x") == RiskLevel.DANGEROUS
    assert classify_shell_command("git reset --hard") == RiskLevel.DANGEROUS


def test_file_tree_and_memory(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x")
    (tmp_path / "DEVASSIST.md").write_text("важная заметка")

    tree = build_file_tree(tmp_path)
    assert "src/" in tree and "main.py" in tree
    assert "node_modules" not in tree  # игнорируется

    assert read_memory(tmp_path) == "важная заметка"
    ctx = build_project_context(tmp_path)
    assert "важная заметка" in ctx
    assert "Структура проекта" in ctx
