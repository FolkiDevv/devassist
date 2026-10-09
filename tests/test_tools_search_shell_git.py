"""Тесты инструментов: поиск по содержимому, shell, git."""

from __future__ import annotations

import subprocess
import sys

import pytest

from devassist.tools.base import ToolError
from devassist.tools.fs import WriteFileTool
from devassist.tools.git import GitTool
from devassist.tools.search import SearchContentTool
from devassist.tools.shell import RunShellTool


def _write(ctx, path, content):
    w = WriteFileTool()
    w.run(w.parse({"path": path, "content": content}), ctx)


# ------------------------------- search ------------------------------- #
def test_search_content_finds_matches(ctx):
    _write(ctx, "a.py", "def foo():\n    return 42\n")
    _write(ctx, "b.py", "x = foo()\n")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": r"foo"}), ctx)
    assert "a.py:1:" in out.content
    assert "b.py:1:" in out.content


def test_search_glob_filter(ctx):
    _write(ctx, "a.py", "needle")
    _write(ctx, "a.txt", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle", "glob": "*.py"}), ctx)
    assert "a.py" in out.content and "a.txt" not in out.content


def test_search_skips_ignored_binary_large_and_secrets(ctx):
    _write(ctx, "src/a.py", "needle")
    _write(ctx, "node_modules/m.js", "needle")
    _write(ctx, ".env", "needle=secret")
    _write(ctx, ".env.example", "needle=")
    (ctx.root / "bin.dat").write_bytes(b"needle\0\1\2")
    (ctx.root / "big.txt").write_text("needle\n" * 200_000, encoding="utf-8")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    files = sorted({line.split(":", 1)[0] for line in out.splitlines()})
    assert files == [".env.example", "src/a.py"]


def test_search_glob_with_path(ctx):
    _write(ctx, "src/x/a.py", "needle")
    _write(ctx, "lib/b.py", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle", "glob": "src/**/*.py"}), ctx).content
    assert "src/x/a.py" in out and "lib/b.py" not in out


def test_search_max_results_is_clamped(ctx):
    _write(ctx, "a.txt", "hit\n" * 2000)
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "hit", "max_results": 10**6}), ctx).content
    assert len([ln for ln in out.splitlines() if ln.startswith("a.txt:")]) == 500


@pytest.mark.skipif(sys.platform == "win32", reason="симлинки")
def test_search_does_not_follow_symlinks_outside(ctx, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("needle", encoding="utf-8")
    (ctx.root / "link.txt").symlink_to(outside)
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    assert "link.txt" not in out


def test_search_invalid_regex(ctx):
    s = SearchContentTool()
    with pytest.raises(ToolError):
        s.run(s.parse({"pattern": "([unclosed"}), ctx)


# ------------------------------- shell -------------------------------- #
def test_shell_echo(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "echo hello"}), ctx)
    assert res.ok
    assert "hello" in res.content
    assert "exit code: 0" in res.content


def test_shell_nonzero_exit(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "exit 3"}), ctx)
    assert not res.ok
    assert "exit code: 3" in res.content


def test_shell_runs_in_project_root(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "pwd"}), ctx)
    assert str(ctx.root) in res.content


def test_shell_dangerous_classification(ctx):
    from devassist.security import RiskLevel

    sh = RunShellTool()
    safe = sh.parse({"command": "ls -la"})
    danger = sh.parse({"command": "rm -rf /tmp/x"})
    assert sh.risk(safe, ctx) == RiskLevel.WRITE
    assert sh.risk(danger, ctx) == RiskLevel.DANGEROUS


# -------------------------------- git --------------------------------- #
@pytest.fixture
def git_repo(ctx):
    root = ctx.root
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    return ctx


def test_git_status(git_repo):
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    g = GitTool()
    res = g.run(g.parse({"subcommand": "status"}), ctx)
    assert res.ok
    assert "f.txt" in res.content


def test_git_add_and_commit(git_repo):
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    g = GitTool()
    g.run(g.parse({"subcommand": "add", "args": ["f.txt"]}), ctx)
    res = g.run(g.parse({"subcommand": "commit", "args": ["-m", "init"]}), ctx)
    assert res.ok
    log = g.run(g.parse({"subcommand": "log"}), ctx)
    assert "init" in log.content


def test_git_disallowed_subcommand(git_repo):
    g = GitTool()
    with pytest.raises(ToolError):
        g.run(g.parse({"subcommand": "push"}), git_repo)


@pytest.mark.skipif(sys.platform == "win32", reason="FIFO")
def test_search_skips_fifo(ctx):
    import os

    os.mkfifo(ctx.root / "pipe")
    _write(ctx, "a.txt", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    assert "a.txt" in out
