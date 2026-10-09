"""Тесты слоя project: Workspace, обход файлов, glob, дерево, инструкции, промпт."""

from __future__ import annotations

import os

import pytest

from devassist.agent.prompts import build_system_prompt
from devassist.project.files import build_file_tree, glob_match, walk_files
from devassist.project.instructions import load_instructions
from devassist.project.workspace import Workspace


# ------------------------------ Workspace ------------------------------ #
def test_workspace_constructor_has_no_side_effects(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.data_dir == tmp_path.resolve() / ".devassist"
    assert ws.chats_dir == ws.data_dir / "chats"
    assert ws.index_dir == ws.data_dir / "index"
    assert not ws.data_dir.exists()


def test_ensure_data_dir_creates_gitignore_idempotently(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.ensure_data_dir() == ws.data_dir
    gi = ws.data_dir / ".gitignore"
    assert "*" in gi.read_text(encoding="utf-8").splitlines()
    gi.write_text("# мой\n*\n!keep\n", encoding="utf-8")
    ws.ensure_data_dir()  # повторно — без ошибок и без перезаписи
    assert gi.read_text(encoding="utf-8") == "# мой\n*\n!keep\n"


# ------------------------------ glob_match ------------------------------ #
@pytest.mark.parametrize(
    "path, pattern, expected",
    [
        ("a.py", "*.py", True),
        ("src/x/a.py", "*.py", True),  # без '/' — по имени файла
        ("src/a.py", "src/*.py", True),
        ("src/x/a.py", "src/*.py", False),  # '*' не пересекает '/'
        ("src/a.py", "src/**/*.py", True),  # '**/' — ноль и более каталогов
        ("src/x/y/a.py", "src/**/*.py", True),
        ("lib/a.py", "src/**/*.py", False),
        ("src/a.py", "./src/*.py", True),
        ("a1.txt", "a?.txt", True),
        ("ab.txt", "a[!b].txt", False),
        ("ac.txt", "a[!b].txt", True),
        ("x/README.md", "**/README.md", True),
        ("README.md", "**/README.md", True),
        ("a+b.txt", "a+b.txt", True),  # спецсимволы regex экранируются
    ],
)
def test_glob_match(path, pattern, expected):
    assert glob_match(path, pattern) is expected


# ------------------------------ walk / tree ------------------------------ #
def _make(root, rel, text="x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_walk_files_prunes_ignored_and_sorts(tmp_path):
    for rel in ("b.py", "a.py", "src/c.py", "node_modules/m.js", ".git/HEAD", "pkg.egg-info/x"):
        _make(tmp_path, rel)
    rels = [p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path)]
    assert rels == ["a.py", "b.py", "src/c.py"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_walk_files_skips_symlinks_outside_root(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    _make(outside, "secret.txt")
    _make(root, "in.txt")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "inner_link.txt").symlink_to(root / "in.txt")
    rels = sorted(p.name for p in walk_files(root))
    assert rels == ["in.txt", "inner_link.txt"]


def test_file_tree_shows_dotfiles_hides_service_dirs(tmp_path):
    _make(tmp_path, "src/main.py")
    _make(tmp_path, ".github/workflows/ci.yml")
    _make(tmp_path, "node_modules/junk.js")
    _make(tmp_path, ".git/HEAD")
    _make(tmp_path, ".devassist/chats/1.json")
    tree = build_file_tree(tmp_path)
    assert "src/" in tree and "main.py" in tree
    assert ".github/" in tree
    for hidden in ("node_modules", ".git/", ".devassist"):
        assert hidden not in tree


# ------------------------------ instructions ------------------------------ #
def test_load_instructions_order_and_truncation(tmp_path):
    (tmp_path / "DEVASSIST.local.md").write_text("личное", encoding="utf-8")
    (tmp_path / "DEVASSIST.md").write_text("x" * 50, encoding="utf-8")
    docs = load_instructions(tmp_path, max_chars=10)
    assert [d.name for d in docs] == ["DEVASSIST.md", "DEVASSIST.local.md"]
    assert docs[0].truncated and docs[0].content == "x" * 10
    assert docs[1].content == "личное" and not docs[1].truncated


def test_load_instructions_missing_and_empty(tmp_path):
    (tmp_path / "DEVASSIST.md").write_text("   \n", encoding="utf-8")
    assert load_instructions(tmp_path) == []


def test_system_prompt_contains_project_context(tmp_path):
    _make(tmp_path, "src/main.py")
    (tmp_path / "DEVASSIST.md").write_text("важная заметка", encoding="utf-8")
    prompt = build_system_prompt(Workspace(tmp_path))
    assert str(tmp_path.resolve()) in prompt
    assert "важная заметка" in prompt
    assert "Структура проекта" in prompt and "main.py" in prompt


def test_system_prompt_points_large_projects_to_index_tools(tmp_path):
    _make(tmp_path, "small.py")
    assert "find_symbol" in build_system_prompt(Workspace(tmp_path))  # в правилах работы
    assert "дерево неполное" not in build_system_prompt(Workspace(tmp_path))
    for i in range(250):
        _make(tmp_path, f"pkg/m{i:03}.py")
    assert "дерево неполное" in build_system_prompt(Workspace(tmp_path))
