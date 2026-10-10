"""Тесты извлечения символов для индекса проекта."""

from __future__ import annotations

import textwrap

import pytest

from devassist.project.symbols import (
    _RULES,
    MAX_SYMBOLS_PER_FILE,
    _regex_symbols,
    extract,
    extract_symbols,
    language_of,
    module_parts,
)


def _names(text: str, language: str) -> list[tuple[str, str]]:
    return [(s.kind, s.qualname) for s in extract_symbols(textwrap.dedent(text), language)]


@pytest.mark.parametrize(
    "path, expected",
    [
        ("a.py", "python"),
        ("src/x.TS", "typescript"),
        ("Makefile", "make"),
        ("README.md", "markdown"),
        ("data.bin", None),
        ("noext", None),
    ],
)
def test_language_of(path, expected):
    assert language_of(path) == expected


def test_python_symbols_with_nesting_and_ranges():
    src = textwrap.dedent(
        '''\
        """Модуль."""
        import os

        MAX_SIZE = 10
        name = "не константа"


        class Agent(Base):
            """Документация."""

            def run(self, x: int) -> int:
                def inner():
                    pass
                return x

            async def stream(
                self,
                prompt: str,
            ) -> None:
                pass

            class Config:
                pass


        async def main():
            pass


        if True:
            def conditional():
                pass
        '''
    )
    symbols = extract_symbols(src, "python")
    got = [(s.kind, s.qualname, s.line, s.end_line, s.depth) for s in symbols]
    assert got == [
        ("constant", "MAX_SIZE", 4, 4, 0),
        ("class", "Agent", 8, 23, 0),
        ("method", "Agent.run", 11, 14, 1),
        ("method", "Agent.stream", 16, 20, 1),
        ("class", "Agent.Config", 22, 23, 1),
        ("function", "main", 26, 27, 0),
        ("function", "conditional", 31, 32, 0),
    ]
    stream = symbols[3]
    assert stream.signature == "async def stream( self, prompt: str, ) -> None:"
    assert symbols[1].signature == "class Agent(Base):"


def test_python_syntax_error_falls_back_to_tree_sitter():
    symbols = extract_symbols(
        textwrap.dedent(
            """\
            class Broken:
                def method(self):
                    return (
            def tail():
                pass
            """
        ),
        "python",
    )
    assert [(s.kind, s.qualname) for s in symbols][:2] == [
        ("class", "Broken"),
        ("method", "Broken.method"),
    ]


def test_python_without_grammar_falls_back_to_regex(monkeypatch):
    from devassist.project import treesitter

    monkeypatch.setattr(treesitter, "extract", lambda text, language, path="": None)
    got = _names("class Broken:\n    def method(self):\n        return (\n", "python")
    assert got == [("class", "Broken"), ("function", "method")]


def test_markdown_headings_with_parents_skip_code_fences():
    symbols = extract_symbols(
        "# Заголовок\n\n## Раздел 1\n```\n# не заголовок\n```\n### Подраздел\n## Раздел 2 ##\n",
        "markdown",
    )
    got = [(s.name, s.parent, s.depth, s.line) for s in symbols]
    assert got == [
        ("Заголовок", "", 0, 1),
        ("Раздел 1", "Заголовок", 1, 3),
        ("Подраздел", "Раздел 1", 2, 7),
        ("Раздел 2", "Заголовок", 1, 8),
    ]


@pytest.mark.parametrize(
    "language, src, expected",
    [
        (
            "typescript",
            """\
            export default class Store {
              private async load(id: string): Promise<void> {
                if (x) {
                }
              }
            }
            export interface Props {}
            export type Id = string;
            export const enum Color { Red }
            export const handler = async (req: Req) => {};
            function* gen() {}
            """,
            [
                ("class", "Store"),
                ("method", "load"),
                ("interface", "Props"),
                ("type", "Id"),
                ("enum", "Color"),
                ("function", "handler"),
                ("function", "gen"),
            ],
        ),
        (
            "go",
            """\
            type Server struct {
            type Handler interface {
            type ID string
            func (s *Server) Serve(addr string) error {
            func main() {
            """,
            [
                ("struct", "Server"),
                ("interface", "Handler"),
                ("type", "ID"),
                ("method", "Server.Serve"),
                ("function", "main"),
            ],
        ),
        (
            "rust",
            """\
            pub struct Index {
            enum Kind {
            pub(crate) trait Search {
            impl Search for Index {
                pub async fn find(&self) -> Vec<Hit> {
            macro_rules! hit {
            """,
            [
                ("struct", "Index"),
                ("enum", "Kind"),
                ("trait", "Search"),
                ("impl", "Index"),
                ("function", "find"),
                ("macro", "hit"),
            ],
        ),
        (
            "java",
            """\
            public final class Service {
                @Override
                public List<String> names(int n) {
                private static void helper() {
            public interface Repo {
            record Point(int x, int y) {
            """,
            [
                ("class", "Service"),
                ("method", "names"),
                ("method", "helper"),
                ("interface", "Repo"),
                ("record", "Point"),
            ],
        ),
        (
            "ruby",
            """\
            module Billing
              class Invoice
                def self.build
                def paid?
            """,
            [
                ("module", "Billing"),
                ("class", "Invoice"),
                ("method", "build"),
                ("method", "paid?"),
            ],
        ),
        (
            "php",
            """\
            final class Controller {
                public static function index() {
            function helper() {
            """,
            [("class", "Controller"), ("function", "index"), ("function", "helper")],
        ),
        (
            "c",
            """\
            struct node {
            static int parse(const char *s)
            {
                if (x) {
                return foo(1);
            int main(void) {
            int declared(void);
            """,
            [("struct", "node"), ("function", "parse"), ("function", "main")],
        ),
        (
            "kotlin",
            """\
            data class User(val id: Int)
            fun <T> List<T>.second(): T = this[1]
            """,
            [("class", "User"), ("function", "second")],
        ),
        (
            "shell",
            """\
            deploy() {
            function cleanup {
            """,
            [("function", "deploy"), ("function", "cleanup")],
        ),
    ],
)
def test_regex_fallback_languages(language, src, expected):
    """Запасной путь без грамматики: построчные шаблоны (здесь — на обрывках кода)."""
    symbols = _regex_symbols(textwrap.dedent(src), _RULES[language])
    assert [(s.kind, s.qualname) for s in symbols] == expected


def test_unknown_language_has_no_symbols():
    assert extract_symbols("def x(): pass", None) == []
    assert extract_symbols("{}", "json") == []


def test_symbols_limit_per_file():
    src = "\n".join(f"def f{i}(): pass" for i in range(MAX_SYMBOLS_PER_FILE + 50))
    assert len(extract_symbols(src, "python")) == MAX_SYMBOLS_PER_FILE


def test_markdown_commonmark_headings_and_fences():
    text = (
        " # С отступом\n"
        "# C#\n"
        "#хештег не заголовок\n"
        "~~~~\n"
        "```\n"
        "# внутри блока\n"
        "~~~~\n"
        "# После блока\n"
        "```python\n"
        "# комментарий\n"
        "``` не закрывает\n"
        "```\n"
        "## Конец ##\n"
    )
    names = [s.name for s in extract_symbols(text, "markdown")]
    assert names == ["С отступом", "C#", "После блока", "Конец"]


def test_python_methods_under_class_level_if_and_try():
    src = textwrap.dedent(
        """\
        class C:
            if True:
                def run(self):
                    pass
            try:
                def fast(self):
                    pass
            except ImportError:
                def slow(self):
                    pass
        """
    )
    got = [(s.kind, s.qualname, s.depth) for s in extract_symbols(src, "python")]
    assert got == [
        ("class", "C", 0),
        ("method", "C.run", 1),
        ("method", "C.fast", 1),
        ("method", "C.slow", 1),
    ]


def test_python_try_blocks_in_source_order():
    src = textwrap.dedent(
        """\
        try:
            def a(): pass
        except ImportError:
            def b(): pass
        else:
            def c(): pass
        finally:
            def d(): pass
        """
    )
    assert [s.name for s in extract_symbols(src, "python")] == ["a", "b", "c", "d"]


# ------------------------- использования и импорты ------------------------- #
def _facts(src: str, path: str = "pkg/mod.py"):
    return extract(textwrap.dedent(src), "python", path)


def test_python_refs_scopes_kinds_and_locals():
    facts = _facts(
        """\
        LIMIT = 5


        class Agent(Base):
            def run(self, text, n=LIMIT):
                def inner(y):
                    return helper(y) + text
                local = Thing()
                local.go()
                self.index.refresh(text)
                print(len(text))
                return [x for x in items if x]


        def helper(value):
            return Agent().run(value)
        """
    )
    got = [(r.name, r.kind, r.line, r.scope) for r in facts.refs]
    assert got == [
        ("Base", "name", 4, "Agent"),
        ("LIMIT", "name", 5, "Agent.run"),
        ("helper", "call", 7, "Agent.run"),  # вложенная функция — в области внешней
        ("Thing", "call", 8, "Agent.run"),
        ("go", "call", 9, "Agent.run"),
        ("refresh", "call", 10, "Agent.run"),
        ("index", "attr", 10, "Agent.run"),
        ("items", "name", 12, "Agent.run"),
        ("run", "call", 16, "helper"),
        ("Agent", "call", 16, "helper"),
    ]
    # параметры, локальные переменные, переменные включений, self и builtins — не использования
    names = {r.name for r in facts.refs}
    assert not names & {"text", "y", "local", "x", "value", "self", "print", "len", "inner"}


def test_python_ref_columns_are_characters():
    facts = _facts("ё = 'ж'; obj.метод()\n")
    (ref,) = [r for r in facts.refs if r.name == "метод"]
    assert (ref.kind, ref.col) == ("call", 13)  # в символах: в байтах было бы 16


def test_python_imports_absolute_relative_and_aliases():
    facts = _facts(
        """\
        import os.path as osp
        import json, a.b.c
        from . import sibling
        from .core import Engine as E
        from ..up import thing
        from ...top import y
        from ....beyond import x
        """,
        path="pkg/sub/mod.py",
    )
    got = [(i.module, i.name, i.alias) for i in facts.imports]
    assert got == [
        ("os.path", "", "osp"),
        ("json", "", "json"),
        ("a.b.c", "", "a"),
        ("pkg.sub", "sibling", "sibling"),
        ("pkg.sub.core", "Engine", "E"),
        ("pkg.up", "thing", "thing"),
        ("top", "y", "y"),  # модуль в корне проекта
        ("....beyond", "x", "x"),  # выше корня проекта — не разрешается
    ]
    init = _facts("from . import a\n", path="pkg/__init__.py")
    assert [(i.module, i.name) for i in init.imports] == [("pkg", "a")]


def test_python_docstring_and_name_column():
    symbols = extract_symbols(
        textwrap.dedent(
            '''\
            class Ёж:
                """

                Первая строка.
                Вторая.
                """

                async def  бег(self):
                    """Очень длинная документация """ + "x" * 0
            '''
        ),
        "python",
    )
    by_name = {s.name: s for s in symbols}
    assert (by_name["Ёж"].doc, by_name["Ёж"].col) == ("Первая строка.", 6)
    assert (by_name["бег"].doc, by_name["бег"].col) == ("", 15)  # не литерал — не докстринг


def test_python_syntax_error_keeps_calls_but_not_imports():
    facts = _facts("import os\n\ndef ok():\n    call_me()\n\ndef broken(:\n")
    assert [s.name for s in facts.symbols] == ["ok", "broken"]
    assert [(r.name, r.kind, r.scope) for r in facts.refs] == [("call_me", "call", "ok")]
    assert facts.imports == []  # импорты — только из ast


def test_regex_symbols_have_name_column():
    (sym,) = extract_symbols("export function runTurn() {}\n", "typescript")
    assert sym.col == len("export function ")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("a/b/c.py", ["a", "b", "c"]),
        ("a/b/__init__.py", ["a", "b"]),
        ("stubs/x.pyi", ["stubs", "x"]),
        ("top.py", ["top"]),
        ("README.md", None),
    ],
)
def test_module_parts(path, expected):
    assert module_parts(path) == expected


def test_python_signature_skips_comment_lines_before_body():
    src = "class Conversation:\n    # пояснение к полям\n    # ещё строка\n    items: list = []\n"
    (sym,) = extract_symbols(src, "python")
    assert sym.signature == "class Conversation:"
