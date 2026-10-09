"""Тесты извлечения символов для индекса проекта."""

from __future__ import annotations

import textwrap

import pytest

from devassist.project.symbols import MAX_SYMBOLS_PER_FILE, extract_symbols, language_of


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


def test_python_syntax_error_falls_back_to_regex():
    got = _names(
        """\
        class Broken:
            def method(self):
                return (
        def tail():
        """,
        "python",
    )
    assert got == [("class", "Broken"), ("function", "method"), ("function", "tail")]


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
def test_regex_languages(language, src, expected):
    assert _names(src, language) == expected


def test_unknown_language_has_no_symbols():
    assert extract_symbols("def x(): pass", None) == []
    assert extract_symbols("{}", "json") == []


def test_symbols_limit_per_file():
    src = "\n".join(f"def f{i}(): pass" for i in range(MAX_SYMBOLS_PER_FILE + 50))
    assert len(extract_symbols(src, "python")) == MAX_SYMBOLS_PER_FILE
