"""Извлечение определений (символов) из исходников для индекса проекта.

Python разбирается модулем :mod:`ast` (точно: вложенность, диапазон строк);
остальные языки — построчными регулярными выражениями (быстро, без
зависимостей, но эвристично: диапазон строк неизвестен, вложенность почти не
отслеживается). Новый язык — расширения в :data:`EXTENSIONS` и правила в
:data:`_RULES`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePath

MAX_SYMBOLS_PER_FILE = 2000
_MAX_SIGNATURE = 200
_MAX_SIGNATURE_LINES = 6

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".sh": "shell",
    ".bash": "shell",
    ".md": "markdown",
    ".markdown": "markdown",
    # без символов, но полезны для статистики индекса
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".sql": "sql",
    ".txt": "text",
    ".rst": "text",
}
FILENAMES: dict[str, str] = {
    "Makefile": "make",
    "Dockerfile": "docker",
    "CMakeLists.txt": "cmake",
}


@dataclass(frozen=True)
class Symbol:
    """Определение в файле. ``end_line`` = None — конец неизвестен (эвристика)."""

    name: str
    kind: str  # class, function, method, constant, interface, struct, section, ...
    line: int
    end_line: int | None = None
    parent: str = ""  # объемлющее определение (``Class`` для метода)
    depth: int = 0  # уровень вложенности (для оглавления)
    signature: str = ""

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


def language_of(path: str | PurePath) -> str | None:
    """Язык файла по имени/расширению (None — неизвестен)."""
    p = PurePath(path)
    return FILENAMES.get(p.name) or EXTENSIONS.get(p.suffix.lower())


def extract_symbols(text: str, language: str | None) -> list[Symbol]:
    """Символы файла в порядке появления (не больше :data:`MAX_SYMBOLS_PER_FILE`)."""
    if language == "python":
        try:
            symbols = _python_symbols(text)
        except (SyntaxError, ValueError, RecursionError):
            symbols = _regex_symbols(text, _RULES["python"])
    elif language == "markdown":
        symbols = _markdown_symbols(text)
    elif language in _RULES:
        symbols = _regex_symbols(text, _RULES[language])
    else:
        return []
    return symbols[:MAX_SYMBOLS_PER_FILE]


def _clip(signature: str) -> str:
    signature = " ".join(signature.split())
    if len(signature) > _MAX_SIGNATURE:
        signature = signature[: _MAX_SIGNATURE - 1] + "…"
    return signature


# --------------------------------- Python --------------------------------- #
def _python_symbols(text: str) -> list[Symbol]:
    tree = ast.parse(text)
    lines = text.splitlines()
    out: list[Symbol] = []

    def signature(node: ast.AST) -> str:
        start = node.lineno
        body = getattr(node, "body", None)
        last = body[0].lineno - 1 if body else start
        last = min(max(last, start), start + _MAX_SIGNATURE_LINES - 1)
        sig = " ".join(lines[i - 1].strip() for i in range(start, last + 1) if i <= len(lines))
        return _clip(sig)

    def visit(body: list[ast.stmt], parent: str, depth: int, in_class: bool) -> None:
        for node in body:
            if len(out) >= MAX_SYMBOLS_PER_FILE:
                return
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if in_class else "function"
            elif isinstance(node, (ast.If, ast.Try)):
                # определения под `if TYPE_CHECKING:` / `try: import ...` — того же уровня
                for block in (node.body, node.orelse, getattr(node, "finalbody", [])):
                    visit(block, parent, depth, in_class)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body, parent, depth, in_class)
                continue
            elif depth == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 1:
                        out.append(
                            Symbol(
                                name=t.id,
                                kind="constant",
                                line=node.lineno,
                                end_line=node.end_lineno,
                                signature=_clip(lines[node.lineno - 1]),
                            )
                        )
                continue
            else:
                continue
            out.append(
                Symbol(
                    name=node.name,
                    kind=kind,
                    line=node.lineno,
                    end_line=node.end_lineno,
                    parent=parent,
                    depth=depth,
                    signature=signature(node),
                )
            )
            if kind == "class":  # вложенные функции — шум; методы и вложенные классы — нет
                qual = f"{parent}.{node.name}" if parent else node.name
                visit(node.body, qual, depth + 1, True)

    visit(tree.body, "", 0, False)
    return out


# -------------------------------- Markdown -------------------------------- #
# CommonMark: до 3 пробелов отступа; закрывающие '#' — только после пробела ("# C#").
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _markdown_symbols(text: str) -> list[Symbol]:
    out: list[Symbol] = []
    stack: list[tuple[int, str]] = []  # (уровень, заголовок)
    fence = ""  # открывающая ограда блока кода ("```", "~~~~"…), пусто — вне блока
    for i, line in enumerate(text.splitlines(), start=1):
        m = _FENCE_RE.match(line)
        if fence:
            # закрывает только ограда того же символа, не короче, без текста после
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                if not line[m.end() :].strip():
                    fence = ""
            continue
        if m:
            fence = m.group(1)
            continue
        m = _HEADING_RE.match(line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        out.append(
            Symbol(
                name=title[:_MAX_SIGNATURE],
                kind="section",
                line=i,
                parent=stack[-1][1] if stack else "",
                depth=len(stack),
                signature=_clip(line),
            )
        )
        stack.append((level, title))
        if len(out) >= MAX_SYMBOLS_PER_FILE:
            break
    return out


# ------------------------------ Регэкспы ------------------------------ #
# Группы: name — имя; kind — вид (если не задан правилом); parent — владелец.
_Rule = tuple[str, "re.Pattern[str]"]
_NOT_NAMES = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "else", "new", "sizeof", "do", "with"}
)
_ID = r"[A-Za-z_$][\w$]*"


def _rules(*pairs: tuple[str, str]) -> list[_Rule]:
    return [(kind, re.compile(pattern)) for kind, pattern in pairs]


_JS_RULES = _rules(
    (
        "function",
        rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>{_ID})",
    ),
    ("class", rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>{_ID})"),
    ("interface", rf"^\s*(?:export\s+)?(?:declare\s+)?interface\s+(?P<name>{_ID})"),
    ("type", rf"^\s*(?:export\s+)?(?:declare\s+)?type\s+(?P<name>{_ID})\s*(?:<[^=]*>)?\s*="),
    ("enum", rf"^\s*(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+(?P<name>{_ID})"),
    (
        "function",
        rf"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>{_ID})\s*(?::[^=]+)?=\s*"
        rf"(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|{_ID}\s*=>)",
    ),
    (
        "method",
        r"^\s+(?:(?:public|private|protected|static|readonly|async|override|get|set)\s+)*"
        rf"(?P<name>{_ID})\s*\([^)]*\)\s*(?::\s*[^{{]+)?\{{\s*$",
    ),
)
_TYPES_JVM = (
    r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|internal|static|final|"
    r"abstract|sealed|open|data|partial|inline|value|annotation|case|readonly)\s+)*"
    r"(?P<kind>class|interface|enum|record|struct|object|trait|protocol)\s+(?P<name>[A-Za-z_]\w*)"
)
_JAVA_METHOD = (
    r"^\s+(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|"
    r"default|override|virtual|async|sealed)\s+)+[\w<>\[\],.?\s]*?\s(?P<name>[A-Za-z_]\w*)\s*\("
)

_RULES: dict[str, list[_Rule]] = {
    "python": _rules(
        ("class", r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_]\w*)"),
        ("function", r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)"),
    ),
    "javascript": _JS_RULES,
    "typescript": _JS_RULES,
    "go": _rules(
        (
            "method",
            r"^func\s+\(\s*\w*\s*\*?\s*(?P<parent>[A-Za-z_]\w*)[^)]*\)\s*(?P<name>[A-Za-z_]\w*)",
        ),
        ("function", r"^func\s+(?P<name>[A-Za-z_]\w*)"),
        ("type", r"^type\s+(?P<name>[A-Za-z_]\w*)\s+(?P<kind>struct|interface)\b"),
        ("type", r"^type\s+(?P<name>[A-Za-z_]\w*)"),
    ),
    "rust": _rules(
        (
            "function",
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
            r'(?:extern\s+"[^"]*"\s+)?fn\s+(?P<name>[A-Za-z_]\w*)',
        ),
        (
            "type",
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?P<kind>struct|enum|trait|union|mod|type)\s+"
            r"(?P<name>[A-Za-z_]\w*)",
        ),
        (
            "impl",
            r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?(?P<name>[A-Za-z_][\w:]*)",
        ),
        ("macro", r"^\s*macro_rules!\s*(?P<name>[A-Za-z_]\w*)"),
    ),
    "java": _rules(("class", _TYPES_JVM), ("method", _JAVA_METHOD)),
    "csharp": _rules(("class", _TYPES_JVM), ("method", _JAVA_METHOD)),
    "kotlin": _rules(
        ("class", _TYPES_JVM),
        (
            "function",
            r"^\s*(?:\w+\s+)*fun\s+(?:<[^>]*>\s*)?(?:[\w.<>?, ]+\.)?(?P<name>[A-Za-z_]\w*)",
        ),
    ),
    "scala": _rules(("class", _TYPES_JVM), ("function", r"^\s*(?:\w+\s+)*def\s+(?P<name>\w+)")),
    "swift": _rules(("class", _TYPES_JVM), ("function", r"^\s*(?:\w+\s+)*func\s+(?P<name>\w+)")),
    "ruby": _rules(
        ("class", r"^\s*(?P<kind>class|module)\s+(?P<name>[A-Z][\w:]*)"),
        ("method", r"^\s*def\s+(?:self\.)?(?P<name>[\w?!=]+)"),
    ),
    "php": _rules(
        (
            "class",
            r"^\s*(?:(?:abstract|final|readonly)\s+)*(?P<kind>class|interface|trait|enum)\s+"
            r"(?P<name>\w+)",
        ),
        (
            "function",
            r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+&?\s*"
            r"(?P<name>\w+)",
        ),
    ),
    "c": _rules(
        ("type", r"^\s*(?:typedef\s+)?(?P<kind>struct|union|enum)\s+(?P<name>[A-Za-z_]\w*)\s*\{"),
        (
            "function",
            r"^(?!\s)(?!(?:return|else|if|while|for|switch|do)\b)[\w\s\*]*?\b"
            r"(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$",
        ),
    ),
    "cpp": _rules(
        (
            "type",
            r"^\s*(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?(?P<kind>class|struct|union|enum)"
            r"(?:\s+class)?\s+(?P<name>[A-Za-z_]\w*)\s*(?::[^{;]*)?\{?\s*$",
        ),
        ("namespace", r"^\s*namespace\s+(?P<name>[A-Za-z_][\w:]*)"),
        (
            "function",
            r"^(?!\s)(?!(?:return|else|if|while|for|switch|do)\b)[\w\s\*&:<>,~]*?\b"
            r"(?P<name>[A-Za-z_~][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?"
            r"(?:override\s*)?\{?\s*$",
        ),
    ),
    "shell": _rules(
        ("function", r"^\s*(?:function\s+)?(?P<name>[A-Za-z_][\w-]*)\s*\(\)\s*\{?"),
        ("function", r"^\s*function\s+(?P<name>[A-Za-z_][\w-]*)"),
    ),
}


def _regex_symbols(text: str, rules: list[_Rule]) -> list[Symbol]:
    out: list[Symbol] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or len(line) > 1000:
            continue
        for kind, regex in rules:
            m = regex.match(line)
            if not m:
                continue
            name = m.group("name")
            if name in _NOT_NAMES:
                break
            groups = m.groupdict()
            indent = groups.get("indent")
            out.append(
                Symbol(
                    name=name,
                    kind=(groups.get("kind") or kind).lower(),
                    line=i,
                    parent=groups.get("parent") or "",
                    depth=1 if indent else 0,
                    signature=_clip(line),
                )
            )
            break
        if len(out) >= MAX_SYMBOLS_PER_FILE:
            break
    return out
