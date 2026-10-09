"""Базовые абстракции для инструментов агента.

Каждый инструмент:
  * объявляет имя, описание и pydantic-модель параметров (Params);
  * из модели автоматически строится JSON-schema для function calling;
  * объявляет уровень риска (для модели прав);
  * умеет (опционально) показать превью (например, дифф) перед выполнением;
  * выполняется методом ``run`` и возвращает ``ToolResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from devassist.errors import ToolError
from devassist.llm.types import ToolSpec
from devassist.project.workspace import Workspace
from devassist.security import RiskLevel

__all__ = [
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
]

P = TypeVar("P", bound=BaseModel)


@dataclass(frozen=True)
class ToolContext:
    """Контекст выполнения, доступный инструменту.

    Намеренно не содержит Config: инструментам не нужны (и не должны быть
    доступны) реквизиты API.
    """

    workspace: Workspace

    @property
    def root(self) -> Path:
        return self.workspace.root


@dataclass
class ToolResult:
    """Результат выполнения инструмента."""

    content: str  # текст, который вернётся модели
    ok: bool = True
    summary: str = ""  # краткая строка для UI
    display: str | None = None  # доп. вывод для пользователя (дифф/листинг)

    def as_function_content(self) -> str:
        if self.ok:
            return self.content
        return f"ОШИБКА: {self.content}"


def _normalize_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Приводит описание свойства к формату, который принимает GigaChat.

    GigaChat не понимает union-типы (``anyOf``/``oneOf``), которые pydantic
    генерирует для ``Optional[...]``. Сворачиваем их в одиночный ``type``,
    отбрасывая ветку ``null`` (необязательность отражается отсутствием в
    ``required``). Также убираем служебные ключи (title/default).
    """
    prop = dict(prop)
    prop.pop("title", None)
    prop.pop("default", None)

    variants = prop.pop("anyOf", None) or prop.pop("oneOf", None)
    if variants:
        non_null = [v for v in variants if v.get("type") != "null"]
        chosen = non_null[0] if non_null else variants[0]
        merged = {k: v for k, v in prop.items()}
        merged.update(chosen)
        merged.pop("title", None)
        prop = merged

    # Рекурсивно нормализуем элементы массива
    if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
        prop["items"] = _normalize_property(prop["items"])
    return prop


def _clean_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON-schema параметров в формате function calling GigaChat."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    props = schema.get("properties", {})
    schema["properties"] = {k: _normalize_property(v) for k, v in props.items()}
    schema.setdefault("type", "object")
    return schema


class Tool(ABC, Generic[P]):
    name: str = ""
    description: str = ""
    Params: type[BaseModel] = BaseModel

    # ------------------------------------------------------------------ #
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=_clean_schema(self.Params),
        )

    def parse(self, arguments: dict[str, Any]) -> BaseModel:
        return self.Params.model_validate(arguments or {})

    def risk(self, params: BaseModel, ctx: ToolContext) -> RiskLevel:  # noqa: ARG002
        """Уровень риска по умолчанию. Переопределяется инструментами."""
        return RiskLevel.SAFE

    def preview(self, params: BaseModel, ctx: ToolContext) -> str | None:  # noqa: ARG002
        """Текст/дифф для показа перед подтверждением. None — нечего показывать."""
        return None

    @abstractmethod
    def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """Реестр инструментов: хранение, спецификации, диспетчеризация вызовов."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Инструмент без имени")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]


def build_default_registry() -> ToolRegistry:
    """Собирает реестр со всеми штатными инструментами."""
    # Импорт здесь, чтобы избежать циклов.
    from devassist.tools.fs import (
        EditFileTool,
        FindFilesTool,
        ListDirTool,
        ReadFileTool,
        WriteFileTool,
    )
    from devassist.tools.git import GitTool
    from devassist.tools.search import SearchContentTool
    from devassist.tools.shell import RunShellTool

    reg = ToolRegistry()
    for tool in (
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirTool(),
        FindFilesTool(),
        SearchContentTool(),
        RunShellTool(),
        GitTool(),
    ):
        reg.register(tool)
    return reg
