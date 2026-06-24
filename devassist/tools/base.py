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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

from devassist.devassist.config import Config
from devassist.devassist.llm.types import ToolSpec
from devassist.devassist.security import RiskLevel

P = TypeVar("P", bound=BaseModel)


@dataclass
class ToolContext:
    """Контекст выполнения, доступный инструменту."""

    config: Config

    @property
    def root(self) -> Path:
        return self.config.project_root


@dataclass
class ToolResult:
    """Результат выполнения инструмента."""

    content: str                      # текст, который вернётся модели
    ok: bool = True
    summary: str = ""                 # краткая строка для UI
    display: Optional[str] = None     # доп. вывод для пользователя (дифф/листинг)

    def as_function_content(self) -> str:
        if self.ok:
            return self.content
        return f"ОШИБКА: {self.content}"


class ToolError(Exception):
    """Ожидаемая ошибка инструмента (возвращается модели, не роняет агента)."""


def _normalize_property(prop: Dict[str, Any]) -> Dict[str, Any]:
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


def _clean_schema(model: Type[BaseModel]) -> Dict[str, Any]:
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
    Params: Type[BaseModel] = BaseModel

    # ------------------------------------------------------------------ #
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=_clean_schema(self.Params),
        )

    def parse(self, arguments: Dict[str, Any]) -> BaseModel:
        return self.Params.model_validate(arguments or {})

    def risk(self, params: BaseModel, ctx: ToolContext) -> RiskLevel:  # noqa: ARG002
        """Уровень риска по умолчанию. Переопределяется инструментами."""
        return RiskLevel.SAFE

    def preview(self, params: BaseModel, ctx: ToolContext) -> Optional[str]:  # noqa: ARG002
        """Текст/дифф для показа перед подтверждением. None — нечего показывать."""
        return None

    @abstractmethod
    def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        ...


class ToolRegistry:
    """Реестр инструментов: хранение, спецификации, диспетчеризация вызовов."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Инструмент без имени")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def specs(self) -> List[ToolSpec]:
        return [t.spec() for t in self._tools.values()]


def build_default_registry() -> ToolRegistry:
    """Собирает реестр со всеми штатными инструментами."""
    # Импорт здесь, чтобы избежать циклов.
    from devassist.devassist.tools.fs import (
        EditFileTool,
        FindFilesTool,
        ListDirTool,
        ReadFileTool,
        WriteFileTool,
    )
    from devassist.devassist.tools.git import GitTool
    from devassist.devassist.tools.search import SearchContentTool
    from devassist.devassist.tools.shell import RunShellTool

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
