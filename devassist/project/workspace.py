"""Рабочее пространство проекта и служебная папка ``.devassist/``.

``.devassist/`` в корне проекта — место для данных агента (индекс проекта, чаты).
Папка создаётся лениво, первым компонентом, которому нужно туда писать
(:meth:`Workspace.ensure_data_dir`), и сразу получает собственный ``.gitignore``
со звёздочкой, чтобы её содержимое никогда не попадало в репозиторий.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR_NAME = ".devassist"
_GITIGNORE_BODY = "# Создано devassist: рабочие данные агента не коммитятся.\n*\n"


@dataclass(frozen=True)
class Workspace:
    """Корень проекта и производные пути. Конструктор не трогает файловую систему."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def data_dir(self) -> Path:
        """``<root>/.devassist`` (может ещё не существовать)."""
        return self.root / DATA_DIR_NAME

    @property
    def chats_dir(self) -> Path:
        """Каталог сохранённых чатов."""
        return self.data_dir / "chats"

    @property
    def index_dir(self) -> Path:
        """Каталог индекса проекта."""
        return self.data_dir / "index"

    def ensure_data_dir(self) -> Path:
        """Создаёт ``.devassist/`` с ``.gitignore`` (идемпотентно) и возвращает путь.

        Существующий ``.gitignore`` не перезаписывается — пользователь мог его изменить.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.data_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_GITIGNORE_BODY, encoding="utf-8")
        return self.data_dir
