"""Знание о проекте: рабочая папка .devassist, файлы проекта, инструкции, индекс.

Слой не зависит от LLM и UI — на него опираются инструменты и ядро агента.
"""

from devassist.project.files import (
    IGNORE_DIRS,
    build_file_tree,
    glob_match,
    is_excluded,
    walk_files,
)
from devassist.project.index import INDEX_ERRORS, ProjectIndex
from devassist.project.instructions import (
    INSTRUCTION_FILES,
    InstructionFile,
    NestedInstructions,
    load_instructions,
)
from devassist.project.symbols import Symbol, extract_symbols, language_of
from devassist.project.workspace import DATA_DIR_NAME, Workspace

__all__ = [
    "DATA_DIR_NAME",
    "IGNORE_DIRS",
    "INDEX_ERRORS",
    "INSTRUCTION_FILES",
    "InstructionFile",
    "NestedInstructions",
    "ProjectIndex",
    "Symbol",
    "Workspace",
    "build_file_tree",
    "extract_symbols",
    "glob_match",
    "is_excluded",
    "language_of",
    "load_instructions",
    "walk_files",
]
