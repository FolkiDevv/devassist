"""Знание о проекте: рабочая папка .devassist, файлы проекта, инструкции.

Слой не зависит от LLM и UI — на него опираются инструменты и ядро агента
(а в будущем — индекс проекта).
"""

from devassist.project.files import IGNORE_DIRS, build_file_tree, glob_match, walk_files
from devassist.project.instructions import INSTRUCTION_FILES, InstructionFile, load_instructions
from devassist.project.workspace import DATA_DIR_NAME, Workspace

__all__ = [
    "DATA_DIR_NAME",
    "IGNORE_DIRS",
    "INSTRUCTION_FILES",
    "InstructionFile",
    "Workspace",
    "build_file_tree",
    "glob_match",
    "load_instructions",
    "walk_files",
]
