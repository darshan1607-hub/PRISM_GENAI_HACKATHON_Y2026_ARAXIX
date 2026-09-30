"""
Structural Code Intelligence (Stage 1 + Stage 2/3 Query Engine).

Exposes models, repository scanner, Tree-Sitter parser, and
structural query engine for JavaScript and TypeScript codebases.
"""

from .models import NodeInfo, ParseResult, SourceFile
from .repository import (
    IGNORED_DIRECTORIES,
    SUPPORTED_EXTENSIONS,
    RepositoryScanner,
)
from .parser import (
    LANGUAGE_MAPPINGS,
    TreeSitterParser,
    get_node_info,
    get_node_text,
    traverse_ast,
)
from .query import StructuralMatch, StructuralQueryEngine

__all__ = [
    "SourceFile",
    "NodeInfo",
    "ParseResult",
    "RepositoryScanner",
    "TreeSitterParser",
    "SUPPORTED_EXTENSIONS",
    "IGNORED_DIRECTORIES",
    "LANGUAGE_MAPPINGS",
    "traverse_ast",
    "get_node_info",
    "get_node_text",
    "StructuralMatch",
    "StructuralQueryEngine",
]