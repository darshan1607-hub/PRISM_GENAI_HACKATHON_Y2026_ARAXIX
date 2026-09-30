"""
Data models for Structural Code Intelligence (Stage 1).

These models define structured representations for:
- Source files discovered in a repository.
- Tree-Sitter parse results and AST node summaries.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List


@dataclass
class SourceFile:
    """Represents a discovered source code file within a repository.

    Attributes:
        path: Relative path of the file from the repository root (using forward slashes).
        language: Identified language (e.g. 'javascript', 'jsx', 'typescript', 'tsx').
        content: Full text content of the source file (UTF-8 decoded).
        line_count: Total number of lines in the file.
        abs_path: Optional absolute filesystem path to the file.
        byte_size: Optional file size in bytes.
    """
    path: str
    language: str
    content: str
    line_count: int
    abs_path: Optional[str] = None
    byte_size: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"SourceFile(path={self.path!r}, language={self.language!r}, "
            f"line_count={self.line_count})"
        )


@dataclass
class NodeInfo:
    """Lightweight, serializable summary of a Tree-Sitter AST node.

    Preserves critical syntactic and positional information without retaining
    live C-pointers, making it safe for JSON serialization and logging.

    Attributes:
        node_type: Grammatical type of the node (e.g., 'program', 'function_declaration').
        start_point: (row, column) tuple representing the start position (0-indexed).
        end_point: (row, column) tuple representing the end position (0-indexed).
        start_byte: Start byte offset in the source buffer.
        end_byte: End byte offset in the source buffer.
        has_error: Whether this node or any descendant contains a syntax error.
        is_named: Whether this node is a named syntax rule vs anonymous symbol.
    """
    node_type: str
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    start_byte: int
    end_byte: int
    has_error: bool = False
    is_named: bool = True

    @property
    def start_line(self) -> int:
        """1-indexed start line number for user-facing output."""
        return self.start_point[0] + 1

    @property
    def end_line(self) -> int:
        """1-indexed end line number for user-facing output."""
        return self.end_point[0] + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert NodeInfo to a JSON-serializable dictionary."""
        return {
            "node_type": self.node_type,
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "has_error": self.has_error,
            "is_named": self.is_named,
        }


@dataclass
class ParseResult:
    """Structured result of parsing a source file with Tree-Sitter.

    Maintains both live AST references (for fast downstream traversal in
    subsequent stages) and summary metadata (for error reporting and serialization).

    Attributes:
        file_path: Relative path of the parsed file.
        language: Language used for parsing.
        success: True if parsing succeeded without fatal errors or syntax violations.
        root_node: The live Tree-Sitter Node instance (or None if parsing failed).
        root_node_type: Type of the root AST node (e.g. 'program').
        error: Human-readable error description if parsing failed or syntax error found.
        tree: Optional reference to the live Tree-Sitter Tree object.
        syntax_errors: List of detected syntax error summaries, if any.
    """
    file_path: str
    language: str
    success: bool
    root_node: Optional[Any] = None
    root_node_type: Optional[str] = None
    error: Optional[str] = None
    tree: Optional[Any] = field(default=None, repr=False)
    syntax_errors: List[NodeInfo] = field(default_factory=list)

    def get_root_summary(self) -> Optional[NodeInfo]:
        """Extract a NodeInfo summary from the live root node if available."""
        if self.root_node is None:
            return None
        return NodeInfo(
            node_type=self.root_node.type,
            start_point=self.root_node.start_point,
            end_point=self.root_node.end_point,
            start_byte=self.root_node.start_byte,
            end_byte=self.root_node.end_byte,
            has_error=getattr(self.root_node, "has_error", False),
            is_named=getattr(self.root_node, "is_named", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Produce a clean dictionary representation of the parse result."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "success": self.success,
            "root_node_type": self.root_node_type,
            "error": self.error,
            "root_summary": self.get_root_summary().to_dict() if self.get_root_summary() else None,
            "syntax_error_count": len(self.syntax_errors),
        }

    def __repr__(self) -> str:
        return (
            f"ParseResult(file_path={self.file_path!r}, language={self.language!r}, "
            f"success={self.success}, root_node_type={self.root_node_type!r}, "
            f"error={self.error!r})"
        )
