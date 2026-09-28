"""
Code Chunking Module (Member 2 - Semantic Code Search).

Extracts semantically meaningful chunks (functions, classes, methods,
interfaces, logical blocks) using Tree-Sitter AST information from Member 3,
with robust line-based sliding-window chunking as a fallback.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

try:
    from backend.structural.models import SourceFile, ParseResult
    from backend.structural.parser import TreeSitterParser, get_node_text
except ImportError:
    # Allow running if structural module is accessed differently
    SourceFile = Any  # type: ignore
    ParseResult = Any  # type: ignore
    TreeSitterParser = Any  # type: ignore
    get_node_text = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a searchable segment of source code.

    Attributes:
        file: Relative file path from repository root (e.g. 'src/example.ts').
        start_line: 1-indexed starting line number in source file.
        end_line: 1-indexed ending line number in source file.
        code: Exact source code content of the chunk.
        chunk_type: Category of chunk ('function', 'method', 'class', 'interface', 'block', 'lines').
        name: Name of symbol/identifier if detected (e.g. 'authenticateUser').
    """

    file: str
    start_line: int
    end_line: int
    code: str
    chunk_type: str = "block"
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert CodeChunk to a dictionary matching the required specification."""
        data: Dict[str, Any] = {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
        }
        if self.chunk_type:
            data["chunk_type"] = self.chunk_type
        if self.name:
            data["name"] = self.name
        return data

    def __repr__(self) -> str:
        name_str = f" ({self.name})" if self.name else ""
        return (
            f"CodeChunk({self.file}:{self.start_line}-{self.end_line} "
            f"[{self.chunk_type}{name_str}])"
        )


class CodeChunker:
    """Divides source code into meaningful, searchable code chunks.

    Combines:
    1. AST-aware chunking: extracts functions, methods, classes, and interfaces
       using Tree-Sitter parsing from Member 3.
    2. Docstring attachment: preserves JSDoc/block comments preceding definitions.
    3. Line-based chunking fallback: handles files with syntax errors, unsupported
       grammars, or when line-based chunking is explicitly requested.
    """

    def __init__(
        self,
        line_chunk_size: int = 30,
        line_overlap: int = 5,
        attach_docstrings: bool = True,
        parser: Optional[Any] = None,
    ) -> None:
        """Initialize the CodeChunker.

        Args:
            line_chunk_size: Default window size in lines for line-based fallback.
            line_overlap: Number of overlapping lines between consecutive line chunks.
            attach_docstrings: Whether to include immediately preceding comments in chunk.
            parser: Optional TreeSitterParser instance (instantiated if None).
        """
        self.line_chunk_size = max(5, line_chunk_size)
        self.line_overlap = max(0, min(line_overlap, self.line_chunk_size - 1))
        self.attach_docstrings = attach_docstrings

        if parser is not None:
            self._parser = parser
        else:
            try:
                self._parser = TreeSitterParser()
            except Exception:
                self._parser = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def chunk_file(
        self,
        source_file: Any,
        parse_result: Optional[Any] = None,
        strategy: str = "auto",
    ) -> List[CodeChunk]:
        """Divide a single SourceFile into CodeChunks.

        Args:
            source_file: SourceFile object (with path, content, language).
            parse_result: Optional precomputed ParseResult from Member 3.
            strategy: 'auto' (AST with line fallback), 'ast' (AST only), or 'lines'.

        Returns:
            List of CodeChunk objects.
        """
        if strategy == "lines":
            return self.chunk_lines(
                file_path=source_file.path,
                code=source_file.content,
                chunk_size=self.line_chunk_size,
                overlap=self.line_overlap,
            )

        # Attempt AST chunking
        if self._parser or parse_result:
            if parse_result is None and self._parser:
                parse_result = self._parser.parse(source_file)

            if parse_result and parse_result.success and parse_result.root_node:
                ast_chunks = self._chunk_from_ast(
                    file_path=source_file.path,
                    code=source_file.content,
                    root_node=parse_result.root_node,
                )
                if ast_chunks:
                    return ast_chunks

        # Fallback to line-based chunking if strategy is 'auto' or AST produced nothing
        if strategy == "auto":
            return self.chunk_lines(
                file_path=source_file.path,
                code=source_file.content,
                chunk_size=self.line_chunk_size,
                overlap=self.line_overlap,
            )

        return []

    def chunk_code(
        self,
        code: str,
        file_path: str = "<in-memory>",
        language: Optional[str] = None,
        strategy: str = "auto",
    ) -> List[CodeChunk]:
        """Divide raw source code string into CodeChunks.

        Args:
            code: Source code text.
            file_path: Relative path label for chunks.
            language: Programming language name (e.g. 'javascript', 'typescript').
            strategy: 'auto', 'ast', or 'lines'.

        Returns:
            List of CodeChunk objects.
        """
        if strategy == "lines" or not language or not self._parser:
            return self.chunk_lines(
                file_path=file_path,
                code=code,
                chunk_size=self.line_chunk_size,
                overlap=self.line_overlap,
            )

        parse_result = self._parser.parse_code(code, language, file_path)
        if parse_result.success and parse_result.root_node:
            ast_chunks = self._chunk_from_ast(
                file_path=file_path,
                code=code,
                root_node=parse_result.root_node,
            )
            if ast_chunks:
                return ast_chunks

        if strategy == "auto":
            return self.chunk_lines(
                file_path=file_path,
                code=code,
                chunk_size=self.line_chunk_size,
                overlap=self.line_overlap,
            )

        return []

    def chunk_repository(
        self,
        source_files: List[Any],
        parser: Optional[Any] = None,
    ) -> List[CodeChunk]:
        """Chunk an entire repository represented by a list of SourceFiles.

        Reuses Member 1 SourceFile list and Member 3 TreeSitterParser.

        Args:
            source_files: List of SourceFile objects discovered by RepositoryScanner.
            parser: Optional parser to override default parser.

        Returns:
            Consolidated list of CodeChunk objects across all files.
        """
        active_parser = parser or self._parser
        all_chunks: List[CodeChunk] = []

        for sf in source_files:
            parse_result = None
            if active_parser:
                parse_result = active_parser.parse(sf)

            chunks = self.chunk_file(
                source_file=sf,
                parse_result=parse_result,
                strategy="auto",
            )
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_lines(
        self,
        file_path: str,
        code: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[CodeChunk]:
        """Divide source code into fixed-line window chunks with overlap.

        Guaranteed fallback that never loses or modifies original source code.

        Args:
            file_path: Relative path of the file.
            code: Source code string.
            chunk_size: Number of lines per chunk (defaults to self.line_chunk_size).
            overlap: Overlap in lines between windows (defaults to self.line_overlap).

        Returns:
            List of CodeChunk objects.
        """
        lines = code.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return []

        c_size = chunk_size if chunk_size is not None else self.line_chunk_size
        o_lap = overlap if overlap is not None else self.line_overlap

        # If file is smaller than chunk_size, return 1 chunk for the whole file
        if total_lines <= c_size:
            return [
                CodeChunk(
                    file=file_path,
                    start_line=1,
                    end_line=total_lines,
                    code=code,
                    chunk_type="lines",
                )
            ]

        chunks: List[CodeChunk] = []
        step = max(1, c_size - o_lap)
        start_idx = 0

        while start_idx < total_lines:
            end_idx = min(start_idx + c_size, total_lines)
            chunk_lines_slice = lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines_slice)

            chunks.append(
                CodeChunk(
                    file=file_path,
                    start_line=start_idx + 1,
                    end_line=end_idx,
                    code=chunk_text,
                    chunk_type="lines",
                )
            )

            if end_idx >= total_lines:
                break
            start_idx += step

        return chunks

    # -------------------------------------------------------------------------
    # AST Traversal & Chunk Extraction
    # -------------------------------------------------------------------------

    def _chunk_from_ast(
        self,
        file_path: str,
        code: str,
        root_node: Any,
    ) -> List[CodeChunk]:
        """Extract meaningful code units from a Tree-Sitter AST."""
        lines = code.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return []

        chunks: List[CodeChunk] = []
        covered_lines: set[int] = set()

        children = list(root_node.children)
        i = 0
        while i < len(children):
            child = children[i]

            # Check if this child is a docstring/comment immediately preceding a declaration
            doc_comment_node = None
            if (
                self.attach_docstrings
                and child.type == "comment"
                and i + 1 < len(children)
            ):
                next_child = children[i + 1]
                # If comment ends within 2 lines of the next declaration
                comment_end_line = self._get_end_line(child)
                next_start_line = self._get_start_line(next_child)
                if 0 <= next_start_line - comment_end_line <= 2:
                    if self._is_declaration_node(next_child):
                        doc_comment_node = child
                        child = next_child
                        i += 1  # Advance past the doc comment

            # Process the child node
            extracted = self._extract_node_chunks(
                file_path=file_path,
                lines=lines,
                node=child,
                doc_node=doc_comment_node,
            )

            for chunk in extracted:
                chunks.append(chunk)
                for l_num in range(chunk.start_line, chunk.end_line + 1):
                    covered_lines.add(l_num)

            i += 1

        # Check for unchunked top-level code blocks (e.g. script statements, configs)
        unchunked_blocks = self._collect_uncovered_blocks(
            file_path=file_path,
            lines=lines,
            covered_lines=covered_lines,
        )
        chunks.extend(unchunked_blocks)

        # Sort chunks deterministically by start_line
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    def _extract_node_chunks(
        self,
        file_path: str,
        lines: List[str],
        node: Any,
        doc_node: Optional[Any] = None,
    ) -> List[CodeChunk]:
        """Inspect an AST node and extract chunks (including nested methods if class)."""
        ntype = node.type
        chunks: List[CodeChunk] = []

        # 1. Export statements: unwrap inner declaration
        if ntype in ("export_statement", "export_default_declaration"):
            inner = self._get_exported_declaration(node)
            if inner:
                inner_chunks = self._extract_node_chunks(
                    file_path=file_path,
                    lines=lines,
                    node=inner,
                    doc_node=doc_node,
                )
                if inner_chunks:
                    # Expand the first chunk's boundaries to include the 'export' keyword
                    start_line = self._get_start_line(doc_node if doc_node else node)
                    first = inner_chunks[0]
                    if start_line < first.start_line:
                        first.start_line = start_line
                        first.code = "\n".join(lines[start_line - 1 : first.end_line])
                    return inner_chunks

        # 2. Function declarations
        if ntype in ("function_declaration", "generator_function_declaration"):
            name = self._find_child_text(node, "identifier")
            chunk = self._build_chunk(
                file_path=file_path,
                lines=lines,
                node=node,
                doc_node=doc_node,
                chunk_type="function",
                name=name,
            )
            if chunk:
                chunks.append(chunk)
            return chunks

        # 3. Variable declarations containing arrow functions
        if ntype in ("lexical_declaration", "variable_declaration"):
            arrow_func, func_name = self._find_arrow_function(node)
            if arrow_func is not None:
                chunk = self._build_chunk(
                    file_path=file_path,
                    lines=lines,
                    node=node,
                    doc_node=doc_node,
                    chunk_type="function",
                    name=func_name,
                )
                if chunk:
                    chunks.append(chunk)
                return chunks

        # 4. Class declarations: extract class AND individual methods
        if ntype == "class_declaration":
            class_name = self._find_child_text(node, ("identifier", "type_identifier"))
            # Build chunk for the class overall
            class_chunk = self._build_chunk(
                file_path=file_path,
                lines=lines,
                node=node,
                doc_node=doc_node,
                chunk_type="class",
                name=class_name,
            )
            if class_chunk:
                chunks.append(class_chunk)

            # Extract individual methods
            methods = self._extract_class_methods(file_path, lines, node, class_name)
            chunks.extend(methods)
            return chunks

        # 5. Interface declarations
        if ntype == "interface_declaration":
            name = self._find_child_text(node, "type_identifier")
            chunk = self._build_chunk(
                file_path=file_path,
                lines=lines,
                node=node,
                doc_node=doc_node,
                chunk_type="interface",
                name=name,
            )
            if chunk:
                chunks.append(chunk)
            return chunks

        # 6. Type alias declarations
        if ntype == "type_alias_declaration":
            name = self._find_child_text(node, "type_identifier")
            chunk = self._build_chunk(
                file_path=file_path,
                lines=lines,
                node=node,
                doc_node=doc_node,
                chunk_type="type",
                name=name,
            )
            if chunk:
                chunks.append(chunk)
            return chunks

        return chunks

    def _extract_class_methods(
        self,
        file_path: str,
        lines: List[str],
        class_node: Any,
        class_name: Optional[str],
    ) -> List[CodeChunk]:
        """Extract method definitions from a class body."""
        methods: List[CodeChunk] = []
        body_node = None
        for child in class_node.children:
            if child.type == "class_body":
                body_node = child
                break

        if not body_node:
            return methods

        body_children = list(body_node.children)
        j = 0
        while j < len(body_children):
            m_node = body_children[j]
            doc_node = None
            if self.attach_docstrings and m_node.type == "comment" and j + 1 < len(body_children):
                next_m = body_children[j + 1]
                if next_m.type == "method_definition":
                    doc_node = m_node
                    m_node = next_m
                    j += 1

            if m_node.type == "method_definition":
                m_name = self._find_child_text(m_node, "property_identifier")
                full_name = f"{class_name}.{m_name}" if class_name and m_name else m_name
                chunk = self._build_chunk(
                    file_path=file_path,
                    lines=lines,
                    node=m_node,
                    doc_node=doc_node,
                    chunk_type="method",
                    name=full_name,
                )
                if chunk:
                    methods.append(chunk)

            j += 1

        return methods

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    def _build_chunk(
        self,
        file_path: str,
        lines: List[str],
        node: Any,
        doc_node: Optional[Any],
        chunk_type: str,
        name: Optional[str] = None,
    ) -> Optional[CodeChunk]:
        """Build a CodeChunk from an AST node, optionally attaching leading docstring."""
        start_line = self._get_start_line(doc_node if doc_node else node)
        end_line = self._get_end_line(node)

        if start_line > end_line or start_line > len(lines):
            return None

        actual_end = min(end_line, len(lines))
        chunk_lines = lines[start_line - 1 : actual_end]
        code_text = "\n".join(chunk_lines)

        if not code_text.strip():
            return None

        return CodeChunk(
            file=file_path,
            start_line=start_line,
            end_line=actual_end,
            code=code_text,
            chunk_type=chunk_type,
            name=name,
        )

    def _collect_uncovered_blocks(
        self,
        file_path: str,
        lines: List[str],
        covered_lines: set[int],
    ) -> List[CodeChunk]:
        """Collect contiguous unchunked lines (excluding pure whitespace/single comments) into blocks."""
        blocks: List[CodeChunk] = []
        total = len(lines)
        curr_start: Optional[int] = None

        for l_num in range(1, total + 1):
            is_covered = l_num in covered_lines
            line_str = lines[l_num - 1].strip()
            is_empty = len(line_str) == 0

            if not is_covered and not is_empty:
                if curr_start is None:
                    curr_start = l_num
            else:
                if curr_start is not None:
                    # Finish the current block if it contains meaningful content
                    block_lines = lines[curr_start - 1 : l_num - 1]
                    block_text = "\n".join(block_lines)
                    if block_text.strip() and len(block_lines) >= 1:
                        blocks.append(
                            CodeChunk(
                                file=file_path,
                                start_line=curr_start,
                                end_line=l_num - 1,
                                code=block_text,
                                chunk_type="block",
                            )
                        )
                    curr_start = None

        if curr_start is not None:
            block_lines = lines[curr_start - 1 : total]
            block_text = "\n".join(block_lines)
            if block_text.strip():
                blocks.append(
                    CodeChunk(
                        file=file_path,
                        start_line=curr_start,
                        end_line=total,
                        code=block_text,
                        chunk_type="block",
                    )
                )

        return blocks

    def _is_declaration_node(self, node: Any) -> bool:
        """Check if an AST node is a declaration eligible for docstring attachment."""
        t = node.type
        if t in (
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "export_statement",
            "export_default_declaration",
        ):
            return True
        if t in ("lexical_declaration", "variable_declaration"):
            arrow_node, _ = self._find_arrow_function(node)
            return arrow_node is not None
        return False

    def _get_exported_declaration(self, export_node: Any) -> Optional[Any]:
        """Find the inner declaration within an export statement."""
        for child in export_node.children:
            if child.type in (
                "function_declaration",
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "lexical_declaration",
                "variable_declaration",
                "declaration",
            ):
                return child
        return None

    def _find_arrow_function(self, decl_node: Any) -> Tuple[Optional[Any], Optional[str]]:
        """Check if a lexical declaration contains an arrow function assignment."""
        for child in decl_node.children:
            if child.type == "variable_declarator":
                var_name = self._find_child_text(child, "identifier")
                for sub in child.children:
                    if sub.type in ("arrow_function", "function_expression"):
                        return sub, var_name
        return None, None

    def _find_child_text(self, node: Any, target_types: Union[str, Tuple[str, ...]]) -> Optional[str]:
        """Extract text of first child matching specified target types."""
        types_tuple = (target_types,) if isinstance(target_types, str) else target_types
        for child in node.children:
            if child.type in types_tuple:
                try:
                    return child.text.decode("utf-8", errors="replace")
                except Exception:
                    return getattr(child, "text", None)
        return None

    def _get_start_line(self, node: Any) -> int:
        """1-indexed start line of AST node."""
        return node.start_point[0] + 1

    def _get_end_line(self, node: Any) -> int:
        """1-indexed end line of AST node."""
        if node.end_point[1] == 0 and node.end_point[0] > node.start_point[0]:
            return node.end_point[0]
        return node.end_point[0] + 1
