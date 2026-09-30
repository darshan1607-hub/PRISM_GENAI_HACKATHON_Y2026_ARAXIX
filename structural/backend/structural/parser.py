"""
Tree-Sitter parser for JavaScript, JSX, TypeScript, and TSX source files.

Provides robust multi-language AST generation, syntax error detection,
and AST traversal helpers for downstream static analysis and call graph extraction.
"""

from typing import Any, Dict, Iterator, List, Optional, Union
import logging

from .models import NodeInfo, ParseResult, SourceFile

logger = logging.getLogger(__name__)

# Centralized mapping from language names to grammar keys
LANGUAGE_MAPPINGS: Dict[str, str] = {
    "javascript": "javascript",
    "jsx": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
}


class TreeSitterParser:
    """Parser wrapping Tree-Sitter grammars for JavaScript and TypeScript variants.

    Supports:
    - JavaScript (.js)
    - JSX (.jsx)
    - TypeScript (.ts)
    - TSX (.tsx)

    Gracefully detects syntax errors and returns structured ParseResult instances.
    """

    def __init__(self) -> None:
        """Initialize and cache Tree-Sitter language grammars and parsers."""
        self._parsers: Dict[str, Any] = {}
        self._languages: Dict[str, Any] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """Load Tree-Sitter language grammars using available Python packages.

        Prefers modern modular tree-sitter packages (tree_sitter, tree_sitter_javascript,
        tree_sitter_typescript), falling back to tree_sitter_languages if installed.
        """
        # Try modern tree-sitter (>= 0.22)
        try:
            from tree_sitter import Language, Parser

            # Attempt importing official modular grammar packages
            try:
                import tree_sitter_javascript as ts_js
                import tree_sitter_typescript as ts_ts

                self._languages["javascript"] = Language(ts_js.language())
                # tree-sitter-javascript natively supports JSX expressions
                self._languages["jsx"] = self._languages["javascript"]
                self._languages["typescript"] = Language(ts_ts.language_typescript())
                self._languages["tsx"] = Language(ts_ts.language_tsx())
            except ImportError:
                # Grammar packages not found under standard modern names
                pass

            if self._languages:
                for lang_name, lang_obj in self._languages.items():
                    parser = self._create_parser(Parser, lang_obj)
                    self._parsers[lang_name] = parser
                return
        except ImportError:
            pass

        # Fallback to tree_sitter_languages package if installed
        try:
            import tree_sitter_languages

            for lang_name in ["javascript", "typescript", "tsx"]:
                try:
                    lang = tree_sitter_languages.get_language(lang_name)
                    parser = tree_sitter_languages.get_parser(lang_name)
                    self._languages[lang_name] = lang
                    self._parsers[lang_name] = parser
                except Exception as err:
                    logger.warning("Failed loading tree_sitter_languages for %s: %s", lang_name, err)

            if "javascript" in self._languages:
                self._languages["jsx"] = self._languages["javascript"]
                self._parsers["jsx"] = self._parsers["javascript"]
            return
        except ImportError:
            pass

        # If we reach here, Tree-Sitter packages are not installed or could not be loaded
        logger.warning(
            "Tree-Sitter grammars not initialized. "
            "Please ensure tree-sitter, tree-sitter-javascript, and tree-sitter-typescript are installed."
        )

    def _create_parser(self, parser_class: Any, language: Any) -> Any:
        """Create a parser instance compatible across tree-sitter versions."""
        try:
            # tree-sitter >= 0.22
            return parser_class(language)
        except TypeError:
            # tree-sitter < 0.22
            p = parser_class()
            if hasattr(p, "set_language"):
                p.set_language(language)
            else:
                p.language = language
            return p

    def is_language_supported(self, language: str) -> bool:
        """Check if the given language is recognized and initialized."""
        normalized = language.lower()
        target_grammar = LANGUAGE_MAPPINGS.get(normalized, normalized)
        return target_grammar in self._parsers

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        return list(LANGUAGE_MAPPINGS.keys())

    def _find_syntax_errors(self, root_node: Any, max_errors: int = 5) -> List[NodeInfo]:
        """Collect syntax errors from the AST tree.

        Tree-Sitter creates ERROR or missing nodes when encountering invalid syntax.

        Args:
            root_node: Root node of the parsed AST.
            max_errors: Maximum number of error nodes to collect.

        Returns:
            List of NodeInfo describing error locations.
        """
        errors: List[NodeInfo] = []
        if not getattr(root_node, "has_error", False):
            return errors

        stack = [root_node]
        while stack and len(errors) < max_errors:
            current = stack.pop()
            is_error_node = current.type == "ERROR" or getattr(current, "is_missing", False)
            if is_error_node:
                errors.append(
                    NodeInfo(
                        node_type=current.type,
                        start_point=current.start_point,
                        end_point=current.end_point,
                        start_byte=current.start_byte,
                        end_byte=current.end_byte,
                        has_error=True,
                        is_named=current.is_named,
                    )
                )

            # Recurse into children that contain errors
            for child in reversed(current.children):
                if getattr(child, "has_error", False):
                    stack.append(child)

        return errors

    def parse_code(
        self,
        code: str,
        language: str,
        file_path: str = "<in-memory>",
    ) -> ParseResult:
        """Parse source code string directly.

        Args:
            code: Source code text.
            language: Language name (e.g. 'javascript', 'jsx', 'typescript', 'tsx').
            file_path: Path label for reporting.

        Returns:
            ParseResult object with AST root node or error information.
        """
        lang_key = language.lower()
        target_grammar = LANGUAGE_MAPPINGS.get(lang_key)

        if not target_grammar:
            return ParseResult(
                file_path=file_path,
                language=language,
                success=False,
                error=f"Unsupported language: {language}",
            )

        parser = self._parsers.get(target_grammar)
        if parser is None:
            return ParseResult(
                file_path=file_path,
                language=language,
                success=False,
                error=(
                    f"Tree-Sitter parser for '{target_grammar}' is not initialized. "
                    f"Check tree-sitter packages."
                ),
            )

        try:
            source_bytes = code.encode("utf-8")
            tree = parser.parse(source_bytes)
            root = tree.root_node

            syntax_errors = self._find_syntax_errors(root)
            if syntax_errors:
                first_err = syntax_errors[0]
                error_msg = (
                    f"Syntax error at line {first_err.start_line}, "
                    f"column {first_err.start_point[1] + 1} "
                    f"(node type: '{first_err.node_type}')"
                )
                return ParseResult(
                    file_path=file_path,
                    language=language,
                    success=False,
                    root_node=root,
                    root_node_type=root.type,
                    error=error_msg,
                    tree=tree,
                    syntax_errors=syntax_errors,
                )

            return ParseResult(
                file_path=file_path,
                language=language,
                success=True,
                root_node=root,
                root_node_type=root.type,
                error=None,
                tree=tree,
                syntax_errors=[],
            )
        except Exception as exc:
            logger.exception("Unexpected error while parsing %s: %s", file_path, exc)
            return ParseResult(
                file_path=file_path,
                language=language,
                success=False,
                error=f"Parsing exception: {exc}",
            )

    def parse(self, source_file: SourceFile) -> ParseResult:
        """Parse a SourceFile instance.

        Args:
            source_file: The SourceFile object returned by RepositoryScanner.

        Returns:
            ParseResult with AST root node and metadata.
        """
        return self.parse_code(
            code=source_file.content,
            language=source_file.language,
            file_path=source_file.path,
        )

    def parse_all(self, source_files: List[SourceFile]) -> List[ParseResult]:
        """Parse multiple source files sequentially.

        Ensures errors on individual files do not halt parsing of remaining files.

        Args:
            source_files: List of SourceFile objects.

        Returns:
            List of ParseResult objects corresponding to input files.
        """
        results: List[ParseResult] = []
        for sf in source_files:
            results.append(self.parse(sf))
        return results


def traverse_ast(root_node: Any) -> Iterator[Any]:
    """Depth-first pre-order traversal of Tree-Sitter AST nodes.

    Allows subsequent stages (symbol extraction, call graph extraction)
    to traverse:
        root node -> child nodes -> nested nodes

    Args:
        root_node: The root Tree-Sitter Node.

    Yields:
        Each Tree-Sitter Node in depth-first order.
    """
    if root_node is None:
        return

    stack = [root_node]
    while stack:
        node = stack.pop()
        yield node
        # Push children in reverse order so leftmost child is visited first
        for child in reversed(node.children):
            stack.append(child)


def get_node_info(node: Any) -> NodeInfo:
    """Extract a NodeInfo summary from a Tree-Sitter node.

    Preserves node type, start point, end point, start byte, and end byte.

    Args:
        node: A Tree-Sitter Node instance.

    Returns:
        NodeInfo instance.
    """
    return NodeInfo(
        node_type=node.type,
        start_point=node.start_point,
        end_point=node.end_point,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        has_error=getattr(node, "has_error", False),
        is_named=getattr(node, "is_named", True),
    )


def get_node_text(node: Any, source: Union[str, bytes]) -> str:
    """Extract the source code snippet for a given AST node.

    Args:
        node: A Tree-Sitter Node instance.
        source: Source code as string or bytes.

    Returns:
        Extracted source code substring.
    """
    if isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
