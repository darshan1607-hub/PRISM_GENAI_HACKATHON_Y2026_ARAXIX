"""
Structural Query Engine (Member 3 - Stage 2/3 Extension).

Builds upon TreeSitterParser and RepositoryScanner to provide:
1. Definition and declaration discovery (functions, classes, interfaces, variables)
2. Call graph analysis & caller context identification
3. Symbol searches and AST node inspection
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .models import SourceFile
from .parser import TreeSitterParser, traverse_ast, get_node_text
from .repository import RepositoryScanner


class StructuralMatch:
    """Represents an extracted structural code element (definition, call site, or reference)."""

    def __init__(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        snippet: str,
        match_type: str,
        symbol_name: str,
        enclosing_symbol: Optional[str] = None,
        language: str = "javascript",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.snippet = snippet
        self.match_type = match_type  # 'declaration', 'definition', 'call_site', 'caller_context'
        self.symbol_name = symbol_name
        self.enclosing_symbol = enclosing_symbol
        self.language = language
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "snippet": self.snippet,
            "match_type": self.match_type,
            "symbol_name": self.symbol_name,
            "enclosing_symbol": self.enclosing_symbol,
            "language": self.language,
            "metadata": self.metadata,
        }


class StructuralQueryEngine:
    """AST-driven structural intelligence query engine for JavaScript & TypeScript."""

    def __init__(
        self,
        repo_path: Union[str, Path],
        parser: Optional[TreeSitterParser] = None,
    ):
        self.repo_path = Path(repo_path)
        self.parser = parser or TreeSitterParser()
        self._scanner = (
            RepositoryScanner(self.repo_path) if self.repo_path.is_dir() else None
        )

    def _get_source_files(self) -> List[SourceFile]:
        if not self._scanner:
            return []
        try:
            return self._scanner.scan()
        except Exception:
            return []

    def _extract_identifier(self, node: Any, source: str) -> Optional[str]:
        """Extract primary identifier from node children."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return get_node_text(child, source)
        return None

    def query(self, capability: str, target: str) -> List[StructuralMatch]:
        """Execute structural query across the repository.

        Args:
            capability: 'symbol_search', 'call_graph', 'ast_inspect', or generic
            target: symbol name or identifier being investigated
        """
        if not target or not target.strip():
            return []

        target_clean = target.strip()
        target_lower = target_clean.lower()
        files = self._get_source_files()
        results: List[StructuralMatch] = []

        is_call_graph_mode = capability in ("call_graph", "find_callers")
        is_symbol_search_mode = capability in ("symbol_search", "find_definition")
        is_ast_inspect_mode = capability in ("ast_inspect", "caller_context")

        for sf in files:
            res = self.parser.parse(sf)
            if not res.success or not res.root_node:
                continue

            source = sf.content
            lines = source.splitlines()

            stack = [(res.root_node, None)]
            while stack:
                node, enclosing = stack.pop()
                ntype = node.type
                curr_enclosing = enclosing

                if ntype in ("function_declaration", "class_declaration", "method_definition"):
                    fn_name = self._extract_identifier(node, source)
                    if fn_name:
                        curr_enclosing = fn_name

                # 1. Definitions and declarations
                if not is_call_graph_mode:
                    if ntype in (
                        "function_declaration",
                        "class_declaration",
                        "method_definition",
                        "variable_declarator",
                        "interface_declaration",
                        "type_alias_declaration",
                    ):
                        sym_name = self._extract_identifier(node, source)
                        if sym_name and (
                            target_lower == sym_name.lower()
                            or target_lower in sym_name.lower()
                            or sym_name.lower() in target_lower
                        ):
                            s_line = node.start_point[0] + 1
                            e_line = node.end_point[0] + 1
                            snippet = "\n".join(lines[s_line - 1 : min(e_line, s_line + 15)])
                            m_type = (
                                "declaration"
                                if "declaration" in ntype or ntype == "variable_declarator"
                                else "definition"
                            )
                            results.append(
                                StructuralMatch(
                                    file_path=sf.path,
                                    start_line=s_line,
                                    end_line=e_line,
                                    snippet=snippet,
                                    match_type=m_type,
                                    symbol_name=sym_name,
                                    enclosing_symbol=curr_enclosing,
                                    language=sf.language,
                                    metadata={
                                        "component": sf.path.split("/")[0] if "/" in sf.path else "core",
                                        "caller": curr_enclosing,
                                        "node_type": ntype,
                                    },
                                )
                            )

                # 2. Call sites & usages
                if not is_symbol_search_mode:
                    if ntype == "call_expression":
                        call_text = get_node_text(node, source)
                        callee = node.child_by_field_name("function") or (
                            node.children[0] if node.children else None
                        )
                        callee_name = self._extract_identifier(callee, source) if callee else None
                        if not callee_name and callee and callee.type == "member_expression":
                            prop = callee.child_by_field_name("property")
                            if prop:
                                callee_name = get_node_text(prop, source)

                        matched = False
                        if callee_name and (
                            target_lower == callee_name.lower()
                            or target_lower in callee_name.lower()
                            or callee_name.lower() in target_lower
                        ):
                            matched = True
                        elif target_lower in call_text.lower():
                            matched = True

                        if matched:
                            s_line = node.start_point[0] + 1
                            e_line = node.end_point[0] + 1
                            snippet = (
                                lines[s_line - 1].strip()
                                if s_line <= len(lines)
                                else call_text.strip()
                            )
                            results.append(
                                StructuralMatch(
                                    file_path=sf.path,
                                    start_line=s_line,
                                    end_line=e_line,
                                    snippet=snippet,
                                    match_type="call_site",
                                    symbol_name=callee_name or target_clean,
                                    enclosing_symbol=curr_enclosing,
                                    language=sf.language,
                                    metadata={
                                        "caller": curr_enclosing or "module",
                                        "call_site": True,
                                        "call_expression": call_text[:120],
                                    },
                                )
                            )

                # 3. Caller / AST Context
                if is_ast_inspect_mode:
                    if (
                        ntype in ("function_declaration", "class_declaration", "export_statement")
                        and target_lower in get_node_text(node, source).lower()
                    ):
                        s_line = node.start_point[0] + 1
                        e_line = node.end_point[0] + 1
                        snippet = "\n".join(lines[s_line - 1 : min(e_line, s_line + 8)])
                        sym = self._extract_identifier(node, source) or target_clean
                        results.append(
                            StructuralMatch(
                                file_path=sf.path,
                                start_line=s_line,
                                end_line=e_line,
                                snippet=snippet,
                                match_type="caller_context",
                                symbol_name=sym,
                                enclosing_symbol=curr_enclosing,
                                language=sf.language,
                                metadata={"caller": curr_enclosing},
                            )
                        )

                for child in reversed(node.children):
                    stack.append((child, curr_enclosing))

        return results