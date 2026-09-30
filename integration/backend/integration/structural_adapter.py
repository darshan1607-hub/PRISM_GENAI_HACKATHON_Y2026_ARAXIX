"""Adapter for Member 3 (Structural Intelligence). Bridges Tree-Sitter AST queries to normalized evidence."""
import os
from pathlib import Path
from typing import List, Optional
from backend.integration.evidence import NormalizedEvidence, SourceType


class MockStructuralAdapter:
    def query(self, capability: str, target: str) -> List[NormalizedEvidence]:
        m = {"is_mock": True}
        if capability == "symbol_search":
            return [NormalizedEvidence("[MOCK] src/navigation/DeepLinkRouter.js", 42, 42,
                    "// MOCK DATA\nexport const BLUETOOTH_SETTINGS_URI = 'app://settings/bluetooth';",
                    SourceType.STRUCTURAL, "BLUETOOTH_SETTINGS_URI", 1.0, True, m)]
        if capability == "call_graph":
            return [NormalizedEvidence("[MOCK] src/ui/DeviceSettings.jsx", 118, 118,
                    "// MOCK DATA\nnavigateToDeepLink(BLUETOOTH_SETTINGS_URI);",
                    SourceType.CALL_GRAPH, "navigateToDeepLink", 0.95, True, m)]
        return []


class RealStructuralAdapter:
    """Connects to Member 3's Tree-Sitter based StructuralQueryEngine."""
    implemented = True

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path or os.environ.get("REPO_PATH")
        if not self.repo_path:
            # Fallback to local demo repository
            for candidate in ("semantics/demo_repo", "structural/demo_repo", "demo_repo"):
                if Path(candidate).is_dir():
                    self.repo_path = candidate
                    break
            if not self.repo_path:
                self.repo_path = "demo_repo"
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from backend.structural import StructuralQueryEngine
            self._engine = StructuralQueryEngine(self.repo_path)
        return self._engine

    def query(self, capability: str, target: str) -> List[NormalizedEvidence]:
        try:
            matches = self._get_engine().query(capability, target)
        except Exception:
            return []

        out = []
        for m in matches:
            stype = (
                SourceType.CALL_GRAPH
                if m.match_type == "call_site"
                else (SourceType.AST if m.match_type == "caller_context" else SourceType.STRUCTURAL)
            )
            meta = dict(m.metadata)
            meta["match_type"] = m.match_type
            if m.enclosing_symbol:
                meta["caller"] = m.enclosing_symbol
            out.append(
                NormalizedEvidence(
                    file_path=m.file_path,
                    start_line=m.start_line,
                    end_line=m.end_line,
                    snippet=m.snippet,
                    source_type=stype,
                    symbol=m.symbol_name,
                    relevance=1.0,
                    is_mock=False,
                    metadata=meta,
                )
            )
        return out