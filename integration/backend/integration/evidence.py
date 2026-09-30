"""Common evidence model + normalization of Member 1-3 outputs."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class SourceType(str, Enum):
    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"
    CALL_GRAPH = "CALL_GRAPH"
    AST = "AST"
    OTHER = "OTHER"


@dataclass
class NormalizedEvidence:
    file_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    snippet: str
    source_type: SourceType = SourceType.OTHER
    symbol: Optional[str] = None
    relevance: float = 0.0
    is_mock: bool = False  # True => development data, NOT repository evidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        return d


class EvidenceError(ValueError):
    pass


def from_semantic_result(r: dict) -> NormalizedEvidence:
    """Member 2 result dict: file, start_line, end_line, score, code."""
    if not r.get("file") or not r.get("code"):
        raise EvidenceError(f"Malformed semantic result: {sorted(r)}")
    return NormalizedEvidence(
        file_path=str(r["file"]), start_line=r.get("start_line"), end_line=r.get("end_line"),
        snippet=str(r["code"]), source_type=SourceType.SEMANTIC,
        symbol=r.get("name"), relevance=float(r.get("score", 0.0)),
        metadata={"chunk_type": r.get("chunk_type")})


def to_agent_evidence(n: NormalizedEvidence, idx: int, evidence_type: str):
    """Convert to Member 1's Evidence so the existing verifier works unchanged."""
    from backend.agent.verifier import Evidence
    meta = {"normalized": n, "symbol": n.symbol, "is_mock": n.is_mock}
    if n.metadata:
        meta.update(n.metadata)
    return Evidence(
        id=f"ev-{idx:03d}", source_file=n.file_path, line_number=n.start_line,
        content=n.snippet, evidence_type=evidence_type, confidence=max(0.0, min(1.0, n.relevance)),
        metadata=meta)


def from_agent_evidence(e) -> NormalizedEvidence:
    n = e.metadata.get("normalized")
    if n is not None:
        return n
    if not e.source_file or not e.content:
        raise EvidenceError(f"Malformed evidence {e.id}")
    return NormalizedEvidence(e.source_file, e.line_number, e.line_number, e.content,
                              SourceType.OTHER, e.metadata.get("symbol"), e.confidence,
                              bool(e.metadata.get("is_mock")), dict(e.metadata))