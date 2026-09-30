"""IntegrationService: the only thing FastAPI talks to."""
import os
from backend.agent import AgentController
from backend.integration.agent_adapter import IntegrationCollector
from backend.integration.evidence import from_agent_evidence, EvidenceError
from backend.integration.semantic_adapter import MockSemanticAdapter, RealSemanticAdapter
from backend.integration.structural_adapter import MockStructuralAdapter, RealStructuralAdapter
from backend.synthesis.gemini import Synthesizer


def use_mock() -> bool:
    return os.environ.get("USE_MOCK_SERVICES", "true").lower() == "true"


class IntegrationService:
    def __init__(self, mock=None, repo_path=None, synthesizer=None):
        self.mock = use_mock() if mock is None else mock
        self.repo_path = repo_path or os.environ.get("REPO_PATH", "demo_repo")
        self.synth = synthesizer or Synthesizer(mock=self.mock)
        self._semantic = None if self.mock else RealSemanticAdapter(self.repo_path)

    def repository_status(self) -> dict:
        return {"mode": "MOCK" if self.mock else "REAL", "repo_path": None if self.mock else self.repo_path,
                "structural_query_api": self.mock}

    def investigate(self, question: str) -> dict:
        semantic = MockSemanticAdapter() if self.mock else self._semantic
        structural = MockStructuralAdapter() if self.mock else RealStructuralAdapter()
        collector = IntegrationCollector(semantic, structural)
        state = AgentController(collector=collector).investigate(question)

        evidence, errors = [], list(collector.errors)
        for e in state.evidence:
            try:
                evidence.append(from_agent_evidence(e))
            except EvidenceError as ex:
                errors.append(str(ex))
        if not self.mock and not RealStructuralAdapter.implemented:
            errors.append("Structural analysis unavailable: Member 3 query API not implemented.")

        v = state.verification
        answer, synth_error = self.synth.synthesize(question, evidence, v)
        if synth_error:
            errors.append(synth_error)
        return {
            "question": question, "answer": answer, "mode": "MOCK" if self.mock else "REAL",
            "evidence": [e.to_dict() for e in evidence],
            "investigation": {
                "query_type": state.plan.query_type.value if state.plan else None,
                "steps": [{"id": s.step_id, "description": s.description, "capability": s.tool_capability,
                           "status": s.status} for s in (state.plan.investigation_steps if state.plan else [])],
                "sources_used": sorted(set(collector.sources_used)),
                "status": state.status.value,
                "verification_status": "sufficient" if v and v.is_sufficient else "insufficient",
                "confidence": v.confidence_score if v else 0.0,
                "missing_evidence": v.missing_evidence if v else [],
                "log": state.execution_log,
            },
            "errors": errors,
        }
