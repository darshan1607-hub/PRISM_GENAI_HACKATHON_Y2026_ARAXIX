"""Plugs semantic + structural adapters into Member 1's EvidenceCollector hook."""
from backend.agent.controller import EvidenceCollector
from backend.integration.evidence import to_agent_evidence

_EVTYPE = {"symbol_search": "declaration", "call_graph": "call_site", "ast_inspect": "caller_context"}


class IntegrationCollector(EvidenceCollector):
    def __init__(self, semantic, structural):
        self.semantic, self.structural = semantic, structural
        self.errors: list = []
        self.sources_used: list = []
        self._seen: set = set()
        self._n = 0

    def collect(self, step, plan):
        cap = step.tool_capability
        try:
            items = (self.semantic.search(step.target) if cap == "semantic_search"
                     else self.structural.query(cap, step.target))
        except Exception as e:  # keep investigation alive; surface the error
            self.errors.append(f"step {step.step_id} ({cap}): {e}")
            return []
        self.sources_used.append(f"{'semantic' if cap == 'semantic_search' else 'structural'}:{cap}")
        out = []
        for n in items:
            key = (n.file_path, n.start_line, n.snippet)
            if key in self._seen:
                continue
            self._seen.add(key)
            self._n += 1
            m_type = n.metadata.get("match_type")
            ev_type = m_type if m_type in ("call_site", "declaration", "definition", "reference", "caller_context") else _EVTYPE.get(cap, "text_match")
            out.append(to_agent_evidence(n, self._n, ev_type))
        return out