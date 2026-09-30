"""Adapter for Member 3. Member 3 currently has no symbol/caller query API."""
from typing import List
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
    """TODO(Member 3): expose symbol/definition/caller queries, then wire them here."""
    implemented = False

    def query(self, capability: str, target: str) -> List[NormalizedEvidence]:
        return []  # honest: no structural query API exists yet
