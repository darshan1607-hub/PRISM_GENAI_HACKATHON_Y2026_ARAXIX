import pytest
from fastapi.testclient import TestClient
from backend.api import main
from backend.integration.service import IntegrationService
from backend.integration.evidence import from_semantic_result, EvidenceError, SourceType
from backend.synthesis.gemini import Synthesizer

Q = "Where is the Bluetooth settings deep link used?"


@pytest.fixture
def client():
    main._service = IntegrationService(mock=True)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("q", ["", "   "])
def test_validation(client, q):
    assert client.post("/api/investigate", json={"question": q}).status_code == 422


def test_normalization():
    e = from_semantic_result({"file": "a.js", "start_line": 1, "end_line": 2, "score": .5, "code": "x"})
    assert e.source_type == SourceType.SEMANTIC and not e.is_mock
    with pytest.raises(EvidenceError):
        from_semantic_result({"file": "a.js"})


def test_mock_flow_marks_mock(client):
    r = client.post("/api/investigate", json={"question": Q}).json()
    assert r["mode"] == "MOCK" and r["evidence"] and all(e["is_mock"] for e in r["evidence"])
    assert "DEVELOPMENT MODE" in r["answer"]
    assert r["investigation"]["verification_status"] == "sufficient"


def test_gemini_failure_keeps_evidence(monkeypatch):
    s = Synthesizer(mock=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(s, "_call", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    r = IntegrationService(mock=True, synthesizer=s).investigate(Q)
    assert r["evidence"] and any("Gemini synthesis failed" in e for e in r["errors"])


def test_empty_evidence():
    ans, err = Synthesizer(mock=False).synthesize(Q, [], None)
    assert "No evidence" in ans and not err


def test_real_mode_bad_repo_reports_error():
    r = IntegrationService(mock=False, repo_path="/nonexistent").investigate(Q)
    assert r["evidence"] == [] and r["errors"] and r["investigation"]["verification_status"] == "insufficient"
