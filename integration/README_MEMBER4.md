# Member 4 — Product & Integration

    React (Vite) → POST /api/investigate → FastAPI → IntegrationService
      → AgentController (Member 1, unchanged) ← IntegrationCollector
           ├─ semantic_search → RealSemanticAdapter → Member 2 SemanticCodeSearch
           └─ symbol_search / call_graph / ast_inspect → StructuralAdapter → Member 3
      → InvestigationVerifier (Member 1) → Gemini Synthesizer → JSON

Existing Member 1–3 files are not modified.

## Layout
backend/integration/{evidence,semantic_adapter,structural_adapter,agent_adapter,service}.py,
backend/synthesis/gemini.py, backend/api/main.py, tests/test_member4.py, frontend/

## Setup
    pip install -r requirements.txt -r requirements-member4.txt
    cp .env.example .env        # export vars; never commit .env
    cd frontend && npm install

## Run
    USE_MOCK_SERVICES=true uvicorn backend.api.main:app --reload --port 8000
    cd frontend && npm run dev          # http://localhost:5173
    pytest tests/test_member4.py

## Env vars
USE_MOCK_SERVICES (default true), REPO_PATH, GEMINI_API_KEY (server only), GEMINI_MODEL.

## Mock mode
Mock evidence is prefixed `[MOCK]`, flagged `is_mock: true`, and the UI shows a warning banner.
Real mode (`USE_MOCK_SERVICES=false`) needs GEMINI_API_KEY and REPO_PATH.

## API
GET /health · GET /api/repository · POST /api/investigate `{"question": "..."}`
→ `{question, answer, mode, evidence[], investigation{}, errors[]}`

## Troubleshooting
- "Backend unreachable": start uvicorn on :8000.
- `ModuleNotFoundError: backend.structural`: Member 2/3 packages must be on the path (run from project root).
- Real mode shows "Structural analysis unavailable": Member 3 query API is not yet implemented.
