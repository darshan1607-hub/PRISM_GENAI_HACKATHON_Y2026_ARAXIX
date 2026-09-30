"""FastAPI application entry point for Agentic Code Intelligence."""
import logging
import os
import sys
from pathlib import Path

# Bootstrap sys.path to include repo root and member components
_FILE_PATH = Path(__file__).resolve()
_REPO_ROOT = _FILE_PATH.parents[3]
for _p in [
    str(_REPO_ROOT),
    str(_REPO_ROOT / "integration"),
    str(_REPO_ROOT / "semantics"),
    str(_REPO_ROOT / "structural"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pkgutil
import backend
backend.__path__ = pkgutil.extend_path(backend.__path__, backend.__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from backend.integration.service import IntegrationService

log = logging.getLogger("api")
app = FastAPI(title="Agentic Code Intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
                   allow_methods=["*"], allow_headers=["*"])
_service = None


def get_service() -> IntegrationService:
    global _service
    if _service is None:
        _service = IntegrationService()
    return _service


class InvestigateRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Question too long (max 2000 chars).")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/repository")
def repository():
    return get_service().repository_status()


@app.post("/api/investigate")
def investigate(req: InvestigateRequest):
    try:
        return get_service().investigate(req.question)
    except Exception as e:
        log.exception("investigation failed")
        raise HTTPException(status_code=500, detail=f"Investigation failed: {type(e).__name__}")