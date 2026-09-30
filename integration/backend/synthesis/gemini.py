"""Grounded Gemini synthesis. Key comes from env only."""
import os
from typing import Tuple

SYSTEM = """You are a code investigation assistant. Answer ONLY from the supplied evidence.
Do not invent code, files, or line numbers. If evidence is insufficient, say so explicitly.
Label statements as EVIDENCE (directly in a snippet, cite file:line) or INFERENCE.
Be concise and developer-oriented."""


def build_prompt(question, evidence, verification) -> str:
    parts = [SYSTEM, f"\nQUESTION: {question}\n",
             f"VERIFICATION: {'sufficient' if verification and verification.is_sufficient else 'INSUFFICIENT'}"]
    if verification and verification.missing_evidence:
        parts.append("MISSING: " + "; ".join(verification.missing_evidence))
    if not evidence:
        parts.append("EVIDENCE: none")
    for i, e in enumerate(evidence, 1):
        parts.append(f"\n[{i}] {e.file_path}:{e.start_line}-{e.end_line} ({e.source_type.value})\n{e.snippet}")
    return "\n".join(parts)


class Synthesizer:
    def __init__(self, mock=True, model=None):
        self.mock = mock
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def _call(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return client.models.generate_content(model=self.model, contents=prompt).text or ""

    def synthesize(self, question, evidence, verification) -> Tuple[str, str]:
        """Returns (answer, error). Never raises."""
        if not evidence:
            return ("No evidence was found in the repository for this question, so no grounded answer "
                    "can be given.", "")
        if self.mock:
            return ("[DEVELOPMENT MODE] Gemini not called. The evidence below is MOCK data, "
                    "not from a real repository.", "")
        if not os.environ.get("GEMINI_API_KEY"):
            return ("Answer generation unavailable (GEMINI_API_KEY not set). Raw evidence is shown below.",
                    "Gemini synthesis skipped: GEMINI_API_KEY not set.")
        try:
            return self._call(build_prompt(question, evidence, verification)), ""
        except Exception as e:
            return ("Answer generation failed. Raw evidence is shown below.", f"Gemini synthesis failed: {e}")
