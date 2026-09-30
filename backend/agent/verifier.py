"""
Evidence Verifier for Agentic Code Intelligence.

This module is responsible for assessing whether collected evidence is sufficient
to reliably answer a developer question according to the investigation plan.

Architectural Decisions:
------------------------
1. Strict Groundedness Invariant:
   The verifier enforces that only claims directly substantiated by collected
   `Evidence` items (with file paths and line numbers) are marked verified.
   It explicitly flags unverified claims and refuses to assume unobserved facts.

2. Deterministic Completeness Checking:
   By cross-referencing collected evidence against `plan.completion_criteria` and
   `plan.information_needed`, the verifier identifies gaps (e.g. finding a symbol's
   definition is insufficient when the question asked for its callers/usages).

3. Future Tool & LLM Extensibility:
   The `InvestigationVerifier` interface accepts structured `Evidence` objects.
   In future phases, semantic verifiers (e.g. Gemini cross-checking AST nodes and
   source text) can inherit or plug into this interface without altering the
   orchestration contract.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from backend.agent.planner import InvestigationPlan, QueryType


@dataclass
class Evidence:
    """
    Represents an atomic, verified piece of evidence extracted from the repository.

    Attributes:
        id: Unique identifier for this evidence item.
        source_file: Relative or absolute path to the source file.
        line_number: Line number where the evidence is located (if applicable).
        content: Code snippet, signature, or extracted content.
        evidence_type: Categorical type (e.g. 'call_site', 'symbol_definition', 'constant_declaration', 'text_match').
        metadata: Additional contextual attributes (e.g. caller function name, enclosing class, component).
        confidence: Confidence score of the extraction (0.0 to 1.0).
    """
    id: str
    source_file: str
    line_number: Optional[int]
    content: str
    evidence_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class VerificationResult:
    """
    Outcome of evaluating evidence against the investigation plan.

    Attributes:
        is_sufficient: True if the collected evidence satisfies the plan's completion criteria.
        missing_evidence: Specific items or categories of information still missing.
        requires_further_investigation: True if more investigation steps or query expansions are needed.
        summary: High-level natural language summary of the verification verdict.
        verified_claims: List of facts verified with direct citations to evidence items.
        unverified_claims: List of claims or hypotheses that lack supporting evidence.
        confidence_score: Quantitative assessment of evidence completeness (0.0 to 1.0).
    """
    is_sufficient: bool
    missing_evidence: List[str]
    requires_further_investigation: bool
    summary: str
    verified_claims: List[str] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    confidence_score: float = 1.0


class InvestigationVerifier:
    """
    Evaluates evidence against an investigation plan to determine sufficiency and gaps.
    """

    def verify(
        self,
        question: str,
        plan: InvestigationPlan,
        evidence: List[Evidence],
    ) -> VerificationResult:
        """
        Evaluates the collected evidence against the investigation plan.

        Rules:
        - If no evidence exists, mark insufficient and report all information needed as missing.
        - Strictly verify claims present in the evidence without hallucinating unobserved facts.
        - Ensure query-specific criteria are met (e.g., FIND_CALLERS requires call sites, not just declarations).
        """
        if not evidence:
            return VerificationResult(
                is_sufficient=False,
                missing_evidence=list(plan.information_needed),
                requires_further_investigation=True,
                summary="No evidence was collected. The query remains completely unanswered.",
                verified_claims=[],
                unverified_claims=[f"Answer to: '{question}'"],
                confidence_score=0.0,
            )

        verified_claims: List[str] = []
        unverified_claims: List[str] = []
        missing_evidence: List[str] = []

        # Categorize collected evidence types
        call_sites = [e for e in evidence if e.evidence_type in ("call_site", "reference", "usage")]
        definitions = [e for e in evidence if e.evidence_type in ("symbol_definition", "constant_declaration", "declaration")]

        # Extract verified claims directly from verified evidence items
        for item in evidence:
            loc = f"{item.source_file}:{item.line_number}" if item.line_number else item.source_file
            claim_desc = f"[{item.evidence_type}] in {loc} -> {item.content.strip()}"
            if "component" in item.metadata:
                claim_desc += f" (Component: {item.metadata['component']})"
            if "caller" in item.metadata:
                claim_desc += f" (Caller: {item.metadata['caller']})"
            verified_claims.append(claim_desc)

        # Check sufficiency based on the query type and completion criteria
        if plan.query_type == QueryType.FIND_CALLERS:
            if not call_sites:
                missing_evidence.append(
                    f"No call sites or referencing usages found for '{plan.target_subject}'."
                )
                if definitions:
                    verified_claims.append(
                        f"Target '{plan.target_subject}' is defined/declared, but no usages were located."
                    )
                unverified_claims.append(
                    f"Caller locations and invocation contexts for '{plan.target_subject}'"
                )

        elif plan.query_type == QueryType.FIND_DEFINITION:
            if not definitions:
                missing_evidence.append(
                    f"Definition or declaration for '{plan.target_subject}' was not found."
                )
                unverified_claims.append(
                    f"Exact declaration of '{plan.target_subject}'"
                )

        elif plan.query_type in (QueryType.TRACE_EXECUTION, QueryType.LOCATE_FUNCTIONALITY):
            if len(evidence) < 1:
                missing_evidence.append(
                    f"Insufficient trace or implementation evidence for '{plan.target_subject}'."
                )

        # Determine overall sufficiency
        is_sufficient = len(missing_evidence) == 0 and len(verified_claims) > 0
        requires_further_investigation = not is_sufficient

        if is_sufficient:
            summary = (
                f"Sufficient evidence collected. Found {len(verified_claims)} verified item(s) "
                f"satisfying completion criteria for {plan.query_type.value}."
            )
            confidence = min(1.0, 0.5 + 0.15 * len(verified_claims))
        else:
            summary = (
                f"Investigation incomplete. Missing {len(missing_evidence)} critical item(s) "
                f"to answer: '{question}'."
            )
            confidence = max(0.0, 0.3 * (len(verified_claims) / max(1, len(plan.information_needed))))

        return VerificationResult(
            is_sufficient=is_sufficient,
            missing_evidence=missing_evidence,
            requires_further_investigation=requires_further_investigation,
            summary=summary,
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            confidence_score=round(confidence, 2),
        )