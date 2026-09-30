"""
Investigation Planner for Agentic Code Intelligence.

This module is responsible for analyzing natural language developer questions,
classifying the developer intent (query type), and generating a structured
investigation plan.

Architectural Decisions:
------------------------
1. Rule-Based Classification (Phase 3):
   To avoid premature complexity and dependencies on LLM APIs or embeddings,
   the current version uses deterministic pattern matching and heuristics.
   The interface is designed so that future phases can seamlessly substitute or
   augment this with an LLM-based planner (e.g., Gemini) without changing downstream contracts.

2. Structured Plan Representation:
   InvestigationPlan produces explicit `information_needed` and `completion_criteria`.
   This is critical for the Verifier module, which checks evidence strictly against
   these criteria to prevent hallucination and detect under-investigated questions.

3. Tool-Agnostic Capability Tags:
   Each InvestigationStep specifies a `tool_capability` (e.g., 'symbol_search',
   'call_graph', 'ast_inspect') rather than binding to specific libraries like Tree-Sitter
   or FAISS directly.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional


class QueryType(str, Enum):
    """
    Standard query categories supported by the agent controller.
    """
    LOCATE_FUNCTIONALITY = "locate functionality"
    FIND_CALLERS = "find callers"
    TRACE_EXECUTION = "trace execution"
    FIND_DEFINITION = "find definition"
    DEPENDENCY_RELATIONSHIP = "dependency/relationship query"
    GENERAL_CODEBASE = "general codebase question"


@dataclass
class InvestigationStep:
    """
    Represents a single granular action within an investigation plan.

    Attributes:
        step_id: Numeric index or identifier of the step.
        description: Human-readable description of what this step achieves.
        tool_capability: The type of tool capability needed (e.g., 'symbol_search', 'call_graph').
        target: The primary entity, symbol, or term targeted by this step.
        parameters: Optional dictionary of parameters passed to the tool.
        status: Current status of the step ('pending', 'in_progress', 'completed', 'skipped').
    """
    step_id: int
    description: str
    tool_capability: str
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"


@dataclass
class InvestigationPlan:
    """
    Structured investigation plan generated for a developer question.

    Attributes:
        original_question: The original user question text.
        query_type: Classified query type.
        investigation_steps: Ordered sequence of investigation steps to execute.
        information_needed: List of required evidence items needed to answer the question.
        completion_criteria: List of conditions that must be satisfied for the investigation to terminate.
        target_subject: Extracted core entity/subject of the question.
        rationale: Brief explanation of why this plan structure was chosen.
    """
    original_question: str
    query_type: QueryType
    investigation_steps: List[InvestigationStep]
    information_needed: List[str]
    completion_criteria: List[str]
    target_subject: str = ""
    rationale: str = ""


class InvestigationPlanner:
    """
    Analyzes developer queries and creates structured investigation plans.
    """

    # Keyword patterns mapped to query types
    _CALLER_PATTERNS = [
        r"\b(?:who\s+calls|callers\s+of|usages\s+of|references\s+to|where\s+.*?\s+used|used\s+in|invocations\s+of)\b",
        r"\b(?:find\s+callers|find\s+usages|find\s+references)\b",
    ]
    _DEFINITION_PATTERNS = [
        r"\b(?:where\s+is\s+.*?\s+(?:defined|declared)|definition\s+of|declaration\s+of|where\s+is\s+the\s+definition)\b",
        r"\b(?:class\s+definition|interface\s+definition|function\s+signature)\b",
    ]
    _EXECUTION_TRACE_PATTERNS = [
        r"\b(?:trace|execution\s+flow|call\s+flow|call\s+hierarchy|call\s+stack|how\s+does\s+.*?\s+execute|how\s+does\s+.*?\s+work)\b",
        r"\b(?:sequence\s+of\s+events|step\s+by\s+step\s+flow)\b",
    ]
    _LOCATE_PATTERNS = [
        r"\b(?:where\s+is\s+.*?\s+(?:located|implemented)|where\s+do\s+we\b|how\s+to\b|locate\s+the\s+implementation)\b",
        r"\b(?:find\s+the\s+code\s+for|where\s+can\s+i\s+find)\b",
    ]
    _DEPENDENCY_PATTERNS = [
        r"\b(?:depend(?:s|ency|encies)?|import(?:s)?|relationship\s+between|coupled\s+with|package\s+structure)\b",
        r"\b(?:what\s+does\s+.*?\s+depend\s+on|who\s+depends\s+on)\b",
    ]

    def classify_query(self, question: str) -> QueryType:
        """
        Classifies the question into one of the supported QueryType categories.
        Uses deterministic regular expression matching with prioritized order.
        """
        q = question.strip().lower()

        # Check for callers/usages first as "where is X used?" is a common specific pattern
        for pattern in self._CALLER_PATTERNS:
            if re.search(pattern, q):
                return QueryType.FIND_CALLERS

        # Check definition queries
        for pattern in self._DEFINITION_PATTERNS:
            if re.search(pattern, q):
                return QueryType.FIND_DEFINITION

        # Check execution trace queries
        for pattern in self._EXECUTION_TRACE_PATTERNS:
            if re.search(pattern, q):
                return QueryType.TRACE_EXECUTION

        # Check dependency queries
        for pattern in self._DEPENDENCY_PATTERNS:
            if re.search(pattern, q):
                return QueryType.DEPENDENCY_RELATIONSHIP

        # Check locate functionality queries
        for pattern in self._LOCATE_PATTERNS:
            if re.search(pattern, q):
                return QueryType.LOCATE_FUNCTIONALITY

        return QueryType.GENERAL_CODEBASE

    def extract_subject(self, question: str) -> str:
        """
        Extracts the primary subject/entity from a natural language question.
        For example: "Where is the Bluetooth settings deep link used?" -> "Bluetooth settings deep link"
        """
        cleaned = question.strip().rstrip("?.!")

        # Common prefix patterns in code investigation questions
        prefix_patterns = [
            r"^(?:where\s+is\s+(?:the\s+)?|who\s+calls\s+(?:the\s+)?|find\s+callers\s+of\s+(?:the\s+)?|usages\s+of\s+(?:the\s+)?|references\s+to\s+(?:the\s+)?)",
            r"^(?:definition\s+of\s+(?:the\s+)?|where\s+can\s+i\s+find\s+(?:the\s+)?|locate\s+(?:the\s+)?)",
            r"^(?:how\s+does\s+(?:the\s+)?|what\s+does\s+(?:the\s+)?|trace\s+(?:the\s+)?)",
        ]
        subject = cleaned
        for p in prefix_patterns:
            match = re.search(p, subject, re.IGNORECASE)
            if match:
                subject = subject[match.end():]
                break

        # Remove trailing usage clauses like "used", "located", "defined", "implemented"
        suffix_patterns = [
            r"\s+(?:defined\s+and\s+who\s+calls\s+it|and\s+who\s+calls\s+it|flow\s+work\s+and\s+what\s+functions\s+are\s+involved|work\s+and\s+what\s+functions\s+are\s+involved|flow\s+work|used|located|defined|declared|implemented|called|flow|work)(?:\s+in\s+the\s+codebase)?$",
        ]
        for p in suffix_patterns:
            subject = re.sub(p, "", subject, flags=re.IGNORECASE)

        subject = subject.strip()
        return subject if subject else cleaned

    def plan(self, question: str) -> InvestigationPlan:
        """
        Analyzes a developer question and produces a structured investigation plan.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_type = self.classify_query(question)
        subject = self.extract_subject(question)

        steps: List[InvestigationStep] = []
        info_needed: List[str] = []
        criteria: List[str] = []
        rationale: str = ""

        if query_type == QueryType.FIND_CALLERS:
            rationale = (
                f"The question seeks call sites and usages for '{subject}'. "
                "The plan first identifies target symbols and definitions, then locates all "
                "referencing call sites and inspects their invocation contexts."
            )
            steps = [
                InvestigationStep(
                    step_id=1,
                    description=f"Identify declaration and canonical symbol names for '{subject}'",
                    tool_capability="symbol_search",
                    target=subject,
                    parameters={"match_mode": "exact_or_identifier"},
                ),
                InvestigationStep(
                    step_id=2,
                    description=f"Find all call sites, references, and usages of '{subject}' across codebase",
                    tool_capability="call_graph",
                    target=subject,
                    parameters={"include_indirect_references": True},
                ),
                InvestigationStep(
                    step_id=3,
                    description=f"Inspect caller contexts (surrounding functions, classes, UI handlers)",
                    tool_capability="ast_inspect",
                    target=subject,
                    parameters={"context_lines": 5},
                ),
            ]
            info_needed = [
                f"Canonical symbol/constant definition for '{subject}'",
                f"File paths and line numbers of all call sites/references for '{subject}'",
                f"Caller enclosing functions or UI components executing the call",
            ]
            criteria = [
                f"At least one confirmed reference or call site found with exact file path and line number, or verified absence across the repository.",
                f"Usage context documented (component, caller function, or trigger mechanism).",
            ]

        elif query_type == QueryType.FIND_DEFINITION:
            rationale = f"Locate the declaration and definition signature of '{subject}'."
            steps = [
                InvestigationStep(
                    step_id=1,
                    description=f"Search for definition/declaration of '{subject}'",
                    tool_capability="symbol_search",
                    target=subject,
                ),
                InvestigationStep(
                    step_id=2,
                    description=f"Inspect AST structure and documentation docstrings for '{subject}'",
                    tool_capability="ast_inspect",
                    target=subject,
                ),
            ]
            info_needed = [
                f"Exact file path and line number defining '{subject}'",
                f"Symbol type (class, function, variable, constant) and signature",
            ]
            criteria = [
                f"Definition location (file path and line number) conclusively identified.",
            ]

        elif query_type == QueryType.TRACE_EXECUTION:
            rationale = f"Trace the execution flow starting from or passing through '{subject}'."
            steps = [
                InvestigationStep(
                    step_id=1,
                    description=f"Identify entry point for '{subject}'",
                    tool_capability="symbol_search",
                    target=subject,
                ),
                InvestigationStep(
                    step_id=2,
                    description=f"Trace downstream invocations and control flow",
                    tool_capability="call_graph",
                    target=subject,
                ),
                InvestigationStep(
                    step_id=3,
                    description=f"Retrieve semantic execution context for '{subject}'",
                    tool_capability="semantic_search",
                    target=subject,
                ),
            ]
            info_needed = [
                f"Entry point for '{subject}'",
                f"Ordered sequence of function invocations",
            ]
            criteria = [
                f"Complete execution path mapped from trigger to destination.",
            ]

        elif query_type == QueryType.LOCATE_FUNCTIONALITY:
            rationale = f"Find code modules implementing functionality related to '{subject}'."
            steps = [
                InvestigationStep(
                    step_id=1,
                    description=f"Search codebase for keywords or semantic concepts related to '{subject}'",
                    tool_capability="semantic_search",
                    target=subject,
                ),
                InvestigationStep(
                    step_id=2,
                    description=f"Inspect candidate modules for primary feature implementation",
                    tool_capability="ast_inspect",
                    target=subject,
                ),
            ]
            info_needed = [
                f"Primary module or package implementing '{subject}'",
                f"Core classes or entry points handling the feature",
            ]
            criteria = [
                f"Feature implementation source files and entry functions identified.",
            ]

        elif query_type == QueryType.DEPENDENCY_RELATIONSHIP:
            rationale = f"Map dependencies and import relationships for '{subject}'."
            steps = [
                InvestigationStep(
                    step_id=1,
                    description=f"Analyze import statements and module dependencies for '{subject}'",
                    tool_capability="ast_inspect",
                    target=subject,
                ),
            ]
            info_needed = [
                f"List of imported modules and external dependencies for '{subject}'",
            ]
            criteria = [
                f"Dependency graph or import manifest established for '{subject}'.",
            ]

        else:  # GENERAL_CODEBASE
            rationale = f"Broad codebase investigation for question: '{question}'."
            steps = [
                InvestigationStep(
                    step_id=1,
                    description=f"Perform semantic search for keywords in '{question}'",
                    tool_capability="semantic_search",
                    target=subject,
                ),
            ]
            info_needed = [
                f"Relevant code snippets addressing '{question}'",
            ]
            criteria = [
                f"Relevant code references identified with source locations.",
            ]

        return InvestigationPlan(
            original_question=question,
            query_type=query_type,
            investigation_steps=steps,
            information_needed=info_needed,
            completion_criteria=criteria,
            target_subject=subject,
            rationale=rationale,
        )