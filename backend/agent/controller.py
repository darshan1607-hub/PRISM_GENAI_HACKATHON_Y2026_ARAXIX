"""
Agent Controller for Agentic Code Intelligence.

This module coordinates the complete lifecycle of a code investigation:
question -> planner -> evidence collection -> verifier -> final state.

Architectural Decisions:
------------------------
1. Pluggable Evidence Collector Interface (Tool Injection):
   To avoid hardcoding mock data or concrete external tools into the controller,
   we define the `EvidenceCollector` abstract base class.
   In Phase 3, `MockEvidenceCollector` is provided as default.
   In future phases, concrete tools (FAISS semantic search, Tree-Sitter AST parser,
   Call Graph builder, Gemini reasoning engine) will be injected via this interface
   without modifying the controller logic.

2. Explicit State Machine (InvestigationState):
   The investigation state is encapsulated in a dedicated dataclass that tracks:
   - Current stage (planning, collecting, verifying, completed, requires_more_info)
   - Iteration counter and execution log
   - Generated plan, gathered evidence, and verification verdict
   This facilitates debugging, auditability, and resumption.

3. Grounded Iterative Feedback Loop:
   If the Verifier identifies missing evidence or insufficient criteria,
   the controller records the gap and marks `requires_more_info` or can iterate
   up to `max_iterations` to gather deeper evidence in subsequent phases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.agent.planner import (
    InvestigationPlan,
    InvestigationPlanner,
    InvestigationStep,
    QueryType,
)
from backend.agent.verifier import (
    Evidence,
    InvestigationVerifier,
    VerificationResult,
)


class InvestigationStatus(str, Enum):
    """
    Lifecycle states of an investigation session.
    """
    INITIALIZED = "initialized"
    PLANNING = "planning"
    COLLECTING_EVIDENCE = "collecting_evidence"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    REQUIRES_MORE_INFO = "requires_more_info"
    FAILED = "failed"


@dataclass
class InvestigationState:
    """
    Encapsulates the complete state of a code intelligence investigation.

    Attributes:
        question: The original user question.
        status: Current lifecycle state.
        plan: Structured plan created by the planner.
        evidence: List of evidence items gathered so far.
        verification: Outcome produced by the verifier.
        iteration: Current iteration index.
        max_iterations: Maximum allowed iterative refinement loops.
        execution_log: Chronological list of events/milestones during investigation.
    """
    question: str
    status: InvestigationStatus = InvestigationStatus.INITIALIZED
    plan: Optional[InvestigationPlan] = None
    evidence: List[Evidence] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    iteration: int = 0
    max_iterations: int = 3
    execution_log: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Appends an event to the chronological audit log."""
        self.execution_log.append(message)


class EvidenceCollector(ABC):
    """
    Abstract interface for evidence gathering tools.

    Future phases will implement this interface to connect:
    - Semantic Search (FAISS + embeddings)
    - AST Parsing & Inspection (Tree-Sitter)
    - Call Graph Analysis
    - LLM-assisted Code Reasoning (Gemini)
    """

    @abstractmethod
    def collect(self, step: InvestigationStep, plan: InvestigationPlan) -> List[Evidence]:
        """
        Executes an investigation step and returns collected evidence items.
        """
        pass


class MockEvidenceCollector(EvidenceCollector):
    """
    Mock evidence collector providing deterministic, structured evidence for Phase 3.
    Demonstrates end-to-end execution without external tool dependencies.
    """

    def __init__(self, custom_evidence_map: Optional[Dict[str, List[Evidence]]] = None):
        """
        Optional custom_evidence_map allows unit tests to inject specific scenarios
        (e.g., empty evidence, partial evidence, full evidence).
        """
        self.custom_evidence_map = custom_evidence_map or {}

    def collect(self, step: InvestigationStep, plan: InvestigationPlan) -> List[Evidence]:
        """
        Returns structured mock evidence relevant to the step capability and target.
        """
        target_lower = step.target.lower()

        # Check for user-injected custom scenarios first
        if target_lower in self.custom_evidence_map:
            return self.custom_evidence_map[target_lower]

        # Scenario: "Bluetooth settings deep link" (Prompt requirement)
        if "bluetooth" in target_lower or "deep link" in target_lower:
            if step.tool_capability == "symbol_search":
                return [
                    Evidence(
                        id="ev-001",
                        source_file="src/navigation/DeepLinkRouter.kt",
                        line_number=42,
                        content='const val BLUETOOTH_SETTINGS_URI = "app://settings/bluetooth"',
                        evidence_type="constant_declaration",
                        metadata={
                            "component": "NavigationRouter",
                            "symbol": "BLUETOOTH_SETTINGS_URI",
                            "scheme": "app://",
                        },
                        confidence=1.0,
                    )
                ]
            elif step.tool_capability == "call_graph":
                return [
                    Evidence(
                        id="ev-002",
                        source_file="src/ui/settings/DeviceSettingsFragment.kt",
                        line_number=118,
                        content="binding.btnBluetoothConfig.setOnClickListener { navigateToDeepLink(BLUETOOTH_SETTINGS_URI) }",
                        evidence_type="call_site",
                        metadata={
                            "component": "DeviceSettingsFragment",
                            "caller": "setupClickListeners",
                            "ui_trigger": "btnBluetoothConfig",
                        },
                        confidence=1.0,
                    ),
                    Evidence(
                        id="ev-003",
                        source_file="src/features/quickactions/BluetoothTileService.kt",
                        line_number=75,
                        content='fun onTileClick() { openDeepLink(context, "app://settings/bluetooth") }',
                        evidence_type="call_site",
                        metadata={
                            "component": "BluetoothTileService",
                            "caller": "onTileClick",
                            "ui_trigger": "QuickActionTile",
                        },
                        confidence=0.95,
                    ),
                ]
            elif step.tool_capability == "ast_inspect":
                return [
                    Evidence(
                        id="ev-004",
                        source_file="src/ui/settings/DeviceSettingsFragment.kt",
                        line_number=115,
                        content="private fun setupClickListeners() { ... }",
                        evidence_type="caller_context",
                        metadata={
                            "component": "DeviceSettingsFragment",
                            "enclosing_method": "setupClickListeners",
                            "scope": "private",
                        },
                        confidence=1.0,
                    )
                ]

        # Generic fallback mock evidence for other developer queries
        return [
            Evidence(
                id=f"ev-gen-{step.step_id}",
                source_file="src/core/Manager.kt",
                line_number=50,
                content=f"// Reference to {step.target} found via {step.tool_capability}",
                evidence_type="reference",
                metadata={"step_id": step.step_id, "capability": step.tool_capability},
                confidence=0.8,
            )
        ]


class AgentController:
    """
    Coordinates the investigation workflow from developer question to verified conclusion.

    Lifecycle:
    1. Question -> Planner (generates structured plan)
    2. InvestigationState initialized
    3. Evidence Collection (dispatches steps to EvidenceCollector)
    4. Verifier (evaluates evidence completeness against completion criteria)
    5. Final InvestigationState updated
    """

    def __init__(
        self,
        planner: Optional[InvestigationPlanner] = None,
        verifier: Optional[InvestigationVerifier] = None,
        collector: Optional[EvidenceCollector] = None,
        max_iterations: int = 3,
    ):
        """
        Initializes the AgentController with modular components.
        Components default to Phase 3 baseline implementations if not provided.
        """
        self.planner = planner or InvestigationPlanner()
        self.verifier = verifier or InvestigationVerifier()
        self.collector = collector or MockEvidenceCollector()
        self.max_iterations = max_iterations

    def investigate(self, question: str) -> InvestigationState:
        """
        Executes an end-to-end investigation for a developer question.

        Args:
            question: Natural language question asked by a developer.

        Returns:
            InvestigationState containing the plan, gathered evidence, verification result,
            and lifecycle status.
        """
        state = InvestigationState(
            question=question,
            max_iterations=self.max_iterations,
        )
        state.log(f"Received question: '{question}'")

        try:
            # Stage 1: Planning
            state.status = InvestigationStatus.PLANNING
            state.log("Generating investigation plan...")
            plan = self.planner.plan(question)
            state.plan = plan
            state.log(
                f"Plan generated. Query classified as '{plan.query_type.value}' "
                f"with {len(plan.investigation_steps)} step(s)."
            )

            # Stage 2: Evidence Collection
            state.status = InvestigationStatus.COLLECTING_EVIDENCE
            all_evidence: List[Evidence] = []

            for step in plan.investigation_steps:
                step.status = "in_progress"
                state.log(
                    f"Executing Step {step.step_id} ({step.tool_capability}): {step.description}"
                )
                step_evidence = self.collector.collect(step, plan)
                step.status = "completed"
                all_evidence.extend(step_evidence)
                state.log(f"Step {step.step_id} collected {len(step_evidence)} evidence item(s).")

            state.evidence = all_evidence

            # Stage 3: Verification
            state.status = InvestigationStatus.VERIFYING
            state.log("Verifying collected evidence against plan completion criteria...")
            verification = self.verifier.verify(question, plan, all_evidence)
            state.verification = verification
            state.log(f"Verification complete: sufficient={verification.is_sufficient}")

            # Stage 4: Final State Determination
            state.iteration += 1
            if verification.is_sufficient:
                state.status = InvestigationStatus.COMPLETED
                state.log("Investigation completed successfully with verified evidence.")
            else:
                state.status = InvestigationStatus.REQUIRES_MORE_INFO
                state.log(
                    f"Investigation incomplete. Further investigation required: "
                    f"{verification.requires_further_investigation}. "
                    f"Missing: {', '.join(verification.missing_evidence)}"
                )

        except Exception as e:
            state.status = InvestigationStatus.FAILED
            state.log(f"Investigation failed with error: {str(e)}")
            raise

        return state