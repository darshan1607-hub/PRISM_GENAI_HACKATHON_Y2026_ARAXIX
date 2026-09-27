"""
Demonstration script for Agentic Code Intelligence (Phase 3).

Demonstrates the initial Agent Controller architecture end-to-end:
Question -> Plan -> Mock Evidence -> Verification -> Final State

Demonstrates the target question:
"Where is the Bluetooth settings deep link used?"
"""

import sys
import os

# Ensure the root project directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.agent.controller import (
    AgentController,
    MockEvidenceCollector,
    InvestigationStatus,
)
from backend.agent.verifier import Evidence


def print_separator(char="=", length=78):
    print(char * length)


def print_investigation_results(state):
    """
    Prints the required outputs cleanly:
    - question
    - query type
    - investigation plan
    - evidence found
    - verification result
    - whether further investigation is required
    """
    print_separator("=")
    print("           AGENTIC CODE INTELLIGENCE - INVESTIGATION REPORT")
    print_separator("=")

    # 1. Question
    print(f"\n[1] QUESTION:")
    print(f"    \"{state.question}\"")

    # 2. Query Type
    plan = state.plan
    query_type_val = plan.query_type.value if plan else "unknown"
    print(f"\n[2] QUERY TYPE:")
    print(f"    Category: {query_type_val.upper()}")
    if plan and plan.target_subject:
        print(f"    Identified Target Subject: '{plan.target_subject}'")

    # 3. Investigation Plan
    print(f"\n[3] INVESTIGATION PLAN:")
    if plan:
        print(f"    Rationale: {plan.rationale}")
        print("    Investigation Steps:")
        for step in plan.investigation_steps:
            print(f"      [{step.step_id}] ({step.tool_capability}) {step.description}")
        print("    Information Needed:")
        for item in plan.information_needed:
            print(f"      - {item}")
        print("    Completion Criteria:")
        for crit in plan.completion_criteria:
            print(f"      * {crit}")
    else:
        print("    No plan generated.")

    # 4. Evidence Found
    print(f"\n[4] EVIDENCE FOUND ({len(state.evidence)} items):")
    if state.evidence:
        for idx, ev in enumerate(state.evidence, 1):
            loc = f"{ev.source_file}:{ev.line_number}" if ev.line_number else ev.source_file
            print(f"    Item #{idx}: [{ev.evidence_type}] @ {loc}")
            print(f"      Code snippet: {ev.content}")
            if ev.metadata:
                meta_strs = [f"{k}={v}" for k, v in ev.metadata.items()]
                print(f"      Metadata: {', '.join(meta_strs)}")
            print(f"      Confidence: {ev.confidence}")
    else:
        print("    No evidence found.")

    # 5. Verification Result
    print(f"\n[5] VERIFICATION RESULT:")
    ver = state.verification
    if ver:
        print(f"    Status: {'SUFFICIENT' if ver.is_sufficient else 'INSUFFICIENT'}")
        print(f"    Summary: {ver.summary}")
        print(f"    Confidence Score: {ver.confidence_score}")
        if ver.verified_claims:
            print("    Verified Claims:")
            for claim in ver.verified_claims:
                print(f"      [VERIFIED] {claim}")
        if ver.missing_evidence:
            print("    Missing Evidence:")
            for missing in ver.missing_evidence:
                print(f"      [MISSING] {missing}")
        if ver.unverified_claims:
            print("    Unverified Claims:")
            for unv in ver.unverified_claims:
                print(f"      [UNVERIFIED] {unv}")
    else:
        print("    No verification executed.")

    # 6. Whether Further Investigation is Required
    print(f"\n[6] WHETHER FURTHER INVESTIGATION IS REQUIRED:")
    req = ver.requires_further_investigation if ver else True
    print(f"    Requires Further Investigation: {req}")
    print(f"    Final Controller Lifecycle Status: {state.status.value.upper()}")
    print_separator("=")


def run_primary_demo():
    print("\nRunning Primary Scenario: Target Question")
    question = "Where is the Bluetooth settings deep link used?"
    controller = AgentController()
    state = controller.investigate(question)
    print_investigation_results(state)


def run_missing_evidence_demo():
    print("\n\nRunning Secondary Scenario: Unanswered Query with Missing Evidence")
    question = "Where is the QuantumEncryptionToken used?"
    
    # Custom mock collector that returns only a declaration but NO call sites
    partial_collector = MockEvidenceCollector(
    custom_evidence_map={
        "quantumencryptiontoken": [
            Evidence(
                id="ev-missing-1",
                source_file="src/security/Tokens.kt",
                line_number=12,
                content="const val QuantumEncryptionToken = 'SEC_QET_TOKEN'",
                evidence_type="constant_declaration",
                metadata={"component": "Security"},
                confidence=1.0,
                )
            ]
        }
    )
    controller = AgentController(collector=partial_collector)
    state = controller.investigate(question)
    print_investigation_results(state)


if __name__ == "__main__":
    run_primary_demo()
    run_missing_evidence_demo()