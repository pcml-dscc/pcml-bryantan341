# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 1: Schema-Constrained Extraction

Complete the `solve()` function. Read problem.md for the full specification.
Drive a local Ollama LLM with a Kaizen Signature (type-safe structured output)
at temperature 0, and extract one structured record per incident report.

Your submission is auto-graded on schema compliance + field accuracy.
"""
from __future__ import annotations

import asyncio

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL

# ════════════════════════════════════════════════════════════════════════
# FIXED CORPUS — six SG last-mile logistics incident reports (given).
# ════════════════════════════════════════════════════════════════════════
INCIDENT_REPORTS: list[str] = [
    (
        "Incident Report INC-3001\n"
        "Severity: HIGH. Location: Tuas Checkpoint.\n"
        "A container truck overturned during transfer. 42 parcels affected. "
        "An insurance claim is required for the damaged goods."
    ),
    (
        "Incident Report INC-3002\n"
        "Severity: LOW. Location: Changi Airfreight Centre.\n"
        "A scanning belt jammed briefly. 3 parcels affected. "
        "No insurance claim is needed."
    ),
    (
        "Incident Report INC-3003\n"
        "Severity: MEDIUM. Location: Jurong Port.\n"
        "A forklift clipped a pallet stack. 17 parcels affected. "
        "An insurance claim is required."
    ),
    (
        "Incident Report INC-3004\n"
        "Severity: HIGH. Location: Woodlands Checkpoint.\n"
        "A refrigeration unit failed in transit. 58 parcels affected. "
        "An insurance claim is required for the spoiled shipment."
    ),
    (
        "Incident Report INC-3005\n"
        "Severity: LOW. Location: Pasir Panjang Terminal.\n"
        "A label printer ran out of ink. 1 parcel affected. "
        "No insurance claim is needed."
    ),
    (
        "Incident Report INC-3006\n"
        "Severity: MEDIUM. Location: Tampines Logistics Hub.\n"
        "A delivery rider was rerouted by road closures. 9 parcels affected. "
        "No insurance claim is needed."
    ),
]


# TODO 1: Define a Kaizen Signature `IncidentExtraction` with one InputField
#         (report_text: str) and five OutputFields with these names and types:
#           incident_id: str, severity: str (one of low/medium/high),
#           location: str, parcels_affected: int, claim_required: bool
#         The OutputField descriptions are what steer the LLM — write them well.
class IncidentExtraction(Signature):
    """Extract structured fields from a last-mile logistics incident report."""

    report_text: str = InputField(
        description="The full logistics incident report to extract structured information from."
    )

    incident_id: str = OutputField(
        description="The incident reference ID exactly as written, e.g. INC-3001."
    )

    severity: str = OutputField(
        description="The severity level. Return exactly one of: low, medium, or high."
    )

    location: str = OutputField(
        description="The facility or location where the incident occurred."
    )

    parcels_affected: int = OutputField(
        description="The integer number of parcels affected by the incident."
    )

    claim_required: bool = OutputField(
        description="Return True if the report states an insurance claim is required, otherwise False."
    )


def _make_agent() -> BaseAgent:
    # TODO 2: Build a BaseAgent subclass backed by IncidentExtraction(), wired
    #         to Ollama: config={"model": DEFAULT_CHAT_MODEL,
    #         "llm_provider": "ollama", "base_url": OLLAMA_BASE_URL,
    #         "use_async_llm": True, "temperature": 0.0}.

    class IncidentAgent(BaseAgent):
        signature = IncidentExtraction()

        def __init__(self):
            super().__init__(
                config={
                    "model": DEFAULT_CHAT_MODEL,
                    "llm_provider": "ollama",
                    "base_url": OLLAMA_BASE_URL,
                    "use_async_llm": True,
                    "temperature": 0.0,
                }
            )

    return IncidentAgent()


async def _extract_all() -> list[dict]:
    results: list[dict] = []

    agent = _make_agent()

    for report in INCIDENT_REPORTS:
        # TODO 3: Run the agent on each report with `await agent.run_async(
        #         report_text=report)` and collect the five fields into a dict.

        response = await agent.run_async(report_text=report)

        results.append(
    {
        "incident_id": response["incident_id"],
        "severity": response["severity"],
        "location": response["location"],
        "parcels_affected": response["parcels_affected"],
        "claim_required": response["claim_required"],
    }
)

    return results


def solve() -> list[dict]:
    """Return a list of six dicts, one structured record per incident report.

    Each dict must have keys: incident_id, severity, location,
    parcels_affected, claim_required.
    """
    return asyncio.run(_extract_all())


if __name__ == "__main__":
    for rec in solve():
        print(rec)