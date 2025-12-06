from __future__ import annotations

from typing import List

from .types import Intent


def classify_intent(question: str) -> Intent:
    """Lightweight rule-based intent classifier.

    This keeps behaviour transparent and easy to debug.
    """
    q = question.lower()

    # Combinations / synergy questions
    if any(k in q for k in ["synergy", "synergize", "combination", "combo", "overcome resistance"]):
        return Intent.DRUG_SYNERGY

    # Mechanism / resistance questions
    if any(k in q for k in ["mechanism", "how does", "why does", "resistance", "acquired resistance"]):
        return Intent.RESISTANCE_MECHANISM

    # Biomarker questions
    if any(k in q for k in ["biomarker", "predict response", "predictive", "prognostic"]):
        return Intent.BIOMARKER

    # Indications / approvals
    if any(k in q for k in ["approved", "indication", "in which cancers", "in which disease"]):
        return Intent.APPROVED_FOR

    return Intent.GENERIC_GRAPH


def relations_for_intent(intent: Intent) -> List[str]:
    """Map intent → whitelist of relation types for expansion."""
    if intent == Intent.DRUG_SYNERGY:
        return [
            "SYNERGIZES_WITH",
            "OVERCOMES_RESISTANCE_WITH",
            "SEQUENTIAL_THERAPY_AFTER",
        ]
    if intent == Intent.RESISTANCE_MECHANISM:
        return [
            "CONFERS_RESISTANCE_TO",
            "PREDICTS_RESPONSE_TO",
            "SENSITIZES_TO",
            "UPREGULATED_IN_RESISTANCE",
            "DOWNREGULATED_IN_RESISTANCE",
            "ACTIVATES",
            "IS_ESSENTIAL_FOR",
            "BYPASSES",
            "RESISTANCE_EMERGES_IN",
        ]
    if intent == Intent.BIOMARKER:
        return [
            "PREDICTIVE_BIOMARKER_FOR",
            "PROGNOSTIC_BIOMARKER_IN",
            "PREDICTS_RESPONSE_TO",
        ]
    if intent == Intent.APPROVED_FOR:
        return [
            "APPROVED_FOR",
            "IN_TRIALS_FOR",
        ]
    # Generic exploration
    return [
        "CONFERS_RESISTANCE_TO",
        "PREDICTS_RESPONSE_TO",
        "SENSITIZES_TO",
        "SYNERGIZES_WITH",
        "OVERCOMES_RESISTANCE_WITH",
        "APPROVED_FOR",
        "IN_TRIALS_FOR",
        "ACTIVATES",
        "IS_ESSENTIAL_FOR",
        "BYPASSES",
    ]


