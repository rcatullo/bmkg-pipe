from __future__ import annotations

from typing import List, Optional, Tuple

from .intent import Intent, classify_intent, relations_for_intent
from .linking import EntityLinker, LinkerAttemptResult
from .types import AnchorEntity, QueryPlan


class QueryPlanner:
    """Turn a natural language question into a structured KG query plan."""

    def __init__(self, linker: EntityLinker):
        self._linker = linker

    def plan(self, question: str) -> Tuple[QueryPlan, LinkerAttemptResult]:
        """Produce a query plan and linker debug info."""
        intent = classify_intent(question)
        anchors, linker_debug = self._linker.link_question_entities(question)

        allowed_relations = relations_for_intent(intent)

        # Simple heuristic for hop depth: mechanism questions usually need more context.
        if intent in (Intent.RESISTANCE_MECHANISM, Intent.DRUG_SYNERGY):
            max_hops = 3
        else:
            max_hops = 2

        # Node type requirements can be tuned per intent; keep it permissive for now.
        required_node_types: Optional[List[str]] = None

        debug_notes = {
            "intent": intent.value,
            "anchor_count": str(len(anchors)),
            "allowed_relations": ", ".join(allowed_relations),
        }

        plan = QueryPlan(
            intent=intent,
            anchors=anchors,
            max_hops=max_hops,
            allowed_relations=allowed_relations,
            required_node_types=required_node_types,
            raw_question=question,
            debug_notes=debug_notes,
        )
        return plan, linker_debug


