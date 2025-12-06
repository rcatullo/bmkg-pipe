from __future__ import annotations

from typing import Any, Dict, List, Tuple

from neo4j import Driver

from .types import QueryPlan


class GraphExpander:
    """Run controlled graph exploration based on a QueryPlan."""

    def __init__(self, driver: Driver):
        self.driver = driver

    def expand(self, plan: QueryPlan) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return a small evidence subgraph and debug metadata.

        This uses a progressively relaxed strategy:
        1) Try anchored expansion via node ids / umls_cui.
        2) If nothing is found, fall back to relation-only search constrained by intent.
        """
        debug: Dict[str, Any] = {"strategy": None, "notes": []}

        if plan.anchors:
            records = self._expand_from_anchors(plan, debug)
            if records:
                debug["strategy"] = "anchors"
                return records, debug
            debug["notes"].append("No results from anchored expansion; falling back.")

        records = self._expand_without_anchors(plan, debug)
        if records:
            debug["strategy"] = "relation_only"
        else:
            debug["strategy"] = "empty"
        return records, debug

    def _expand_from_anchors(self, plan: QueryPlan, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Start from anchored nodes and follow relation whitelist."""
        anchor_ids = [a.node_id for a in plan.anchors if a.node_id]
        anchor_cuis = [a.umls_cui for a in plan.anchors if a.umls_cui]

        if not anchor_ids and not anchor_cuis:
            debug["notes"].append("Anchors had no node ids or umls_cui; skipping anchored expansion.")
            return []

        allowed_rels = plan.allowed_relations
        # Cypher relationship types are stored uppercased in import script.
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (s)-[r]->(o)
                WHERE type(r) IN $allowed_rels
                  AND (
                        (s.id IN $anchor_ids OR o.id IN $anchor_ids)
                     OR (s.umls_cui IS NOT NULL AND s.umls_cui IN $anchor_cuis)
                     OR (o.umls_cui IS NOT NULL AND o.umls_cui IN $anchor_cuis)
                  )
                RETURN
                  properties(s) AS subject,
                  properties(o) AS object,
                  type(r) AS predicate,
                  r.pmids AS pmids,
                  r.sentences AS evidence
                LIMIT 100
                """,
                allowed_rels=allowed_rels,
                anchor_ids=anchor_ids,
                anchor_cuis=anchor_cuis,
            ).data()
        debug["notes"].append(f"Anchored expansion returned {len(records)} rows.")
        return records

    def _expand_without_anchors(self, plan: QueryPlan, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search only by relation types, ignoring anchors.

        This is a safety net to avoid hard empty responses.
        """
        allowed_rels = plan.allowed_relations
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (s)-[r]->(o)
                WHERE type(r) IN $allowed_rels
                RETURN
                  properties(s) AS subject,
                  properties(o) AS object,
                  type(r) AS predicate,
                  r.pmids AS pmids,
                  r.sentences AS evidence
                LIMIT 50
                """,
                allowed_rels=allowed_rels,
            ).data()
        debug["notes"].append(
            f"Relation-only expansion for relations {allowed_rels} returned {len(records)} rows."
        )
        return records


