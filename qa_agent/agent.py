from __future__ import annotations

from typing import Any, Dict, List, Tuple

from llama_index.llms.openai import OpenAI
from neo4j import Driver

from .expansion import GraphExpander
from .linking import EntityLinker
from .planner import QueryPlanner
from .types import QueryPlan


class KgQaAgent:
    """KG-first QA agent implementing Anchor → Expand → Explain."""

    def __init__(
        self,
        driver: Driver,
        llm_client: Any,  # LLMClient
        schema_loader: Any,  # SchemaLoader
        umls_client: Any,  # UMLSClient
        llm_model: str = "gpt-5-nano",
    ):
        self.driver = driver
        self.entity_linker = EntityLinker(driver, llm_client, schema_loader, umls_client)
        self.planner = QueryPlanner(self.entity_linker)
        self.expander = GraphExpander(driver)
        self.llm_model = llm_model

    def answer(self, question: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Answer a question using KG-first reasoning.

        Returns: (answer_text, records, debug_info)
        """
        plan, linker_debug = self.planner.plan(question)
        records, expand_debug = self.expander.expand(plan)

        debug = {
            "plan": plan.debug_notes,
            "linker": {
                "tried_terms": linker_debug.tried_terms,
                "notes": linker_debug.notes,
            },
            "expansion": expand_debug,
        }

        if not records:
            debug["notes"] = ["No graph records found for any strategy."]
            answer = self._build_empty_answer(question, plan, debug)
            return answer, [], debug

        context = self._build_context(records, plan)
        answer = self._synthesize_answer(question, context)
        return answer, records, debug

    def _build_context(self, records: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Turn graph triples into a textual context block for the LLM."""
        lines: List[str] = []
        for rec in records[:40]:
            subj = rec.get("subject") or {}
            obj = rec.get("object") or {}
            predicate = rec.get("predicate") or "RELATED_TO"
            pmids = rec.get("pmids") or []
            evid = rec.get("evidence") or []

            def fmt(node: Dict[str, Any]) -> str:
                name = node.get("name") or node.get("text") or node.get("id") or "UNKNOWN"
                n_cls = node.get("class")
                return f"{name} ({n_cls})" if n_cls else name

            line = f"{fmt(subj)} --[{predicate}]→ {fmt(obj)}"
            if pmids:
                line += f"; PMIDs: {', '.join(pmids[:3])}"

            sentence = None
            if isinstance(evid, list):
                for ev in evid:
                    if isinstance(ev, dict) and ev.get("sentence"):
                        sentence = ev["sentence"]
                        break
                    if isinstance(ev, str):
                        sentence = ev
                        break
            if sentence:
                line += f"; Evidence: {sentence}"
            lines.append(line)

        header = f"Intent: {plan.intent.value}; anchors: {len(plan.anchors)}; relations: {', '.join(plan.allowed_relations)}"
        return header + "\n" + "\n".join(lines)

    def _synthesize_answer(self, question: str, context: str) -> str:
        """Use an LLM to turn relations context into a natural language answer."""
        llm = OpenAI(model=self.llm_model)
        prompt = f"""
You are a biomedical assistant answering questions using ONLY the provided knowledge-graph relations.

Relations:
{context}

Question: {question}

Instructions:
- Base your answer strictly on these relations; do not fabricate unseen facts.
- Explain mechanisms and chains of reasoning using the relations, explicitly naming genes, pathways, and drugs.
- If the relations are suggestive but not definitive, say that the answer is inferred and describe the limitations.
"""
        try:
            completion = llm.complete(prompt)
            return getattr(completion, "text", str(completion)).strip()
        except Exception as exc:  # pragma: no cover - defensive
            return f"Failed to synthesize answer from relations: {exc}"

    def _build_empty_answer(self, question: str, plan: QueryPlan, debug: Dict[str, Any]) -> str:
        """Human-readable explanation when the KG cannot answer."""
        parts = [
            "I could not find supporting facts in the current knowledge graph for this question.",
            "",
            "What was attempted:",
            f"- Detected intent: {plan.intent.value}",
            f"- Tried relation types: {', '.join(plan.allowed_relations)}",
        ]
        linker = debug.get("linker", {})
        tried_terms = linker.get("tried_terms") or []
        if tried_terms:
            parts.append(f"- Candidate entities from the question: {', '.join(tried_terms)}")
        linker_notes = linker.get("notes") or []
        if linker_notes:
            for note in linker_notes:
                parts.append(f"- Entity linking: {note}")
        expansion = debug.get("expansion", {})
        for note in expansion.get("notes", []):
            parts.append(f"- Expansion: {note}")
        return "\n".join(parts)


