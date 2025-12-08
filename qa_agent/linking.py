from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import Driver

from .types import AnchorEntity

logger = logging.getLogger(__name__)


@dataclass
class LinkerAttemptResult:
    """Stores what the linker tried for debugging when no entities are found."""

    tried_terms: List[str]
    notes: List[str]


@dataclass
class AnnotatedEntity:
    """Entity extracted by LLM with original text and canonical form."""

    text: str
    class_name: str
    canonical_form: Optional[str]


class EntityLinker:
    """Resolve free-text mentions in a question to KG nodes.

    Uses LLM-based entity annotation, optional UMLS lookup for KG ids, and Cypher matching:
    - LLM annotates question to extract biomedical entities with canonical forms
    - UMLS API can provide CUIs that map directly to node ids (e.g., UMLS:C1425006)
    - Cypher matching prioritizes id-based matching, falls back to fuzzy matching
    """

    def __init__(
        self,
        driver: Driver,
        llm_client: Any,  # LLMClient
        schema_loader: Any,  # SchemaLoader
        umls_client: Any,  # UMLSClient
    ):
        self.driver = driver
        self.llm_client = llm_client
        self.schema_loader = schema_loader
        self.umls_client = umls_client
        self.classes = list(schema_loader.entity_classes().keys())

    def _build_ner_prompt(self, question: str) -> str:
        """Build the NER prompt (reused from named_entity_recognition/ner.py)."""
        class_list = ", ".join(self.classes)
        return (
            "Identify biomedical entities for the sentence below (if any). If the entity doesn't fit confidently into one of the given classes, even if it is biomedical in nature, omit it.\n"
            f"Classes: {class_list}.\n\n"
            "For each entity, also normalize it to its most appropriate canonical form for UMLS database search. "
            "Convert to the standard canonical form that would be found in medical databases. Consider:\n"
            "- Gene symbols should use official HGNC notation (e.g., \"p53\" → \"TP53\")\n"
            "- Diseases should use standard medical terminology (e.g., \"breast cancer\" → \"breast neoplasms\")\n"
            "- Chemicals should use standard chemical names (e.g., \"aspirin\" → \"acetylsalicylic acid\")\n"
            "- Proteins should use standard protein names\n"
            "- Resolve abbreviations and synonyms to canonical forms\n\n"
            "If the entity is already in canonical form or no normalization is needed, use the original text as the canonical form.\n"
            "If the entity cannot be normalized, set canonical_form to null.\n\n"
            "Return a JSON object with a single key 'entities' containing an array of objects, each with 'text', 'class', and 'canonical_form' fields.\n"
            "Example: {\"entities\": [{\"text\": \"BRCA1\", \"class\": \"Gene\", \"canonical_form\": \"BRCA1\"}]}\n\n"
            f"Question: {question}"
        )

    def _annotate_question(self, question: str) -> List[AnnotatedEntity]:
        """Use LLM to annotate the question and extract biomedical entities."""
        prompt = self._build_ner_prompt(question)
        
        try:
            response = self.llm_client.json_complete(prompt)
            data = response.get("json")
            
            if not data:
                logger.warning("LLM returned no JSON for entity annotation")
                return []
            
            entities_data = data.get("entities", [])
            if not entities_data:
                logger.info("LLM found no entities in question")
                return []
            
            annotated = []
            for e in entities_data:
                text = e.get("text", "").strip()
                class_name = e.get("class", "").strip()
                canonical_form = e.get("canonical_form")
                
                if not text or not class_name:
                    continue
                
                # Use original text as canonical form if not provided
                if canonical_form is None or canonical_form == "":
                    canonical_form = text
                
                annotated.append(
                    AnnotatedEntity(
                        text=text,
                        class_name=class_name,
                        canonical_form=canonical_form.strip() if canonical_form else None,
                    )
                )
            
            return annotated
        except Exception as exc:
            logger.error("Failed to annotate question with LLM: %s", exc, exc_info=True)
            return []

    def _get_umls_id(self, canonical_form: str) -> Optional[str]:
        """Query UMLS API to get a KG-ready id (UMLS:CXXXX)."""
        if not canonical_form or not self.umls_client:
            return None
        
        try:
            cui = self.umls_client.search_concept(canonical_form)
            return f"UMLS:{cui}" if cui else None
        except Exception as exc:
            logger.warning("UMLS search failed for '%s': %s", canonical_form, exc)
            return None

    def _match_by_id(
        self, session: Any, kg_id: str, class_name: str
    ) -> List[AnchorEntity]:
        """Match nodes in Neo4j using the canonical node id."""
        # Match by id and optionally filter by class
        query = """
        MATCH (n)
        WHERE n.id = $kg_id
        """
        
        # Add class filter if the class is valid
        if class_name in self.classes:
            query += f" AND '{class_name}' IN labels(n)"
        
        query += " RETURN n LIMIT 10"
        
        try:
            records = session.run(query, kg_id=kg_id).data()
            anchors = []
            for r in records:
                n = r["n"]
                labels = list(getattr(n, "labels", []))
                props = dict(n)
                anchors.append(
                    AnchorEntity(
                        node_id=props.get("id"),
                        label=props.get("name") or props.get("text") or "",
                        node_type=labels[0] if labels else "Entity",
                        match_score=0.95,
                        match_method="id_exact",
                    )
                )
            return anchors
        except Exception as exc:
            logger.warning("Cypher query failed for id matching: %s", exc)
            return []

    def _match_fuzzy(
        self, session: Any, term: str, class_name: str
    ) -> List[AnchorEntity]:
        """Fallback fuzzy matching when id-based linking fails."""
        anchors = []
        
        # Try exact match first
        query = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($term)
           OR toLower(n.text) = toLower($term)
        """
        
        if class_name in self.classes:
            query += f" AND '{class_name}' IN labels(n)"
        
        query += " RETURN n LIMIT 5"
        
        try:
            records = session.run(query, term=term).data()
            if records:
                for r in records:
                    n = r["n"]
                    labels = list(getattr(n, "labels", []))
                    props = dict(n)
                    anchors.append(
                        AnchorEntity(
                            node_id=props.get("id"),
                            label=props.get("name") or props.get("text") or term,
                            node_type=labels[0] if labels else "Entity",
                            match_score=0.9,
                            match_method="name_exact",
                        )
                    )
                return anchors
        except Exception as exc:
            logger.warning("Cypher query failed for exact matching: %s", exc)
        
        # Try contains match (fuzzy)
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($term)
           OR toLower(n.text) CONTAINS toLower($term)
        """
        
        if class_name in self.classes:
            query += f" AND '{class_name}' IN labels(n)"
        
        query += " RETURN n LIMIT 5"
        
        try:
            records = session.run(query, term=term).data()
            if records:
                for r in records:
                    n = r["n"]
                    labels = list(getattr(n, "labels", []))
                    props = dict(n)
                    anchors.append(
                        AnchorEntity(
                            node_id=props.get("id"),
                            label=props.get("name") or props.get("text") or term,
                            node_type=labels[0] if labels else "Entity",
                            match_score=0.7,
                            match_method="name_contains",
                        )
                    )
        except Exception as exc:
            logger.warning("Cypher query failed for fuzzy matching: %s", exc)
        
        return anchors

    def link_question_entities(
        self, question: str
    ) -> tuple[List[AnchorEntity], LinkerAttemptResult]:
        """Return anchored entities plus debug info about what was tried.

        Process:
        1. Use LLM to annotate question and extract entities with canonical forms
        2. Optionally query UMLS API to get UMLS-based ids for each canonical form
        3. Match in Neo4j using ids first, fall back to fuzzy matching
        """
        tried_terms: List[str] = []
        notes: List[str] = []
        anchors: List[AnchorEntity] = []

        # Step 1: LLM annotation
        annotated_entities = self._annotate_question(question)
        
        if not annotated_entities:
            notes.append("LLM found no biomedical entities in the question.")
            return anchors, LinkerAttemptResult(tried_terms=tried_terms, notes=notes)

        notes.append(f"LLM annotated {len(annotated_entities)} entities from question.")

        with self.driver.session() as session:
            for entity in annotated_entities:
                tried_terms.append(f"{entity.text} ({entity.class_name})")
                
                # Step 2: Get KG id via UMLS (if available)
                kg_id = None
                if entity.canonical_form:
                    kg_id = self._get_umls_id(entity.canonical_form)
                    if kg_id:
                        notes.append(
                            f"Found KG id {kg_id} for '{entity.text}' (canonical: {entity.canonical_form})"
                        )
                    else:
                        notes.append(
                            f"No KG id found for '{entity.text}' (canonical: {entity.canonical_form})"
                        )

                # Step 3: Match in Neo4j
                matched = False
                
                # Try id-based matching first
                if kg_id:
                    id_anchors = self._match_by_id(session, kg_id, entity.class_name)
                    if id_anchors:
                        anchors.extend(id_anchors)
                        matched = True
                        notes.append(f"Matched {len(id_anchors)} nodes via id {kg_id}")

                # Fall back to fuzzy matching if no id match
                if not matched:
                    # Use canonical form if available, otherwise original text
                    search_term = entity.canonical_form or entity.text
                    fuzzy_anchors = self._match_fuzzy(session, search_term, entity.class_name)
                    if fuzzy_anchors:
                        anchors.extend(fuzzy_anchors)
                        matched = True
                        notes.append(
                            f"Matched {len(fuzzy_anchors)} nodes via fuzzy matching for '{search_term}'"
                        )

                if not matched:
                    notes.append(
                        f"No nodes matched for entity '{entity.text}' (class: {entity.class_name})"
                    )

        if not anchors:
            notes.append("No nodes matched any annotated entities at any matching level.")

        return anchors, LinkerAttemptResult(tried_terms=tried_terms, notes=notes)
