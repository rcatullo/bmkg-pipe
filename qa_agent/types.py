from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Intent(str, Enum):
    """High-level question intent used to choose expansion policies."""

    DRUG_SYNERGY = "drug_synergy"
    RESISTANCE_MECHANISM = "resistance_mechanism"
    BIOMARKER = "biomarker"
    APPROVED_FOR = "approved_for"
    GENERIC_GRAPH = "generic_graph"


@dataclass
class AnchorEntity:
    """Grounded KG entity used as a starting point for expansion."""

    node_id: Optional[str]
    umls_cui: Optional[str]
    label: str
    node_type: str  # e.g., "Gene", "Chemical"
    match_score: float
    match_method: str  # "id", "name", "synonym", "fuzzy"


@dataclass
class QueryPlan:
    """Structured plan produced before running any Cypher."""

    intent: Intent
    anchors: List[AnchorEntity]
    max_hops: int
    allowed_relations: List[str]
    required_node_types: Optional[List[str]]
    raw_question: str
    debug_notes: Dict[str, str]


