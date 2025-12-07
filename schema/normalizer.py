import logging
import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

from .loader import SchemaLoader

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(
        self,
        schema: SchemaLoader,
        llm_client=None,
        umls_client=None,
    ):
        self.policy = schema.normalization_policy()
        self.umls_client = umls_client
        # Cache canonical text → assigned id to reuse and enable fuzzy matching
        self._canonical_to_id: Dict[str, Optional[str]] = {}

    @staticmethod
    def _slug(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
        return cleaned.strip("_") or "unknown"

    def _fuzzy_match_canonical(self, text: str) -> Tuple[Optional[str], float]:
        """Return (id, score) for the closest known canonical text."""
        if not text:
            return None, 0.0
        best_id = None
        best_score = 0.0
        for canonical, existing_id in self._canonical_to_id.items():
            score = SequenceMatcher(None, text.lower(), canonical.lower()).ratio()
            if score > best_score:
                best_score = score
                best_id = existing_id
        return best_id, best_score

    def normalize(self, entity: Dict, context: str = "") -> Dict:
        cls = entity.get("class")
        entity_text = entity.get("text", "")
        canonical_form = entity.get("canonical_form") or entity_text

        chosen: Optional[str] = None
        if self.umls_client and cls in ["Gene", "Chemical", "Disease", "Phenotype", "Pathway", "Mutation"]:
            umls_cui = self.umls_client.search_concept(canonical_form)
            if umls_cui:
                chosen = f"UMLS:{umls_cui}"
                self._canonical_to_id[canonical_form] = chosen
            else:
                match_id, score = self._fuzzy_match_canonical(canonical_form)
                if match_id and score >= 0.88:
                    chosen = match_id
        if chosen is None:
            match_id, score = self._fuzzy_match_canonical(canonical_form)
            if match_id and score >= 0.88:
                chosen = match_id

        normalized = entity.copy()
        normalized["id"] = chosen
        if canonical_form:
            normalized["canonical_form"] = canonical_form
        # Remove any legacy ids field
        normalized.pop("ids", None)
        normalized.pop("umls_cui", None)
        return normalized

