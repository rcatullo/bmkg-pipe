import logging
import re
from typing import Dict, Optional

from .loader import SchemaLoader

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(
        self,
        schema: SchemaLoader,
        llm_client=None,
        umls_client=None,
        existing_cui_to_canonical: Optional[Dict[str, str]] = None,
    ):
        self.policy = schema.normalization_policy()
        self.llm_client = llm_client
        self.umls_client = umls_client
        self._cui_to_canonical: Dict[str, str] = existing_cui_to_canonical.copy() if existing_cui_to_canonical else {}

    @staticmethod
    def _slug(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
        return cleaned.strip("_") or "unknown"

    @staticmethod
    def _coerce_ids(raw_ids):
        if isinstance(raw_ids, dict):
            return raw_ids
        if isinstance(raw_ids, list):
            converted = {}
            for item in raw_ids:
                if not isinstance(item, dict):
                    continue
                key = item.get("type") or item.get("namespace") or item.get("name")
                value = item.get("id") or item.get("value")
                if key and value:
                    converted[key] = value
            return converted
        if isinstance(raw_ids, str):
            return {"id": raw_ids}
        if raw_ids is None:
            return {}
        logger.warning("Unsupported ids payload %s", raw_ids)
        return {}

    def _normalize_to_canonical(self, entity_text: str, entity_type: str, context: str = "") -> Optional[str]:
        if not self.llm_client:
            return entity_text

        prompt = (
            'Normalize this biomedical entity mention to its most appropriate canonical form for UMLS database search.\n\n'
            f'Entity: "{entity_text}" (type: {entity_type})\n\n'
            f'Context: "{context}"\n\n'
            'Convert to the standard canonical form that would be found in medical databases. Consider:\n\n'
            '- Gene symbols should use official HGNC notation (e.g., "p53" → "TP53")\n'
            '- Diseases should use standard medical terminology (e.g., "breast cancer" → "breast neoplasms")\n'
            '- Chemicals should use standard chemical names (e.g., "aspirin" → "acetylsalicylic acid")\n'
            '- Proteins should use standard protein names\n'
            '- Resolve abbreviations and synonyms to canonical forms\n\n'
            'Examples:\n\n'
            '- "p53" → "TP53"\n'
            '- "breast cancer" → "breast neoplasms"\n'
            '- "aspirin" → "acetylsalicylic acid"\n'
            '- "TNF-alpha" → "tumor necrosis factor alpha"\n'
            '- "VEGF" → "vascular endothelial growth factor"\n'
            '- "COVID-19" → "coronavirus disease 2019"\n\n'
            'If the entity is already in canonical form or no normalization is needed, return the original text.\n\n'
            'If the entity cannot be normalized, return "None".\n\n'
            'Output only the canonical form itself, with no extra text, punctuation, or formatting.'
        )

        try:
            result = self.llm_client.complete(prompt)
            canonical = result.get("text", "").strip()
            if canonical and canonical.lower() != "none":
                return canonical
            return entity_text
        except Exception as exc:
            logger.warning("LLM normalization failed for '%s': %s", entity_text, exc)
            return entity_text

    def normalize(self, entity: Dict, context: str = "") -> Dict:
        cls = entity.get("class")
        entity_text = entity.get("text", "")
        policy = self.policy.get(cls, {})
        ids = self._coerce_ids(entity.get("ids"))

        umls_cui = ids.get("umls_cui")
        canonical_form = entity.get("canonical_form")
        if self.umls_client and cls in ["Gene", "Chemical", "Disease", "Phenotype", "Pathway", "Mutation"]:
            if umls_cui and umls_cui in self._cui_to_canonical:
                canonical_form = self._cui_to_canonical[umls_cui]
            else:
                if not canonical_form and self.llm_client:
                    canonical_form = self._normalize_to_canonical(entity_text, cls, context)
                if canonical_form:
                    if not umls_cui:
                        umls_cui = self.umls_client.search_concept(canonical_form)
                    if umls_cui:
                        ids["umls_cui"] = umls_cui
                        if umls_cui in self._cui_to_canonical:
                            canonical_form = self._cui_to_canonical[umls_cui]
                        else:
                            self._cui_to_canonical[umls_cui] = canonical_form

        chosen = None
        if policy:
            primary = policy.get("primary")
            alternates = policy.get("alternates", [])
            if primary and ids.get(primary):
                chosen = ids[primary]
            else:
                for alt in alternates:
                    if ids.get(alt):
                        chosen = ids[alt]
                        break

        if not chosen:
            if umls_cui:
                chosen = f"UMLS:{umls_cui}"
            else:
                chosen = f"{cls}:{self._slug(entity_text)}"

        normalized = entity.copy()
        normalized["id"] = chosen
        if ids:
            normalized["ids"] = ids
        if umls_cui:
            normalized["umls_cui"] = umls_cui
        if canonical_form:
            normalized["canonical_form"] = canonical_form
        return normalized

