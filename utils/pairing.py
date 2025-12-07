from dataclasses import dataclass
from typing import Dict, List

from schema.loader import Predicate, SchemaLoader
from .utils import Sentence


class PredicateFilter:
    """Collect allowed predicates per sentence based on entity classes."""

    def __init__(self, schema: SchemaLoader):
        self.predicates = schema.predicates()

    def for_entities(self, entities: List[Dict]) -> List[Predicate]:
        """Return unique predicates whose (domain, range) match any ordered entity-class pair."""
        allowed: Dict[str, Predicate] = {}
        for subj in entities:
            for obj in entities:
                if subj is obj:
                    continue
                subj_cls = subj.get("class")
                obj_cls = obj.get("class")
                if not subj_cls or not obj_cls:
                    continue
                for pred in self.predicates.values():
                    if subj_cls in pred.domain and obj_cls in pred.range:
                        allowed[pred.name] = pred
        return list(allowed.values())

