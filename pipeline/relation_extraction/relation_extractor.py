import logging
from typing import Any, Dict, Optional

from pipeline.model.llm_client import LLMClient
from pipeline.utils.pairing import CandidatePair
from pipeline.schema.loader import SchemaLoader
from pipeline.utils.utils import load_config, timestamp

logger = logging.getLogger(__name__)


class RelationExtractor:
    def __init__(
        self,
        schema: SchemaLoader,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.schema = schema
        self.config = config or load_config()
        self.llm = llm_client or LLMClient(config=self.config)

    def extract(self, pair: CandidatePair) -> Optional[Dict]:
        prompt = self.build_prompt(pair)
        response = self.llm.json_complete(prompt)
        result = response.get("json") or {}
        if not result:
            logger.warning(
                "No predicate returned for pmid=%s sentence_id=%s pair=%s/%s",
                pair.pmid,
                pair.sentence_id,
                pair.subject.get("text"),
                pair.obj.get("text"),
            )
            return None
        predicate = result.get("predicate")
        confidence = float(result.get("confidence", 0.0))
        if predicate not in [p.name for p in pair.predicates]:
            return None
        return {
            "pmid": pair.pmid,
            "sentence_id": pair.sentence_id,
            "sentence": pair.sentence,
            "subject": pair.subject,
            "object": pair.obj,
            "predicate": predicate,
            "confidence": confidence,
            "model_name": self.config["llm"]["model"],
            "model_version": self.config.get("model_version", "v1"),
            "prompt_version": self.config.get("prompt_version", "v1"),
            "timestamp": timestamp(),
            "explanation": result.get("explanation", ""),
        }

    def build_prompt(self, pair: CandidatePair) -> str:
        subject = pair.subject.get("text")
        obj = pair.obj.get("text")
        allowed = "\n".join(
            f"- {pred.name}: {pred.description[:140]}"
            for pred in pair.predicates
        )
        sentence = pair.sentence.replace(subject, f"[SUBJ]{subject}[/SUBJ]", 1)
        sentence = sentence.replace(obj, f"[OBJ]{obj}[/OBJ]", 1)
        return (
            "Determine which predicate (if any) fits the sentence.\n"
            f"Sentence: {sentence}\n"
            f"Allowed predicates:\n{allowed}\n"
            "Respond as JSON {predicate: str, confidence: float, explanation: str}."
        )

