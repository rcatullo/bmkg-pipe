from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from model.llm_client import LLMClient
from schema import SchemaLoader
from utils.api_req_parallel import process_api_requests_from_file
from utils import ensure_dir, timestamp

ROOT_DIR = Path(__file__).resolve().parent.parent
RELATION_REQUESTS_FILE = ROOT_DIR / "relation_extraction" / "tmp" / "requests.jsonl"
RELATION_RESULTS_FILE = ROOT_DIR / "relation_extraction" / "tmp" / "results.jsonl"

logger = logging.getLogger(__name__)


class RelationExtraction:
    def __init__(
        self,
        llm_client: LLMClient,
        config: Dict[str, Any],
        schema: SchemaLoader | None = None,
    ) -> None:
        self.llm = llm_client
        self.config = config
        self.schema = schema or SchemaLoader()
        self.total_requests = 0
        self._requests_handle = None
        self._prepare_request_file()

    def _prepare_request_file(self) -> None:
        ensure_dir(RELATION_REQUESTS_FILE)
        if RELATION_REQUESTS_FILE.exists():
            RELATION_REQUESTS_FILE.unlink()
        self._requests_handle = RELATION_REQUESTS_FILE.open("w", encoding="utf-8")

    def _system_prompt(self) -> str:
        return (
            "You are a precise biomedical relation extraction assistant. "
            "Given a sentence, a set of entities with indices and classes, and allowed predicates, "
            "return only the subject–predicate–object triples that are supported by the sentence. "
            "Use only entities and predicates provided. Respond only with valid JSON."
        )

    def _build_predicate_sections(self, predicates: List) -> str:
        guidelines = self.schema.guidelines
        sections = []
        for pred in predicates:
            guideline = guidelines.get(pred.name, {})
            definition = guideline.get("definition", pred.description) or pred.description
            section = f"{pred.name}: {definition}\n"
            section += f"Domain: {', '.join(pred.domain) if pred.domain else 'any'}; "
            section += f"Range: {', '.join(pred.range) if pred.range else 'any'}"
            sections.append(section)
        return "\n".join(sections)

    def _build_prompt(self, sentence: Dict, entities: List[Dict], predicates: List) -> str:
        predicate_text = self._build_predicate_sections(predicates)
        entity_lines = []
        for idx, ent in enumerate(entities):
            entity_lines.append(f"[{idx}] {ent.get('text')} (class={ent.get('class')})")
        entities_block = "\n".join(entity_lines)

        return (
            "Determine all subject–predicate–object triples supported by the sentence. "
            "Only use the entities and predicates provided. Reference entities by their index. "
            "Consider every valid subject/object pairing; return multiple triples if multiple "
            "relationships are expressed. If no predicates apply, return an empty list. "
            "Critically: if a subject participates in the same predicate with multiple objects, "
            "return a triple for each object.\n\n"
            f"Sentence: {sentence['text']}\n\n"
            f"Entities:\n{entities_block}\n\n"
            f"Allowed predicates:\n{predicate_text}\n\n"
            "Return JSON exactly as: "
            '{"triples":[{"subject": <entity_index>, "predicate": "<predicate_name>", '
            '"object": <entity_index>, "confidence": <0-1 float>}]}'
        )

    def add_sentence(self, sentence: Dict, entities: List[Dict], predicates: List) -> None:
        """Queue one RE request for a sentence with its entities and allowed predicates."""
        if not predicates:
            return
        if self._requests_handle is None:
            self._prepare_request_file()
        prompt = self._build_prompt(sentence, entities, predicates)
        payload = self.llm.build_chat_completion_kwargs(
            prompt=prompt,
            json_mode=True,
            system_prompt=self._system_prompt(),
        )
        payload["metadata"] = self._metadata(sentence, entities, predicates)
        self._requests_handle.write(json.dumps(payload) + "\n")
        self.total_requests += 1

    def _metadata(self, sentence: Dict, entities: List[Dict], predicates: List) -> Dict:
        return {
            "pmid": sentence.get("pmid"),
            "sentence_id": sentence.get("sentence_id"),
            "sentence": sentence.get("text"),
            "entities": entities,
            "predicate_names": [pred.name for pred in predicates],
            "model_name": self.config["llm"]["model"],
            "model_version": self.config.get("model_version", "v1"),
            "prompt_version": self.config.get("prompt_version", "v1"),
        }

    def run(self) -> List[Dict]:
        self._close_request_file()
        if self.total_requests == 0:
            logger.info("No sentences queued for relation extraction; skipping API call.")
            self._clear_results_file()
            return []
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Relation extraction requires an API key. Set OPENAI_API_KEY environment variable."
            )
        logger.info(
            "Starting relation extraction for %d sentences using %s",
            self.total_requests,
            self.config["llm"]["request_url"],
        )
        self._clear_results_file()
        ensure_dir(RELATION_RESULTS_FILE)
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(RELATION_REQUESTS_FILE),
                save_filepath=str(RELATION_RESULTS_FILE),
                request_url=self.config["llm"]["request_url"],
                api_key=api_key,
                max_requests_per_minute=float(self.config["llm"]["max_requests_per_minute"]),
                max_tokens_per_minute=float(self.config["llm"]["max_tokens_per_minute"]),
                token_encoding_name=self.config["llm"]["token_encoding_name"],
                max_attempts=int(self.config["relation_extraction"]["max_attempts"]),
                logging_level=int(self.config["logging"]["logging_level"]),
            )
        )
        return self._read_results()

    def _read_results(self) -> List[Dict]:
        results: List[Dict] = []
        if not RELATION_RESULTS_FILE.exists():
            logger.warning("Relation extraction results file %s not found.", RELATION_RESULTS_FILE)
            return results
        with RELATION_RESULTS_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed relation extraction response line.")
                    continue
                if not isinstance(payload, list) or len(payload) < 2:
                    logger.warning("Unexpected relation extraction payload: %s", payload)
                    continue
                response = payload[1]
                metadata = payload[2] if len(payload) > 2 else None
                relation = self._build_relation(metadata, response)
                if relation:
                    if isinstance(relation, list):
                        results.extend(relation)
                    else:
                        results.append(relation)
        return results

    def _build_relation(self, metadata: Optional[Dict], response: Dict) -> Optional[Dict]:
        if metadata is None:
            logger.warning("Relation extraction response missing metadata.")
            return None
        if isinstance(response, list):
            logger.error(
                "Relation extraction for pmid=%s sentence_id=%s failed after retries: %s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
                response,
            )
            return None
        if "error" in response:
            logger.error(
                "Relation extraction for pmid=%s sentence_id=%s returned API error: %s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
                response["error"],
            )
            return None
        choices = response.get("choices") or []
        if not choices:
            logger.warning(
                "Relation extraction response missing choices for pmid=%s sentence_id=%s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
            )
            return None
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content:
            logger.warning(
                "Relation extraction response missing content for pmid=%s sentence_id=%s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
            )
            return None
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to decode relation extraction JSON for pmid=%s sentence_id=%s: %s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
                content,
            )
            return None
        triples = result.get("triples")
        if not isinstance(triples, list):
            logger.warning(
                "Relation extraction response missing triples for pmid=%s sentence_id=%s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
            )
            return None
        allowed = set(metadata.get("predicate_names", []))
        entities = metadata.get("entities") or []
        sentence_relations = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            subj_idx = triple.get("subject")
            obj_idx = triple.get("object")
            predicate = triple.get("predicate")
            if predicate not in allowed:
                continue
            if not isinstance(subj_idx, int) or not isinstance(obj_idx, int):
                continue
            if subj_idx < 0 or obj_idx < 0 or subj_idx >= len(entities) or obj_idx >= len(entities):
                continue
            confidence = float(triple.get("confidence", 0.0))
            subject_ent = entities[subj_idx]
            object_ent = entities[obj_idx]
            sentence_relations.append(
                {
                    "pmid": metadata.get("pmid"),
                    "sentence_id": metadata.get("sentence_id"),
                    "sentence": metadata.get("sentence"),
                    "subject": subject_ent,
                    "object": object_ent,
                    "predicate": predicate,
                    "confidence": confidence,
                    "model_name": metadata.get("model_name"),
                    "model_version": metadata.get("model_version"),
                    "prompt_version": metadata.get("prompt_version"),
                    "timestamp": timestamp(),
                }
            )
        if not sentence_relations:
            return None
        # Return a list; caller will extend results.
        return sentence_relations

    def _close_request_file(self) -> None:
        if self._requests_handle is not None:
            self._requests_handle.close()
            self._requests_handle = None

    def _clear_results_file(self) -> None:
        ensure_dir(RELATION_RESULTS_FILE)
        if RELATION_RESULTS_FILE.exists():
            RELATION_RESULTS_FILE.unlink()

