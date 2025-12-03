import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Set, Tuple

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

from model.llm_client import LLMClient
from named_entity_recognition import NamedEntityRecognition
from relation_extraction import RelationExtraction
from schema import SchemaLoader, Normalizer
import utils as utils
from utils import PostProcessor, PairGenerator, Sentence
from utils.utils import read_jsonl

logger = logging.getLogger("pipeline.run")


def configure_logging(config: Dict[str, Any]) -> None:
    log_path = Path(config["logging"]["log_file"])
    utils.ensure_dir(log_path)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(getattr(logging, config["logging"]["level"].upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, config["logging"]["level"].upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config["logging"]["level"].upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)


def log_stage(stage: str, **details) -> None:
    if details:
        detail_str = " ".join(f"{key}={value}" for key, value in details.items())
        logger.info("Pipeline stage=%s %s", stage, detail_str)
    else:
        logger.info("Pipeline stage=%s", stage)


def load_processed_pmids(processed_pmids_log_path: Path) -> Set[str]:
    """Load set of already processed pmids from log file."""
    processed_pmids: Set[str] = set()
    if processed_pmids_log_path.exists():
        logger.info("Loading processed pmids from %s", processed_pmids_log_path)
        for record in read_jsonl(processed_pmids_log_path):
            pmid = record.get("pmid")
            if pmid:
                processed_pmids.add(str(pmid))
        logger.info("Found %d processed pmids", len(processed_pmids))
    else:
        logger.info("No processed pmids log found at %s", processed_pmids_log_path)
    return processed_pmids


def load_existing_canonical_forms(relations_path: Path) -> Dict[str, str]:
    """
    Load existing relations.jsonl and extract:
    - Dictionary mapping UMLS CUI to canonical_form for consistent normalization
    """
    cui_to_canonical: Dict[str, str] = {}
    
    if not relations_path.exists():
        logger.info("No existing relations file found at %s", relations_path)
        return cui_to_canonical
    
    logger.info("Loading canonical forms from existing relations at %s", relations_path)
    count = 0
    for record in read_jsonl(relations_path):
        count += 1
        # Extract canonical forms from subject and object
        for entity_key in ["subject", "object"]:
            entity = record.get(entity_key, {})
            umls_cui = entity.get("umls_cui")
            canonical_form = entity.get("canonical_form")
            if umls_cui and canonical_form:
                # If we already have a canonical form for this CUI, keep the existing one
                # (first one wins for consistency)
                if umls_cui not in cui_to_canonical:
                    cui_to_canonical[umls_cui] = canonical_form
    
    logger.info(
        "Loaded %d existing relations: %d unique CUIs with canonical forms",
        count,
        len(cui_to_canonical),
    )
    return cui_to_canonical


def log_processed_pmid(processed_pmids_log_path: Path, pmid: str, timestamp: str):
    """Log a processed pmid to the log file."""
    processed_pmids_log_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_pmids_log_path.open("a", encoding="utf-8") as fh:
        record = {
            "pmid": pmid,
            "timestamp": timestamp,
        }
        fh.write(json.dumps(record) + "\n")


def build_components(existing_cui_to_canonical: Dict[str, str] = None):
    config = utils.load_config()
    schema = SchemaLoader()
    llm = LLMClient(config=config)
    from model.umls_client import UMLSClient
    umls = UMLSClient(
        api_key=config.get("umls", {}).get("api_key", ""),
        api_url=config.get("umls", {}).get("api_url", "https://uts-ws.nlm.nih.gov/rest"),
    )
    normalizer = Normalizer(
        schema, 
        llm_client=llm, 
        umls_client=umls,
        existing_cui_to_canonical=existing_cui_to_canonical,
    )
    ner = NamedEntityRecognition(schema, normalizer, llm, config)
    pair_generator = PairGenerator(schema)
    re = RelationExtraction(llm, config, schema=schema)
    postprocessor = PostProcessor()
    return ner, pair_generator, re, postprocessor


def main():
    config = utils.load_config()
    configure_logging(config)

    input_path = Path(config["data"]["input_file"])
    output_path = Path(config["data"]["output_file"])
    relation_log_path = Path(config["logging"]["relation_log_file"])
    processed_pmids_log_path = Path(config["logging"]["processed_pmids_log"])
    
    # Load processed pmids from separate log file
    log_stage("load_processed_pmids")
    processed_pmids = load_processed_pmids(processed_pmids_log_path)
    log_stage("load_processed_pmids_complete", pmids=len(processed_pmids))
    
    # Load canonical forms from existing relations
    log_stage("load_existing_canonical_forms")
    existing_cui_to_canonical = load_existing_canonical_forms(output_path)
    log_stage("load_existing_canonical_forms_complete", cuis=len(existing_cui_to_canonical))

    log_stage("build_components")
    ner, pair_generator, re, postprocessor = build_components(existing_cui_to_canonical)
    log_stage("build_components_complete")
    
    raw_results = []
    sentence_count = 0
    entity_total = 0
    pair_total = 0

    logger.info(
        "Starting pipeline input=%s output=%s log=%s",
        input_path,
        output_path,
        relation_log_path,
    )

    # Load all sentences and filter out already processed pmids
    all_sentences = list(utils.load_sentences(input_path))
    sentences = [s for s in all_sentences if s.pmid not in processed_pmids]
    
    skipped_count = len(all_sentences) - len(sentences)
    if skipped_count > 0:
        logger.info(
            "Skipping %d sentences from %d already processed pmids",
            skipped_count,
            len(processed_pmids),
        )
    
    log_stage("entity_queue", sentences=len(sentences), skipped=skipped_count)
    
    # Track pmids that will be processed (to log them later)
    pmids_to_process = set(s.pmid for s in sentences)
    
    ner.add_sentences(sentences)
    log_stage("entity_execute", sentences=ner.total_sentences)
    entity_mapping = ner.run()
    log_stage("entity_results", sentences=len(entity_mapping))

    for sentence in sentences:
        sentence_count += 1
        entities = entity_mapping.get((sentence.pmid, sentence.sentence_id), [])
        if not entities:
            logger.debug(
                "No entities for pmid=%s sentence_id=%s", sentence.pmid, sentence.sentence_id
            )
            continue
        entity_total += len(entities)
        log_stage(
            "pair_generation",
            pmid=sentence.pmid,
            sentence_id=sentence.sentence_id,
            entity_count=len(entities),
        )
        pairs = pair_generator.generate(sentence, entities)
        pair_total += len(pairs)
        if not pairs:
            continue
        log_stage(
            "relation_extraction",
            pmid=sentence.pmid,
            sentence_id=sentence.sentence_id,
            pair_count=len(pairs),
        )
        re.add_pairs(pairs)
        if sentence_count and sentence_count % 50 == 0:
            logger.info(
                "Processed %d sentences (%d entities, %d pairs so far)",
                sentence_count,
                entity_total,
                pair_total,
            )

    log_stage("relation_execute", total_pairs=re.total_pairs)
    for classification in re.run():
        utils.log_result(classification, relation_log_path)
        raw_results.append(classification)

    postprocessor.threshold = config["relation_extraction"]["threshold"]
    log_stage("postprocess_filter", total=len(raw_results))
    filtered = postprocessor.filter(raw_results)
    log_stage("postprocess_aggregate", filtered=len(filtered))
    aggregated = postprocessor.aggregate(filtered)
    
    # Log all processed pmids (even if no relations were extracted)
    log_stage("log_processed_pmids", pmids=len(pmids_to_process))
    timestamp = utils.timestamp()
    for pmid in pmids_to_process:
        log_processed_pmid(processed_pmids_log_path, pmid, timestamp)
    log_stage("log_processed_pmids_complete")
    
    # Append new relations to existing file instead of overwriting
    log_stage("write_output", aggregated=len(aggregated), output=output_path)
    if output_path.exists() and len(aggregated) > 0:
        # Append mode: write new relations to the end of the file
        utils.ensure_dir(output_path)
        with output_path.open("a", encoding="utf-8") as fh:
            for row in aggregated:
                fh.write(json.dumps(row) + "\n")
        logger.info("Appended %d new relations to existing file", len(aggregated))
    elif len(aggregated) > 0:
        # First time: create new file
        utils.write_jsonl(output_path, aggregated)
        logger.info("Created new relations file with %d relations", len(aggregated))
    else:
        logger.info("No new relations to write")
    
    logger.info(
        "Finished: sentences=%d edges=%d filtered=%d aggregated=%d pmids_processed=%d",
        sentence_count,
        len(raw_results),
        len(filtered),
        len(aggregated),
        len(pmids_to_process),
    )


if __name__ == "__main__":
    main()

