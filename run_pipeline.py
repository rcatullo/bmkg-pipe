import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

from model.llm_client import LLMClient
from named_entity_recognition import NamedEntityRecognition
from relation_extraction import RelationExtraction
from schema import SchemaLoader, Normalizer
import utils as utils
from utils import PostProcessor, PredicateFilter

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


def build_components():
    config = utils.load_config()
    schema = SchemaLoader()
    llm = LLMClient(config=config)
    from model.umls_client import UMLSClient
    umls = UMLSClient(
        api_key=config.get("umls", {}).get("api_key", ""),
        api_url=config.get("umls", {}).get("api_url", "https://uts-ws.nlm.nih.gov/rest"),
    )
    normalizer = Normalizer(schema, llm_client=llm, umls_client=umls)
    ner = NamedEntityRecognition(schema, normalizer, llm, config)
    predicate_filter = PredicateFilter(schema)
    re = RelationExtraction(llm, config, schema=schema)
    postprocessor = PostProcessor()
    return ner, predicate_filter, re, postprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Biomedical KG pipeline")
    parser.add_argument("--input", dest="input_file", help="Override input JSONL path")
    parser.add_argument("--output", dest="output_file", help="Override output JSONL path")
    parser.add_argument("--log", dest="log_file", help="Override relation log JSONL path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = utils.load_config()
    if args.input_file:
        config["data"]["input_file"] = args.input_file
    if args.output_file:
        config["data"]["output_file"] = args.output_file
    if args.log_file:
        config["logging"]["relation_log_file"] = args.log_file
    configure_logging(config)

    log_stage("build_components")
    ner, predicate_filter, re, postprocessor = build_components()
    log_stage("build_components_complete")
    input_path = Path(config["data"]["input_file"])
    relation_log_path = Path(config["logging"]["relation_log_file"])
    raw_results = []
    sentence_count = 0
    entity_total = 0

    logger.info(
        "Starting pipeline input=%s output=%s log=%s",
        input_path,
        config["data"]["output_file"],
        relation_log_path,
    )

    batch_size = int(config.get("data", {}).get("batch_size", 1))
    sentences = list(utils.load_sentences(input_path, batch_size=batch_size))
    total_sentences = sum(len(sentence.sentence_ids) for sentence in sentences)
    log_stage(
        "entity_queue",
        sentence_batches=len(sentences),
        total_sentences=total_sentences,
        batch_size=batch_size,
    )
    ner.add_sentences(sentences)
    log_stage("entity_execute", sentences=ner.total_sentences)
    entity_mapping = ner.run()
    log_stage("entity_results", sentences=len(entity_mapping))

    for sentence in sentences:
        sentence_count += len(sentence.sentence_ids)
        entities = entity_mapping.get((sentence.pmid, sentence.sentence_id), [])
        if not entities:
            logger.debug(
                "No entities for pmid=%s sentence_id=%s", sentence.pmid, sentence.sentence_id
            )
            continue
        entity_total += len(entities)
        allowed_preds = predicate_filter.for_entities(entities)
        if not allowed_preds:
            continue
        log_stage(
            "relation_extraction",
            pmid=sentence.pmid,
            sentence_id=sentence.sentence_id,
            pair_count=len(allowed_preds),
        )
        re.add_sentence(
            {
                "pmid": sentence.pmid,
                "sentence_id": sentence.sentence_id,
                "sentence_ids": sentence.sentence_ids,
                "text": sentence.text,
            },
            entities,
            allowed_preds,
        )
        if sentence_count and sentence_count % 50 == 0:
            logger.info(
                "Processed %d sentences (%d entities, %d predicate groups so far)",
                sentence_count,
                entity_total,
                re.total_requests,
            )

    log_stage("relation_execute", total_pairs=re.total_requests)
    for classification in re.run():
        utils.log_result(classification, relation_log_path)
        raw_results.append(classification)

    postprocessor.threshold = config["relation_extraction"]["threshold"]
    log_stage("postprocess_filter", total=len(raw_results))
    filtered = postprocessor.filter(raw_results)
    log_stage("postprocess_aggregate", filtered=len(filtered))
    aggregated = postprocessor.aggregate(filtered)
    log_stage("write_output", aggregated=len(aggregated), output=config["data"]["output_file"])
    utils.write_jsonl(Path(config["data"]["output_file"]), aggregated)
    logger.info(
        "Finished: sentences=%d edges=%d filtered=%d aggregated=%d",
        sentence_count,
        len(raw_results),
        len(filtered),
        len(aggregated),
    )


if __name__ == "__main__":
    main()

