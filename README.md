# Talazoparib Knowledge Graph Pipeline

A lightweight LLM-assisted workflow for extracting oncology resistance relations and writing them into a Biolink-compliant knowledge graph.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`
- `OPENAI_API_KEY` set to a model with JSON-mode chat completions

Optional: adjust `pipeline/config.yaml` to point to a different OpenAI model name.

## Data Flow

1. `pipeline/fetch_pubmed.py` collects recent PubMed abstracts for “talazoparib resistance” and writes JSONL into `pipeline/data/`.
2. `pipeline/run_pipeline.py` loads config + schema metadata, splits abstracts into sentences, and batches sentences so one LLM call can tag entities for multiple sentences.
3. Entities are normalized with `schema/idpolicy.yaml`, candidate pairs are filtered by the domains/ranges defined in `schema/model.yaml`, and the classifier prompts the LLM using concise predicate descriptions derived from `schema/annotation_guideline.yaml`.
4. Each evaluated pair is logged to `pipeline/data/relation_log.jsonl`. Low-confidence edges are dropped, duplicates (same subject–predicate–object) are merged, and results are written to `pipeline/data/relations.jsonl` with pmids, confidence, and model metadata.

## Typical Run

```bash
export OPENAI_API_KEY=sk-...
python pipeline/fetch_pubmed.py --output pipeline/data/pubmed_talazoparib.jsonl
python pipeline/run_pipeline.py \
  --input pipeline/data/pubmed_talazoparib.jsonl \
  --output pipeline/data/relations.jsonl \
  --log pipeline/data/relation_log.jsonl \
  --log-level INFO
```

Useful flags:

- `--entity-batch`: number of sentences grouped per entity-extraction prompt (default 5).
- `--threshold`: minimum confidence for keeping a relation (default 0.55).
- `--log`/`--output`: alternate paths for logs or aggregated results.

## Outputs

- `pipeline/data/relation_log.jsonl`: raw per-pair records with pmid, sentence, subject/object IDs, predicate, confidence, prompt/model metadata, and timestamps.
- `pipeline/data/relations.jsonl`: deduplicated `(subject, predicate, object)` edges that pass the confidence threshold, each with merged evidence sentences.
