# Biomedical Knowledge Graph Pipeline

A lightweight LLM-assisted workflow for extracting oncology resistance relations and writing them into a Biolink-compliant knowledge graph.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`
- `OPENAI_API_KEY` set to a model with JSON-mode chat completions

Configurable parameters lie in `pipeline/config.yaml`

## Gather Data: Fetch Pubmed Articles

The file `pipeline/fetch_pubmed.py` collects recent PubMed abstracts for “talazoparib resistance” and writes JSONL into `pipeline/data/`.

```bash
pipeline/fetch_pubmed.py --years 10 --output pipeline/data/pubmed_talazoparib.jsonl
```

## Pipeline

Run the pipeline script by calling

```bash
python pipeline/run_pipeline.py \
  --input pipeline/data/pubmed_talazoparib.jsonl \
  --output pipeline/data/relations.jsonl
```

The script does the following in order.

### Named-Entity Recognition (NER)
Load config + schema metadata, split abstracts into sentences, and queue every sentence for **named-entity recognition** through the shared OpenAI request worker. There are two files generated during this phase, both in `pipeline/named_entity_recognition/tmp/`:
1. `requests.json` - the requests being posted to the OpenAI API in parallel.
2. `results.json` - the results of the requests, returned not necessarily in the same order.

### Relation Extraction (RE)

Entities are normalized with `schema/idpolicy.yaml`, candidate pairs are filtered by the domains/ranges defined in `schema/model.yaml`, and the LLM is prompted to perform **relation extraction** using concise predicate descriptions derived from `schema/annotation_guideline.yaml`. 

Requests and results jsons of relation extraction similarly are logged to `pipeline/relation_extraction/tmp/`.

Each evaluated pair is logged to `pipeline/logs/relation_log.jsonl`. Low-confidence edges are dropped, duplicates (same subject–predicate–object) are merged, and results are written to `pipeline/data/relations.jsonl` with pmids, confidence, and model metadata.

## TL;DR Typical Run

```bash
export OPENAI_API_KEY=sk-...
python pipeline/fetch_pubmed.py --years 10 --output pipeline/data/pubmed_talazoparib.jsonl
python pipeline/run_pipeline.py \
  --input pipeline/data/pubmed_talazoparib.jsonl \
  --output pipeline/data/relations.jsonl
```

## Running the Interface
``` uvicorn server:app --reload --port 8000 ```
Then, open http://localhost:8000/ to see the interface. You can toggle between modes (RAG or KG), ask your own questions, and analyze the responses.