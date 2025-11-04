# Talazoparib Resistance Knowledge Graph

This repository contains a knowledge graph extraction pipeline for studying mechanisms of Talazoparib resistance. It is **schema-only**: no ETL or code lives here yet. The goal is to keep the conceptual model, mappings, and guidelines clean and versioned so they can be reused by multiple pipelines (Neo4j, RDF, LLM-based IE, etc.).

## Directory structure

```text
schema/
├── model.yaml
├── constraints/
│   └── neo4j.cypher
├── tests/
├── annotation_guideline.yaml
├── idpolicy.yaml
├── jsonschema/
│   └── model.schema.json
└── mappings/
    ├── predicates.yaml
    ├── prefixes.yaml
    └── biolink.yaml
````

### `model.yaml`

Canonical **schema definition**.

* Defines:

  * Node classes (e.g., `Drug`, `Gene`, `RNA`, `Variant`, `CancerType`, `Pathway`, `Phenotype`, `Paper`).
  * Edge/predicate types (e.g., `UPREGULATES`, `DOWNREGULATES`, `BINDS`, `PART_OF`, `CAUSES_RESISTANCE_TO`, `TREATS`, `ASSOCIATED_WITH_VARIANT`).
  * Domains / ranges for predicates.
  * Biolink / RO alignment (via `is_a`, `exact_mappings`, etc. where applicable).
* This file is the **single source of truth** for:

  * What exists in the KG.
  * How classes and predicates are semantically defined.

If you change the conceptual model (add/remove classes or relations, change domains/ranges), do it here first.

---

### `mappings/`

Mapping files that expose the semantics in `model.yaml` in a more “operational” way for tooling and ETL.

#### `mappings/biolink.yaml`

* Maps local classes to **Biolink Model** categories.
* Example: `Gene` → `biolink:Gene`, `RNA` → `biolink:RNAProduct`.
* Used for:

  * Exporting to Biolink-compatible formats.
  * Interoperability with external KGs / tools.

#### `mappings/predicates.yaml`

* Maps local predicates to:

  * Biolink slots (`biolink:`),
  * OBO Relation Ontology (RO) term IRIs when available,
  * (Optionally) source predicate sets like SemMedDB.
* Used for:

  * Normalizing IE outputs (e.g., mapping many surface verbs into a small predicate set).
  * Exporting relations as standard RO/Biolink predicates.

#### `mappings/prefixes.yaml`

* Central place for **ID prefixes → IRIs**.
* Example: `HGNC`, `ENSEMBL`, `MIRBASE`, `RXNORM`, `NCIT`, `HPO`, etc.
* Used when:

  * Converting internal IDs to CURIEs/URIs.
  * Exporting to RDF/JSON-LD.

---

### `idpolicy.yaml`

Implementation-level **identifier and merge policy**.

* For each class (e.g., `Drug`, `Gene`, `RNA`, …) defines:

  * `primary` identifier (e.g., `rxnorm_id` for drugs, `hgnc_id` for genes).
  * Optional `alternates` (e.g., `chebi_id`, `entrez_id`, `ensembl_gene_id`).
* Used by:

  * ETL loaders to decide which property to use in `MERGE` / upserts.
  * Constraint generation (e.g., unique constraints in Neo4j).

Think of this as:

> “Given the schema, how do we **uniquely identify and merge** nodes in a concrete store?”

---

### `constraints/neo4j.cypher`

Database-level constraints for a **Neo4j** deployment of this schema.

* Typical contents:

  * `CREATE CONSTRAINT ... REQUIRE <primary_id> IS UNIQUE` for each node type.
  * Helpful indexes (e.g., `Paper.pmid`).
* Generated or maintained based on `idpolicy.yaml` and `model.yaml`.

Apply this file to a Neo4j instance **before** loading data to:

* Prevent duplicate nodes.
* Get consistent performance for lookups/merges.

---

### `annotation_guideline.yaml`

Per-predicate **annotation guidelines** (“relation cards”) for humans and LLMs.

For each predicate (e.g., `UPREGULATES`, `DOWNREGULATES`, `BINDS`, etc.), contains:

* `definition` – 1–2 line description of what the relation means.
* `examples`:

  * `positive` – PubMed-style sentence that *should* be labeled with this relation.
  * `negative` – sentence that looks related but *should not* be labeled with this relation (hedged, correlational, too vague).
* `decision_rule`:

  * `accept_if` – bullet rules for when to accept this label.
  * `reject_if` – bullet rules for when to reject.

Use this file for:

* **Human annotators** building gold-standard data.
* **LLM prompts** as structured, machine-readable “schema cards” for relation extraction and LLM-as-judge.

---

### `jsonschema/model.schema.json`

JSON Schema for validating extracted **triple-like outputs**.

* Defines the expected shape of extracted relations, e.g.:

  * `subj`, `subj_type`, `obj`, `obj_type`, `rel`.
  * Allowed values for `subj_type` / `obj_type`.
  * Allowed values for `rel` (predicate enum).
  * Evidence properties (e.g., `pmid`, `sentence_span`, `confidence`).

Typical usage:

1. LLM / IE system emits candidate relations as JSON.
2. Output is validated against this schema.
3. Only valid relations are passed to the loader (and then to Neo4j).

This is your **guardrail** between noisy extraction and the KG.


## How these pieces fit together in the pipeline

A typical flow using this repo looks like:

1. **Schema & mappings**

   * `model.yaml` defines classes & predicates.
   * `mappings/biolink.yaml`, `mappings/predicates.yaml`, `mappings/prefixes.yaml` expose alignments and ID IRIs.

2. **Extraction & validation**

   * An IE / LLM pipeline reads `annotation_guideline.yaml` to understand predicates.
   * The pipeline outputs candidate relations as JSON.
   * `jsonschema/model.schema.json` validates that outputs are well-formed and use valid types/relations.

3. **ID resolution & merging**

   * `idpolicy.yaml` tells the loader which IDs to use as primary keys (e.g., `hgnc_id` for genes).
   * The loader constructs `MERGE` patterns based on this.

4. **Database constraints**

   * `constraints/neo4j.cypher` is applied to Neo4j to enforce uniqueness and indexes.

5. **Tests & CI**

   * `tests/` verifies schema consistency and helps catch breaking changes early.

## How to extend

* **Add a new node type**

  1. Add class to `model.yaml`.
  2. Add a Biolink mapping in `mappings/biolink.yaml`.
  3. Add ID policy for it in `idpolicy.yaml`.
  4. Add a constraint in `constraints/neo4j.cypher`.
  5. (Optional) extend `model.schema.json` and tests.

* **Add a new predicate**

  1. Define it in `model.yaml` with domain/range.
  2. Add Biolink/RO mapping in `mappings/predicates.yaml`.
  3. Add an entry in `annotation_guideline.yaml` (definition, positive/negative examples, decision rules).
  4. Add it to the `rel` enum in `jsonschema/model.schema.json`.
  5. Update tests to expect it.

## Status

This repo currently contains:

* A **first-pass KG schema** for Talazoparib resistance (RNA-centric).
* Predicate definitions aligned to Biolink / RO.
* ID policies for key biomedical entity types.
* Annotation guidelines for consistent extraction.
* JSON Schema + Neo4j constraints to keep the KG clean.

As the project evolves, we’ll:

* Refine classes/predicates (e.g., add sensitivity relations, context qualifiers).
* Tighten annotation guidelines based on error analysis.
* Add tests to keep everything consistent as the schema grows.
