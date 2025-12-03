import argparse
import json
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

URI = "neo4j+s://4e9b2d24.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "4MyZedIjW_ZdPpR7xzXx5ocCENECin_jZh6WThHHktU"
DATABASE = "neo4j"

def parse_sentences(sentences: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Extract plain sentence strings from the 'sentences' field."""
    if not sentences:
        return []
    return [
        s.get("sentence")
        for s in sentences
        if isinstance(s, dict) and s.get("sentence")
    ]


def check_relation_exists(tx, s_id: str, o_id: str, rel_type: str) -> bool:
    """Check if a relation already exists between subject and object."""
    query = f"""
    MATCH (s)-[r:{rel_type}]->(o)
    WHERE s.id = $s_id AND o.id = $o_id
    RETURN r LIMIT 1
    """
    result = tx.run(query, s_id=s_id, o_id=o_id)
    return result.single() is not None


def import_triple(tx, record: Dict[str, Any]) -> bool:
    """
    Import a single JSONL record as nodes + relationship into Neo4j.
    Returns True if the relation was imported (new), False if it already existed.
    """

    subj = record.get("subject") or {}
    obj = record.get("object") or {}
    predicate = record.get("predicate")

    if not subj or not obj or not predicate:
        return False

    rel_type = predicate.upper()
    confidence = record.get("confidence")

    s_class = subj.get("class")
    o_class = obj.get("class")

    s_text = subj.get("text")
    o_text = obj.get("text")

    s_id = subj.get("id") or f"{s_class}:{s_text}"
    o_id = obj.get("id") or f"{o_class}:{o_text}"

    # Check if relation already exists
    if check_relation_exists(tx, s_id, o_id, rel_type):
        return False

    s_name = subj.get("canonical_form") or s_text
    o_name = obj.get("canonical_form") or o_text
    
    s_umls_cui = subj.get("umls_cui")
    o_umls_cui = obj.get("umls_cui")

    pmids = record.get("pmids") or []
    pmid = pmids[0] if pmids else None

    sentences = parse_sentences(record.get("sentences"))
    timestamp = record.get("timestamp")
    model_name = record.get("model_name")
    model_version = record.get("model_version")

    s_create_props = ["s.name = $s_name", "s.text = $s_text"]
    s_match_props = ["s.name = coalesce($s_name, s.name)", "s.text = coalesce($s_text, s.text)"]
    if s_umls_cui:
        s_create_props.append("s.umls_cui = $s_umls_cui")
        s_match_props.append("s.umls_cui = coalesce($s_umls_cui, s.umls_cui)")
    
    o_create_props = ["o.name = $o_name", "o.text = $o_text"]
    o_match_props = ["o.name = coalesce($o_name, o.name)", "o.text = coalesce($o_text, o.text)"]
    if o_umls_cui:
        o_create_props.append("o.umls_cui = $o_umls_cui")
        o_match_props.append("o.umls_cui = coalesce($o_umls_cui, o.umls_cui)")

    query = f"""
    MERGE (s:{s_class} {{id: $s_id}})
      ON CREATE SET {', '.join(s_create_props)}
      ON MATCH SET {', '.join(s_match_props)}
    MERGE (o:{o_class} {{id: $o_id}})
      ON CREATE SET {', '.join(o_create_props)}
      ON MATCH SET {', '.join(o_match_props)}
    MERGE (s)-[r:{rel_type}]->(o)
    SET
      r.pmid          = coalesce($pmid, r.pmid),
      r.pmids         = coalesce($pmids, r.pmids),
      r.confidence    = coalesce($confidence, r.confidence),
      r.extractor     = coalesce(r.extractor, 'LLM'),
      r.created_at    = coalesce($timestamp, r.created_at),
      r.model_name    = coalesce($model_name, r.model_name),
      r.model_version = coalesce($model_version, r.model_version),
      r.sentences     = coalesce($sentences, r.sentences)
    """

    params = {
        "s_id": s_id,
        "s_name": s_name,
        "s_text": s_text,
        "o_id": o_id,
        "o_name": o_name,
        "o_text": o_text,
        "pmid": pmid,
        "pmids": pmids,
        "confidence": confidence,
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": model_version,
        "sentences": sentences,
    }
    if s_umls_cui:
        params["s_umls_cui"] = s_umls_cui
    if o_umls_cui:
        params["o_umls_cui"] = o_umls_cui
    
    tx.run(query, **params)
    return True


def import_file(path: str) -> None:
    """Stream a JSONL file and import each triple into Neo4j."""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    total = 0
    imported = 0
    skipped = 0

    with driver.session(database=DATABASE) as session:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON error, skipping: {e}")
                    continue

                try:
                    was_imported = session.execute_write(import_triple, record)
                    if was_imported:
                        imported += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"Line {line_num}: Neo4j error, skipping: {e}")

    driver.close()
    print(f"Done. Processed {total} lines, imported {imported} new relations, skipped {skipped} existing relations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import JSONL triples into Neo4j.")
    parser.add_argument("relations_path", help="Path to the relations.jsonl file with triples")
    args = parser.parse_args()

    import_file(args.relations_path)
