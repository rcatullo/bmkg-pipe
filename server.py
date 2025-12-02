# server.py
#
# Biomedical KG Pipeline – QA Demo backend
#
# - POST /query  with JSON { "mode": "rag" | "kg", "question": "..." }
#   - "rag": Generic RAG over PubMed abstracts (pubmed_talazoparib.jsonl)
#   - "kg" : LLM-over-KG using relations.jsonl
#            * deterministic relation selection based on question + schema
#            * answer grounded in those relations
#            * sources = sentence snippets from those same relations, each with a PMID
#
# - GET /         serves frontend/index.html
# - GET /health   simple health check

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter


# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
FRONTEND_DIR = ROOT_DIR / "frontend"

PUBMED_PATH = DATA_DIR / "pubmed_talazoparib.jsonl"
RELATIONS_PATH = DATA_DIR / "relations.jsonl"


# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Biomedical KG Pipeline – QA Demo")

# Permissive CORS for local dev / demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request / Response models
# -----------------------------
class QueryRequest(BaseModel):
    mode: str      # "rag" or "kg"
    question: str


class Source(BaseModel):
    label: str
    pmid: Optional[str] = None
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# -----------------------------
# Global state
# -----------------------------
RAG_INDEX: Optional[VectorStoreIndex] = None
RELATIONS: List[Dict] = []


# -----------------------------
# Helpers: load PubMed data
# -----------------------------
def load_pubmed_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        print(f"[RAG] PubMed file not found at {path}")
        return records

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_documents(pubmed_records: List[Dict]) -> List[Document]:
    """
    Turn PubMed JSONL into LlamaIndex Documents.
    Handles several possible key variants for title/abstract.
    """
    docs: List[Document] = []
    empty = 0

    for rec in pubmed_records:
        pmid = rec.get("pmid") or rec.get("PMID")

        title = (
            rec.get("title")
            or rec.get("Title")
            or rec.get("article_title")
            or rec.get("ArticleTitle")
            or ""
        )

        abstract = (
            rec.get("abstract")
            or rec.get("Abstract")
            or rec.get("abstract_text")
            or rec.get("AbstractText")
            or ""
        )

        # Sometimes abstract is a list of strings / segments
        if isinstance(abstract, list):
            abstract = " ".join(str(x) for x in abstract)

        text = f"{title}\n\n{abstract}".strip()
        if not text:
            empty += 1
            continue

        metadata = {
            "pmid": pmid,
            "title": title,
            "year": rec.get("year") or rec.get("Year"),
        }
        docs.append(Document(text=text, metadata=metadata))

    print(f"[RAG] Built {len(docs)} documents, skipped {empty} empty.")
    return docs


def build_rag_index() -> VectorStoreIndex:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set")

    records = load_pubmed_records(PUBMED_PATH)
    print(f"[RAG] Loaded {len(records)} PubMed records from {PUBMED_PATH}.")

    docs = make_documents(records)
    if not docs:
        raise RuntimeError("[RAG] No non-empty documents were built from PubMed data.")

    print("[RAG] Building in-memory index...")
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"[RAG] Created {len(nodes)} nodes/chunks.")

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    print("[RAG] Index built.")
    return index


# -----------------------------
# Helpers: load KG relations
# -----------------------------
def load_relations(path: Path) -> List[Dict]:
    rels: List[Dict] = []
    if not path.exists():
        print(f"[KG] relations.jsonl not found at {path}")
        return rels

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rels.append(json.loads(line))

    print(f"[KG] Loaded {len(rels)} relations from {path}.")
    return rels


# -----------------------------
# Startup: build index + load KG
# -----------------------------
@app.on_event("startup")
def on_startup():
    global RAG_INDEX, RELATIONS
    print("[Startup] Building RAG index...")
    RAG_INDEX = build_rag_index()
    print("[Startup] Loading KG relations...")
    RELATIONS = load_relations(RELATIONS_PATH)
    print("[Startup] Ready.")


# -----------------------------
# Mode: RAG
# -----------------------------
def answer_with_rag(question: str) -> QueryResponse:
    if RAG_INDEX is None:
        raise RuntimeError("RAG index not initialized")

    llm = OpenAI(model="gpt-4.1-mini")
    query_engine = RAG_INDEX.as_query_engine(
        llm=llm,
        similarity_top_k=5,
    )
    resp = query_engine.query(question)

    sources: List[Source] = []
    for node_with_score in resp.source_nodes:
        node = node_with_score.node
        meta = node.metadata or {}
        pmid = meta.get("pmid")
        title = (meta.get("title") or "").strip()
        snippet = node.get_content()[:400].replace("\n", " ")
        label = f"PMID {pmid} – {title}" if pmid else (title or "Source")
        sources.append(
            Source(
                label=label,
                pmid=str(pmid) if pmid is not None else None,
                snippet=snippet,
            )
        )

    return QueryResponse(
        answer=str(resp),
        sources=sources,
    )


# -----------------------------
# KG relation selector (question-aware but general)
# -----------------------------
def select_kg_relations_for_question(
    question: str,
    relations: List[Dict],
    limit: int = 40,
) -> List[Dict]:
    """
    Heuristic selector for KG relations based on the question text.

    Goal:
    - Use simple signals (talazoparib/PARPi mentions, resistance words, etc.)
      and schema hints (subject class, predicate) to pick a focused subset
      of relations as context for the LLM.
    - Simple, robust, and deterministic: good for a demo + paper.
    """
    q = question.lower()

    # 1) Base filter: keep relations that mention talazoparib / PARPi / PARP inhibitor
    base_rels: List[Dict] = []
    for r in relations:
        subj = r.get("subject", {})
        obj = r.get("object", {})
        subj_text = str(subj.get("text", "")).lower()
        obj_text = str(obj.get("text", "")).lower()
        if (
            "talazoparib" in subj_text
            or "talazoparib" in obj_text
            or "parpi" in subj_text
            or "parpi" in obj_text
            or "parp inhibitor" in subj_text
            or "parp inhibitor" in obj_text
        ):
            base_rels.append(r)

    # If nothing clearly talazoparib/PARPi-related, just return a small sample
    if not base_rels:
        return relations[:limit]

    rels = base_rels

    # 2) If the question mentions genes, focus on gene-like subjects
    if "gene" in q or "genes" in q:
        rels_gene = [
            r
            for r in rels
            if r.get("subject", {}).get("class") in ("Gene", "Protein")
        ]
        if rels_gene:
            rels = rels_gene

    # 3) If the question mentions resistance, focus on resistance-related predicates
    if "resistance" in q or "resistant" in q:
        resistance_preds = {
            "confers_resistance_to",
            "upregulated_in_resistance",
            "downregulated_in_resistance",
            "associated_with_resistance_to",
            "predicts_resistance_to",
        }
        rels_res = [r for r in rels if r.get("predicate") in resistance_preds]
        if rels_res:
            rels = rels_res

    # (Future: you can add similar blocks for "combination", "biomarker", etc.)

    return rels[:limit]


# -----------------------------
# Mode: KG (LLM-over-KG)
# -----------------------------
def kg_answer_with_llm(question: str) -> QueryResponse:
    """
    Use a question-aware subset of KG relations as context and let an LLM
    generate a natural-language answer grounded only in those relations.

    Sources are *exact sentence snippets* from those same relations,
    each labeled with its PMID. This is the most faithful and interpretable
    evidence from the KG.
    """
    if not RELATIONS:
        return QueryResponse(
            answer="Knowledge graph is not available (no relations loaded).",
            sources=[],
        )

    # 1) Select relevant relations based on question + simple schema heuristics
    rel_for_context = select_kg_relations_for_question(question, RELATIONS, limit=40)

    if not rel_for_context:
        return QueryResponse(
            answer="The knowledge graph does not contain any relations clearly related to this question.",
            sources=[],
        )

    # 2) Format KG context for the LLM
    lines: List[str] = []
    for r in rel_for_context:
        subj = r.get("subject", {})
        obj = r.get("object", {})
        subj_text = subj.get("text", "UNKNOWN_SUBJECT")
        subj_class = subj.get("class")
        obj_text = obj.get("text", "UNKNOWN_OBJECT")
        obj_class = obj.get("class")
        pred = r.get("predicate", "UNKNOWN_PREDICATE")
        pmids = r.get("pmids", [])
        pmid_str = ", ".join(pmids) if pmids else "N/A"
        line = f"{subj_text} ({subj_class}) --[{pred}]→ {obj_text} ({obj_class}); PMIDs: {pmid_str}"
        lines.append(line)

    kg_context = "\n".join(lines)

    # 3) Ask LLM to answer using ONLY these relations
    llm = OpenAI(model="gpt-4.1-mini")
    prompt = f"""
You are a biomedical assistant. You are given a set of knowledge-graph relations
about talazoparib and related entities.

Each relation has the form:
  SUBJECT (class) --[predicate]→ OBJECT (class); PMIDs: ...

Use ONLY these relations as your source of truth. Do not invent new genes, drugs,
mechanisms, or clinical outcomes that are not supported by these relations.
If the answer is not clearly supported by the relations, explicitly say that the
knowledge graph does not contain enough information to answer the question.

Relations:
{kg_context}

User question: {question}

Now, answer the question in clear, concise natural language, referencing genes,
drugs, pathways, or biomarkers from the relations where appropriate. If useful,
you may mention PMIDs as evidence.
"""
    completion = llm.complete(prompt)
    answer_text = getattr(completion, "text", str(completion)).strip()

    # 4) Build sources directly from the SAME relations (sentence-level evidence)
    max_sources = 10
    seen_pairs = set()  # (pmid, snippet) to dedup
    source_objs: List[Source] = []

    for r in rel_for_context:
        sentences = r.get("sentences", [])
        if not sentences:
            # Fallback: relation-level only if no sentence detail exists
            pmids = r.get("pmids", [])
            pmid = pmids[0] if pmids else None
            label = f"Snippet from PMID {pmid}" if pmid else "Snippet from KG relation (no PMID)"
            key = (pmid, None)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            source_objs.append(
                Source(
                    label=label,
                    pmid=str(pmid) if pmid is not None else None,
                    snippet=None,
                )
            )
        else:
            for s in sentences:
                pmid = s.get("pmid") or (r.get("pmids") or [None])[0]
                sentence = s.get("sentence")
                if not sentence:
                    continue
                key = (pmid, sentence)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                label = f"Snippet from PMID {pmid}" if pmid else "Snippet from KG"
                source_objs.append(
                    Source(
                        label=label,
                        pmid=str(pmid) if pmid is not None else None,
                        snippet=sentence,
                    )
                )
                if len(source_objs) >= max_sources:
                    break

        if len(source_objs) >= max_sources:
            break

    return QueryResponse(
        answer=answer_text,
        sources=source_objs,
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    mode = req.mode.lower()
    if mode == "rag":
        return answer_with_rag(req.question)
    elif mode == "kg":
        return kg_answer_with_llm(req.question)
    else:
        return QueryResponse(
            answer=f"Unknown mode '{req.mode}'. Use 'rag' or 'kg'.",
            sources=[],
        )


@app.get("/")
def index():
    """Serve the frontend page."""
    return FileResponse(FRONTEND_DIR / "index.html")
