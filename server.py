# server.py
#
# Biomedical KG Pipeline – QA Demo backend

# server.py
#
# Biomedical KG Pipeline – QA Demo backend

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

from neo4j import GraphDatabase

# Add CT-Engine to path for semantic_parser imports
ROOT_DIR = Path(__file__).parent
CT_ENGINE_DIR = ROOT_DIR / "CT-Engine"
if str(CT_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(CT_ENGINE_DIR))

from semantic_parser.modules.cypher import create_action_registry

from qa_agent import KgQaAgent
from model.llm_client import LLMClient
from model.umls_client import UMLSClient
from schema import SchemaLoader
from utils import load_config


# -----------------------------
# Paths
# -----------------------------
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
    mode: str      # "rag", "kg", or "kg_agent"
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
CYTHER_REGISTRY = None
GENERATE_CYPHER_ACTION = None
NEO4J_DRIVER = None
KG_AGENT: Optional[KgQaAgent] = None

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://4e9b2d24.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "4MyZedIjW_ZdPpR7xzXx5ocCENECin_jZh6WThHHktU",
)
MAX_GRAPH_SOURCES = 5


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
# Cypher / Neo4j helpers
# -----------------------------
def init_cypher_components() -> None:
    """Initialize Neo4j driver, CT-Engine registry, and KG QA agent."""
    global CYTHER_REGISTRY, GENERATE_CYPHER_ACTION, NEO4J_DRIVER, KG_AGENT

    if NEO4J_DRIVER is None:
        NEO4J_DRIVER = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    CYTHER_REGISTRY = create_action_registry()
    GENERATE_CYPHER_ACTION = CYTHER_REGISTRY.get_action("GenerateCypher")
    if GENERATE_CYPHER_ACTION is None:
        raise RuntimeError("GenerateCypher action not available in registry")

    # Initialize dependencies for entity linking
    config = load_config()
    schema_loader = SchemaLoader()
    llm_client = LLMClient(config=config)
    umls_client = UMLSClient(
        api_key=config.get("umls", {}).get("api_key", ""),
        api_url=config.get("umls", {}).get("api_url", "https://uts-ws.nlm.nih.gov/rest"),
    )

    # Initialize KG-first QA agent with dependencies
    KG_AGENT = KgQaAgent(
        NEO4J_DRIVER,
        llm_client=llm_client,
        schema_loader=schema_loader,
        umls_client=umls_client,
    )


def execute_cypher_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a Cypher query against Neo4j."""
    if NEO4J_DRIVER is None:
        raise RuntimeError("Neo4j driver is not initialized")
    with NEO4J_DRIVER.session() as session:
        result = session.run(query, **(parameters or {}))
        return [record.data() for record in result]


def format_records_as_sources(records: List[Dict[str, Any]]) -> List[Source]:
    """Convert Neo4j rows to user-facing Source objects."""
    sources: List[Source] = []
    for idx, record in enumerate(records[:MAX_GRAPH_SOURCES]):
        subj = record.get("subject") or {}
        obj = record.get("object") or {}
        predicate = record.get("predicate") or "RELATED_TO"
        pmids = record.get("pmids") or []
        evidence = record.get("evidence") or []

        subj_label = subj.get("name") or subj.get("text") or subj.get("id") or "Subject"
        obj_label = obj.get("name") or obj.get("text") or obj.get("id") or "Object"
        label = f"{subj_label} --[{predicate}]→ {obj_label}"

        snippet = None
        if isinstance(evidence, list) and evidence:
            first = evidence[0]
            if isinstance(first, dict):
                snippet = first.get("sentence") or first.get("text")
            elif isinstance(first, str):
                snippet = first
        if not snippet:
            snippet = json.dumps(
                {"subject": subj_label, "predicate": predicate, "object": obj_label},
                indent=2,
            )

        sources.append(
            Source(
                label=label,
                pmid=str(pmids[0]) if pmids else None,
                snippet=snippet,
            )
        )
    return sources


def build_cypher_answer(records: List[Dict[str, Any]]) -> str:
    """Create a natural-language summary of Cypher results."""
    if not records:
        return "The knowledge graph does not contain matching facts for this question."

    def describe(node: Dict[str, Any]) -> str:
        if not node:
            return "Unknown entity"
        name = node.get("name") or node.get("text") or node.get("id") or "Unknown entity"
        node_class = node.get("class")
        return f"{name} ({node_class})" if node_class else name

    lines: List[str] = []
    for record in records[:MAX_GRAPH_SOURCES]:
        subj = describe(record.get("subject"))
        obj = describe(record.get("object"))
        predicate = record.get("predicate") or "RELATED_TO"
        pmids = record.get("pmids") or []
        sentence = None
        evidence = record.get("evidence") or []
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict) and item.get("sentence"):
                    sentence = item["sentence"]
                    break
                if isinstance(item, str):
                    sentence = item
                    break
        line = f"{subj} --[{predicate}]→ {obj}"
        if pmids:
            line += f" [PMID(s): {', '.join(pmids[:2])}]"
        if sentence:
            line += f" — Evidence: {sentence}"
        lines.append(line)

    summary = f"Knowledge graph returned {len(records)} result(s)."
    if lines:
        summary += " Key matches:\n- " + "\n- ".join(lines)
    return summary


def build_relation_context(records: List[Dict[str, Any]]) -> str:
    """Format graph triples into text snippets for LLM grounding."""
    context_lines: List[str] = []
    for record in records[:40]:
        subj = record.get("subject") or {}
        obj = record.get("object") or {}
        predicate = record.get("predicate") or "RELATED_TO"
        pmids = record.get("pmids") or []
        evidence = record.get("evidence") or []

        def fmt(node: Dict[str, Any]) -> str:
            name = node.get("name") or node.get("text") or node.get("id") or "UNKNOWN"
            n_cls = node.get("class")
            return f"{name} ({n_cls})" if n_cls else name

        line = f"{fmt(subj)} --[{predicate}]→ {fmt(obj)}"
        if pmids:
            line += f"; PMIDs: {', '.join(pmids[:3])}"

        sentence = None
        if isinstance(evidence, list):
            for ev in evidence:
                if isinstance(ev, dict) and ev.get("sentence"):
                    sentence = ev["sentence"]
                    break
                if isinstance(ev, str):
                    sentence = ev
                    break
        if sentence:
            line += f"; Evidence: {sentence}"

        context_lines.append(line)

    return "\n".join(context_lines)


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
    print("[Startup] Initializing Cypher components...")
    init_cypher_components()
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
# Mode: KG (Cypher-over-Neo4j)
# -----------------------------
def answer_with_kg(question: str) -> QueryResponse:
    """
    Convert the user's question into a Cypher query via CT-Engine actions,
    execute it against Neo4j, and summarize the results.
    """
    if GENERATE_CYPHER_ACTION is None:
        raise RuntimeError("Cypher action registry not initialized")

    action_output = GENERATE_CYPHER_ACTION.execute(query=question)
    if not action_output.success:
        return QueryResponse(
            answer=f"Failed to generate Cypher query: {action_output.error}",
            sources=[],
        )

    payload = action_output.result
    if isinstance(payload, dict):
        cypher_query = payload.get("query")
        parameters = payload.get("parameters", {})
    else:
        cypher_query = str(payload)
        parameters = {}

    if not cypher_query:
        return QueryResponse(
            answer="Cypher generator did not return a query.",
            sources=[],
        )

    try:
        records = execute_cypher_query(cypher_query, parameters)
    except Exception as exc:
        return QueryResponse(
            answer=f"Error executing Cypher query: {exc}",
            sources=[],
        )

    if not records:
        return QueryResponse(
            answer="The knowledge graph does not contain matching facts for this question.",
            sources=[],
        )

    context = build_relation_context(records)
    prompt = f"""
You are a biomedical assistant. Use ONLY the provided relations to answer the user's question.
If the relations do not support an answer, explicitly say so.

Relations:
{context}

User question: {question}

Provide a concise answer grounded in the relations above, citing genes, drugs, and PMIDs when helpful.
"""

    llm = OpenAI(model="gpt-4.1-mini")
    try:
        completion = llm.complete(prompt)
        answer_text = getattr(completion, "text", str(completion)).strip()
    except Exception as exc:
        answer_text = f"Failed to synthesize answer from relations: {exc}"

    return QueryResponse(
        answer=answer_text,
        sources=format_records_as_sources(records),
    )


# -----------------------------
# Mode: KG-first QA agent
# -----------------------------
def answer_with_kg_agent(question: str) -> QueryResponse:
    """
    Use the KG-first QA agent (Anchor → Expand → Explain) instead of raw Cypher.
    """
    if KG_AGENT is None:
        raise RuntimeError("KG QA agent not initialized")

    answer_text, records, _debug = KG_AGENT.answer(question)
    return QueryResponse(
        answer=answer_text,
        sources=format_records_as_sources(records),
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
        return answer_with_kg(req.question)
    elif mode == "kg_agent":
        return answer_with_kg_agent(req.question)
    else:
        return QueryResponse(
            answer=f"Unknown mode '{req.mode}'. Use 'rag', 'kg', or 'kg_agent'.",
            sources=[],
        )


@app.get("/")
def index():
    """Serve the frontend page."""
    return FileResponse(FRONTEND_DIR / "index.html")
