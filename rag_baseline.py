# rag_baseline.py
#
# Minimal RAG baseline over PubMed talazoparib resistance abstracts
# using LlamaIndex, entirely in memory (no persistence).

import json
import argparse
import os
from typing import List, Dict

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter


def load_pubmed_jsonl(path: str) -> List[Dict]:
    """Load PubMed records from JSONL produced by fetch_pubmed.py."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_documents(pubmed_records: List[Dict]) -> List[Document]:
    """Turn each PubMed record into a LlamaIndex Document."""
    docs = []
    for rec in pubmed_records:
        pmid = rec.get("pmid")
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or ""
        year = rec.get("year")

        text = f"{title}\n\n{abstract}"

        metadata = {
            "pmid": pmid,
            "title": title,
            "year": year,
        }

        docs.append(Document(text=text, metadata=metadata))
    return docs


def build_index_in_memory(
    docs: List[Document],
    embed_model_name: str = "text-embedding-3-large",
) -> VectorStoreIndex:
    """Build a VectorStoreIndex in memory (no on-disk persistence)."""
    embed_model = OpenAIEmbedding(model=embed_model_name)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index


def run_query(
    index: VectorStoreIndex,
    query: str,
    model: str = "gpt-4.1-mini",
    top_k: int = 5,
):
    """Run a single RAG query and return the LlamaIndex response."""
    llm = OpenAI(model=model)
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
    )
    response = query_engine.query(query)
    return response


def print_response_with_sources(response):
    print("\n=== Answer ===\n")
    print(str(response))

    print("\n=== Sources ===\n")
    for i, node_with_score in enumerate(response.source_nodes, start=1):
        node = node_with_score.node
        score = node_with_score.score

        meta = node.metadata or {}
        pmid = meta.get("pmid", "N/A")
        title = (meta.get("title") or "")[:200]

        print(f"[{i}] PMID: {pmid}, score={score:.3f}")
        print(f"    Title: {title}")
        snippet = node.get_content()[:400].replace("\n", " ")
        print(f"    Snippet: {snippet}...")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/pubmed_talazoparib.jsonl",
        help="Path to PubMed JSONL file from fetch_pubmed.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI chat model for RAG generation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top similar chunks to retrieve",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional one-off query. If omitted, enter interactive mode.",
    )
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY in the environment.")

    print(f"Loading PubMed data from {args.input} ...")
    records = load_pubmed_jsonl(args.input)
    print(f"Loaded {len(records)} records.")

    print("Converting to LlamaIndex Documents...")
    docs = make_documents(records)

    print("Building in-memory index (this will call embeddings)...")
    index = build_index_in_memory(docs)

    if args.query:
        # One-shot query mode
        response = run_query(index, query=args.query, model=args.model, top_k=args.top_k)
        print_response_with_sources(response)
    else:
        # Interactive mode
        print("\nRAG baseline ready. Type a question or 'quit' to exit.")
        while True:
            q = input("\nQ> ").strip()
            if not q or q.lower() in {"q", "quit", "exit"}:
                break
            response = run_query(index, query=q, model=args.model, top_k=args.top_k)
            print_response_with_sources(response)


if __name__ == "__main__":
    main()
