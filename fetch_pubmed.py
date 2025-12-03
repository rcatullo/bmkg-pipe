import argparse
import datetime as dt
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
import yaml

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def esearch_ids(query: str, mindate: str, maxdate: str, batch_size: int, max_articles: Optional[int] = None) -> List[str]:
    ids: List[str] = []
    retstart = 0
    count = None
    fetched = 0
    while True:
        # If max_articles is set, do not request beyond that limit
        curr_batch_size = batch_size
        if max_articles is not None:
            remaining = max_articles - fetched
            if remaining <= 0:
                break
            curr_batch_size = min(curr_batch_size, remaining)
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": curr_batch_size,
            "retstart": retstart,
            "datetype": "pdat",
            "mindate": mindate,
            "maxdate": maxdate,
        }
        resp = requests.get(f"{EUTILS_BASE}esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        if count is None:
            count_text = root.findtext("Count", default="0")
            count = int(count_text)
        batch_ids = [elem.text for elem in root.findall(".//IdList/Id") if elem.text]
        if not batch_ids:
            break
        ids.extend(batch_ids)
        retstart += len(batch_ids)
        fetched += len(batch_ids)
        if retstart >= count:
            break
        if max_articles is not None and fetched >= max_articles:
            break
        time.sleep(0.34)
    if max_articles is not None:
        ids = ids[:max_articles]
    return ids


def parse_article(article: ET.Element) -> Dict[str, object]:
    pmid = article.findtext(".//PMID")
    title = article.findtext(".//Article/ArticleTitle") or ""
    abstract_nodes = article.findall(".//Abstract/AbstractText")
    abstract_parts = []
    for node in abstract_nodes:
        text = node.text or ""
        label = node.get("Label")
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(part.strip() for part in abstract_parts if part.strip())
    pub_date = article.find(".//Article/Journal/JournalIssue/PubDate")
    year = pub_date.findtext("Year") if pub_date is not None else None
    if not year:
        medline_date = pub_date.findtext("MedlineDate") if pub_date is not None else ""
        if medline_date:
            year = medline_date.split(" ")[0]
    journal = article.findtext(".//Article/Journal/Title")
    mesh_terms = [
        node.text for node in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName") if node.text
    ]
    authors = []
    for author in article.findall(".//AuthorList/Author"):
        last = author.findtext("LastName")
        fore = author.findtext("ForeName")
        collective = author.findtext("CollectiveName")
        if collective:
            authors.append(collective)
        elif last or fore:
            authors.append(", ".join(filter(None, [last, fore])))
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "meta": {"year": year, "journal": journal, "mesh_terms": mesh_terms, "authors": authors},
    }


def efetch_records(ids: List[str], batch_size: int, max_records: Optional[int] = None) -> Iterable[dict]:
    yielded = 0
    for batch in chunked(ids, batch_size):
        if max_records is not None:
            remaining = max_records - yielded
            if remaining <= 0:
                break
            batch = batch[:remaining]
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "rettype": "abstract",
            "id": ",".join(batch),
        }
        resp = requests.get(f"{EUTILS_BASE}efetch.fcgi", params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            if max_records is not None and yielded >= max_records:
                break
            yield parse_article(article)
            yielded += 1
        if max_records is not None and yielded >= max_records:
            break
        time.sleep(0.34)


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
            # Resolve relative paths
            if "logging" in config:
                logging = config["logging"]
                for key in ["fetched_queries_log", "processed_pmids_log"]:
                    if key in logging:
                        path = Path(logging[key])
                        if not path.is_absolute():
                            logging[key] = str(Path(__file__).resolve().parent / path)
            return config
    return {}


def load_fetched_queries(fetched_queries_log_path: Path) -> set:
    """Load set of already fetched queries from log file."""
    fetched_queries = set()
    if fetched_queries_log_path.exists():
        print(f"Loading fetched queries from {fetched_queries_log_path}...")
        with fetched_queries_log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    query = record.get("query")
                    if query:
                        fetched_queries.add(query)
                except json.JSONDecodeError:
                    continue
        print(f"  Found {len(fetched_queries)} already fetched queries")
    return fetched_queries


def log_fetched_query(fetched_queries_log_path: Path, query: str, timestamp: str):
    """Log a fetched query to the log file."""
    fetched_queries_log_path.parent.mkdir(parents=True, exist_ok=True)
    with fetched_queries_log_path.open("a", encoding="utf-8") as fh:
        record = {
            "query": query,
            "timestamp": timestamp,
        }
        fh.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="Single query to use (overrides default multi-query)")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--esearch-batch", type=int, default=100)
    parser.add_argument("--efetch-batch", type=int, default=20)
    parser.add_argument("--output", default="data/pubmed_talazoparib.jsonl")
    parser.add_argument("--max-articles-per-query", type=int, default=10, help="Maximum articles per query")
    args = parser.parse_args()

    config = load_config()
    
    # Load queries from config or use command line argument
    if args.query:
        queries = [args.query]
    else:
        queries = config.get("pubmed", {}).get("queries", [])
        if not queries:
            print("Warning: No queries found in config.yaml under pubmed.queries")
            queries = []

    # Load fetched queries log path from config
    fetched_queries_log_str = config.get("logging", {}).get("fetched_queries_log", "logs/fetched_queries.jsonl")
    fetched_queries_log_path = Path(fetched_queries_log_str)
    if not fetched_queries_log_path.is_absolute():
        fetched_queries_log_path = Path(__file__).resolve().parent / fetched_queries_log_path

    # Load already fetched queries
    fetched_queries = load_fetched_queries(fetched_queries_log_path)
    
    # Filter out already fetched queries
    new_queries = [q for q in queries if q not in fetched_queries]
    skipped_queries = len(queries) - len(new_queries)
    
    if skipped_queries > 0:
        print(f"Skipping {skipped_queries} already fetched queries")
    if not new_queries:
        print("No new queries to fetch. All queries have already been fetched.")
        return
    
    today = dt.date.today()
    start_date = today - dt.timedelta(days=365 * args.years)
    mindate = start_date.strftime("%Y/%m/%d")
    maxdate = today.strftime("%Y/%m/%d")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing PMIDs from file if it exists to avoid duplicates
    seen_pmids = set()
    if output_path.exists():
        print(f"Loading existing PMIDs from {output_path}...")
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pmid = record.get("pmid")
                    if pmid:
                        seen_pmids.add(pmid)
                except json.JSONDecodeError:
                    continue
        print(f"  Found {len(seen_pmids)} existing PMIDs")
    
    num_written = 0
    num_new = 0
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    
    # Open in append mode to add new articles
    with output_path.open("a", encoding="utf-8") as fh:
        for query in new_queries:
            print(f"Fetching articles for query: {query}")
            query_count = 0
            all_ids = []
            retstart = 0
            max_fetch = args.max_articles_per_query * 3
            
            while len(all_ids) < max_fetch:
                params = {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "xml",
                    "retmax": args.esearch_batch,
                    "retstart": retstart,
                    "datetype": "pdat",
                    "mindate": mindate,
                    "maxdate": maxdate,
                }
                resp = requests.get(f"{EUTILS_BASE}esearch.fcgi", params=params, timeout=30)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                batch_ids = [elem.text for elem in root.findall(".//IdList/Id") if elem.text]
                
                if not batch_ids:
                    break
                
                all_ids.extend(batch_ids)
                retstart += len(batch_ids)
                
                if len(batch_ids) < args.esearch_batch:
                    break
                
                time.sleep(0.34)
            
            print(f"  Found {len(all_ids)} article IDs, fetching records...")
            
            for record in efetch_records(all_ids, args.efetch_batch, None):
                if query_count >= args.max_articles_per_query:
                    break
                
                pmid = record.get("pmid")
                if not pmid or pmid in seen_pmids:
                    continue
                
                abstract = record.get("abstract", "").strip()
                if not abstract:
                    continue
                
                seen_pmids.add(pmid)
                # Tag the record with the query that fetched it
                record["fetched_by_query"] = query
                fh.write(json.dumps(record) + "\n")
                num_written += 1
                num_new += 1
                query_count += 1
            
            print(f"  Wrote {query_count} new articles with abstracts")
            
            # Log that this query has been fetched
            if query_count > 0:
                log_fetched_query(fetched_queries_log_path, query, timestamp)
    
    print(f"Total articles: {len(seen_pmids)} (added {num_new} new articles)")


if __name__ == "__main__":
    main()