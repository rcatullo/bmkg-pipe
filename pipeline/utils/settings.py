from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

PIPELINE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PIPELINE_DIR / "config.yaml"


@dataclass
class Settings:
    llm_model: str
    paths: "PathSettings"
    relation_extraction: "RequestSettings"
    named_entity_extraction: "RequestSettings"
    prompt_version: str = "v1"
    model_version: str = "v1"
    logging_level: str = "INFO"
    threshold: float = 0.55


@dataclass
class PathSettings:
    input: Path
    output: Path
    log: Path


@dataclass
class RequestSettings:
    request_url: str
    api_key: str
    requests_file: Path
    results_file: Path
    max_requests_per_minute: float
    max_tokens_per_minute: float
    token_encoding_name: str
    max_attempts: int
    logging_level: int


_SETTINGS: Settings | None = None


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PIPELINE_DIR / path
    return path


def _request_settings(
    raw: Dict[str, Any],
    default_requests: Path,
    default_results: Path,
) -> RequestSettings:
    request_url = raw.get("request_url", "https://api.openai.com/v1/chat/completions")
    api_key = raw.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    return RequestSettings(
        request_url=request_url,
        api_key=api_key,
        requests_file=_resolve_path(raw.get("requests_file"), default_requests),
        results_file=_resolve_path(raw.get("results_file"), default_results),
        max_requests_per_minute=float(raw.get("max_requests_per_minute", 1500.0)),
        max_tokens_per_minute=float(raw.get("max_tokens_per_minute", 125000.0)),
        token_encoding_name=raw.get("token_encoding_name", "cl100k_base"),
        max_attempts=int(raw.get("max_attempts", 5)),
        logging_level=int(raw.get("logging_level", 20)),
    )


def load_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        raw: Dict[str, Any] = {}
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        llm_cfg = raw.get("llm", {})
        paths_cfg = raw.get("paths", {})
        logging_cfg = raw.get("logging", {})
        relation_cfg = raw.get("relation_extraction", {})
        ner_cfg = raw.get("named_entity_extraction", {})

        default_input = PIPELINE_DIR / "data" / "pubmed_talazoparib.jsonl"
        default_output = PIPELINE_DIR / "data" / "relations.jsonl"
        default_log = PIPELINE_DIR / "logs" / "relation_log.jsonl"

        _SETTINGS = Settings(
            llm_model=llm_cfg.get("model", "gpt-4o-mini"),
            paths=PathSettings(
                input=_resolve_path(paths_cfg.get("input"), default_input),
                output=_resolve_path(paths_cfg.get("output"), default_output),
                log=_resolve_path(paths_cfg.get("log"), default_log),
            ),
            relation_extraction=_request_settings(
                relation_cfg,
                default_requests=PIPELINE_DIR / "relation_extraction" / "tmp" / "requests.jsonl",
                default_results=PIPELINE_DIR / "relation_extraction" / "tmp" / "results.jsonl",
            ),
            named_entity_extraction=_request_settings(
                ner_cfg,
                default_requests=PIPELINE_DIR / "named_entity_extraction" / "tmp" / "requests.jsonl",
                default_results=PIPELINE_DIR / "named_entity_extraction" / "tmp" / "results.jsonl",
            ),
            prompt_version=str(raw.get("prompt_version", "v1")),
            model_version=str(raw.get("model_version", "v1")),
            logging_level=str(logging_cfg.get("level", "INFO")),
            threshold=float(raw.get("threshold", 0.55)),
        )
    return _SETTINGS


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#"):
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

