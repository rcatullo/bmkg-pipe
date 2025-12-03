"""
Cypher Module for Semantic Parser

This module provides Cypher-based semantic parsing for graph database queries.
"""

import re
from typing import Dict, Any, List

from semantic_parser.action_protocol import (
    Action,
    ActionOutput,
    ActionRegistry,
    ModuleConfig,
)


# Module configuration
CYPHER_CONFIG = ModuleConfig(
    name="cypher",
    target_format="Cypher",
    description="Cypher-based semantic parsing for graph database queries (Neo4j)",
    predecided_actions=[],
    metadata={
        "database_type": "Neo4j",
        "domain": "graph",
    }
)


STOPWORDS = {
    "what",
    "which",
    "that",
    "have",
    "with",
    "show",
    "tell",
    "about",
    "for",
    "and",
    "the",
    "are",
    "was",
    "were",
    "does",
    "do",
    "how",
    "why",
    "who",
    "is",
    "of",
    "to",
    "in",
    "on",
    "from",
    "related",
    "relation",
    "between",
}


class GenerateCypher(Action):
    def __init__(self, max_keywords: int = 8):
        self.max_keywords = max_keywords
        super().__init__(
            name="GenerateCypher",
            description=(
                "Converts natural language questions into parameterized Cypher queries "
                "that retrieve subject–predicate–object triples from Neo4j."
            ),
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"}
            },
            "required": ["query"]
        }

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9\\-]+", query.lower())
        keywords: List[str] = []
        seen = set()
        for token in tokens:
            if len(token) < 3:
                continue
            if token in STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= self.max_keywords:
                break
        return keywords

    def execute(self, query: str) -> ActionOutput:
        keywords = self._extract_keywords(query)
        cypher = """
        MATCH (s)-[r]->(o)
        WHERE size($keywords) = 0 OR ANY(keyword IN $keywords WHERE
            keyword <> "" AND (
                toLower(coalesce(s.name, "")) CONTAINS keyword OR
                toLower(coalesce(s.text, "")) CONTAINS keyword OR
                toLower(coalesce(o.name, "")) CONTAINS keyword OR
                toLower(coalesce(o.text, "")) CONTAINS keyword
            )
        )
        RETURN
            s {id: s.id, name: s.name, text: s.text, class: head(labels(s))} AS subject,
            type(r) AS predicate,
            o {id: o.id, name: o.name, text: o.text, class: head(labels(o))} AS object,
            coalesce(r.sentences, []) AS evidence,
            coalesce(r.pmids, []) AS pmids
        LIMIT 25
        """
        return ActionOutput(
            success=True,
            result={
                "query": cypher,
                "parameters": {"keywords": keywords},
            },
            metadata={"keywords": keywords},
        )


def create_action_registry(
    # Add module-specific parameters here
    **kwargs
) -> ActionRegistry:
    """
    Create and configure an ActionRegistry with all Cypher actions.
    
    This is the main factory function for setting up the Cypher module.
    
    Returns:
        Configured ActionRegistry with all Cypher actions registered
        and module configuration set.
    
    Example:
        >>> from semantic_parser.modules.cypher import create_action_registry
        >>> registry = create_action_registry()
        >>> engine = ReACTEngine(llm_client, registry)
    """
    # Create registry with module config
    registry = ActionRegistry(module_config=CYPHER_CONFIG)
    
    # TODO: Register Cypher-specific actions here
    # registry.register(FetchNodeTypes(...))
    # registry.register(GenerateCypher(...))
    # etc.
    
    # Add GenerateCypher action to the registry
    registry.register(GenerateCypher())
    
    return registry


def get_module_config() -> ModuleConfig:
    """
    Get the Cypher module configuration.
    
    Returns:
        ModuleConfig for the Cypher module
    """
    return CYPHER_CONFIG


__all__ = [
    "create_action_registry",
    "get_module_config",
    "CYPHER_CONFIG",
]
