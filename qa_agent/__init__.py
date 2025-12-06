"""
KG-first QA agent package.

This package implements the Anchor → Expand → Explain pattern on top of the
Talazoparib resistance knowledge graph stored in Neo4j.
"""

from .agent import KgQaAgent

__all__ = ["KgQaAgent"]


