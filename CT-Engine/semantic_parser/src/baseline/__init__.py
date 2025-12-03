"""
Baseline Module for Direct LLM Prompting

This module provides a baseline implementation that directly prompts 
the LLM with database content from xlsx files to answer questions.
No intermediate reasoning steps - just direct prompting.
"""

from semantic_parser.src.baseline.baseline_engine import (
    BaselineEngine,
    BaselineResult,
    SmartExcelLoader,
)

__all__ = [
    "BaselineEngine",
    "BaselineResult",
    "SmartExcelLoader",
]

