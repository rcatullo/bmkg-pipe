"""
Verdant Module for Semantic Parser

This module provides SQL-based semantic parsing for financial database queries.
It includes actions for fetching columns, background knowledge, generating SQL, etc.
"""

import os
from typing import Optional

from semantic_parser.action_protocol import ActionRegistry, ModuleConfig
from semantic_parser.modules.verdant.actions import (
    FetchRelatedColumns,
    FetchBackgroundKnowledge,
    SketchSQL,
    GenerateSQL,
    ExecuteSQL,
    GenerateAndExecuteSQL,
    Finish,
)
from semantic_parser.modules.verdant.db_utils import DatabaseManager
from semantic_parser.llm_client import AzureOpenAIClient


# Module configuration
VERDANT_CONFIG = ModuleConfig(
    name="verdant",
    target_format="SQL",
    description="SQL-based semantic parsing for financial database queries (Verdant)",
    predecided_actions=["FetchBackgroundKnowledge"],
    metadata={
        "database_type": "PostgreSQL",
        "domain": "financial",
    }
)

# Default visible tables for Verdant
DEFAULT_VISIBLE_TABLES = [
    "FundCashFlow",
    "FundInvestmentProperties",
    "FundInvestmentTimeProperties",
    "FundTimeProperties",
    "HandTransformedPortfolioInvestmentswithCFRaw"
]


def create_action_registry(
    openai_api_key: Optional[str] = None,
    visible_tables: Optional[list] = None,
    include_separate_generate_execute: bool = False,
    deployment_name: str = "o3",
) -> ActionRegistry:
    """
    Create and configure an ActionRegistry with all Verdant actions.
    
    This is the main factory function for setting up the Verdant module.
    It creates all necessary actions and registers them with a properly
    configured ActionRegistry.
    
    Args:
        openai_api_key: Optional OpenAI API key. If not provided, uses
                       AZURE_OPENAI_API_KEY environment variable.
        visible_tables: Optional list of table names to expose. If not
                       provided, uses DEFAULT_VISIBLE_TABLES.
        include_separate_generate_execute: If True, includes separate
                                          GenerateSQL and ExecuteSQL actions.
                                          If False (default), only includes
                                          the combined GenerateAndExecuteSQL.
        deployment_name: Azure OpenAI deployment name (default: "o3").
    
    Returns:
        Configured ActionRegistry with all Verdant actions registered
        and module configuration set.
    
    Example:
        >>> from semantic_parser.modules.verdant import create_action_registry
        >>> registry = create_action_registry()
        >>> # Use with ReACTEngine
        >>> engine = ReACTEngine(llm_client, registry)
    """
    # Get API key
    api_key = openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key must be provided either as argument or "
            "via AZURE_OPENAI_API_KEY environment variable"
        )
    
    # Initialize clients
    openai_client = AzureOpenAIClient(
        api_key=api_key,
        deployment_name=deployment_name,
    )
    
    tables = visible_tables or DEFAULT_VISIBLE_TABLES
    db_manager = DatabaseManager(visible_tables=tables)
    
    # Create registry with module config
    registry = ActionRegistry(module_config=VERDANT_CONFIG)
    
    # Register actions
    registry.register(FetchRelatedColumns(openai_client=openai_client))
    registry.register(FetchBackgroundKnowledge(openai_client=openai_client))
    registry.register(SketchSQL(openai_client=openai_client))
    
    if include_separate_generate_execute:
        registry.register(GenerateSQL(openai_client=openai_client))
        registry.register(ExecuteSQL(db_manager=db_manager))
    
    registry.register(GenerateAndExecuteSQL(
        openai_client=openai_client,
        db_manager=db_manager
    ))
    registry.register(Finish(openai_client=openai_client))
    
    return registry


def get_module_config() -> ModuleConfig:
    """
    Get the Verdant module configuration.
    
    Returns:
        ModuleConfig for the Verdant module
    """
    return VERDANT_CONFIG


# Export key components
__all__ = [
    # Factory function
    "create_action_registry",
    "get_module_config",
    
    # Configuration
    "VERDANT_CONFIG",
    "DEFAULT_VISIBLE_TABLES",
    
    # Actions (for advanced usage)
    "FetchRelatedColumns",
    "FetchBackgroundKnowledge",
    "SketchSQL",
    "GenerateSQL",
    "ExecuteSQL",
    "GenerateAndExecuteSQL",
    "Finish",
    
    # Utilities
    "DatabaseManager",
    "AzureOpenAIClient",
]

