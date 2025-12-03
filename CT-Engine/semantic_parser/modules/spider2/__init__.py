"""
Spider2 Module for Semantic Parser

This module provides SQL-based semantic parsing for the Spider2 benchmark.
"""

from typing import Optional, List

from semantic_parser.action_protocol import ActionRegistry, ModuleConfig


# Module configuration
SPIDER2_CONFIG = ModuleConfig(
    name="spider2",
    target_format="SQL",
    description="SQL-based semantic parsing for Spider2 benchmark",
    predecided_actions=[],
    metadata={
        "database_type": "SQLite/PostgreSQL",
        "domain": "general",
        "benchmark": "Spider2",
    }
)


def create_action_registry(
    # Add module-specific parameters here
    **kwargs
) -> ActionRegistry:
    """
    Create and configure an ActionRegistry with all Spider2 actions.
    
    This is the main factory function for setting up the Spider2 module.
    
    Returns:
        Configured ActionRegistry with all Spider2 actions registered
        and module configuration set.
    
    Example:
        >>> from semantic_parser.modules.spider2 import create_action_registry
        >>> registry = create_action_registry()
        >>> engine = ReACTEngine(llm_client, registry)
    """
    # Create registry with module config
    registry = ActionRegistry(module_config=SPIDER2_CONFIG)
    
    # TODO: Register Spider2-specific actions here
    # registry.register(FetchSchema(...))
    # registry.register(GenerateSQL(...))
    # etc.
    
    return registry


def get_module_config() -> ModuleConfig:
    """
    Get the Spider2 module configuration.
    
    Returns:
        ModuleConfig for the Spider2 module
    """
    return SPIDER2_CONFIG


__all__ = [
    "create_action_registry",
    "get_module_config",
    "SPIDER2_CONFIG",
]
