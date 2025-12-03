"""
Spider2V Module for Semantic Parser

This module provides SQL-based semantic parsing for the Spider2V benchmark
(visual/multimodal variant).
"""

from typing import Optional, List

from semantic_parser.action_protocol import ActionRegistry, ModuleConfig


# Module configuration
SPIDER2V_CONFIG = ModuleConfig(
    name="spider2v",
    target_format="SQL",
    description="SQL-based semantic parsing for Spider2V benchmark (visual/multimodal)",
    predecided_actions=[],
    metadata={
        "database_type": "SQLite/PostgreSQL",
        "domain": "general",
        "benchmark": "Spider2V",
        "multimodal": True,
    }
)


def create_action_registry(
    # Add module-specific parameters here
    **kwargs
) -> ActionRegistry:
    """
    Create and configure an ActionRegistry with all Spider2V actions.
    
    This is the main factory function for setting up the Spider2V module.
    
    Returns:
        Configured ActionRegistry with all Spider2V actions registered
        and module configuration set.
    
    Example:
        >>> from semantic_parser.modules.spider2v import create_action_registry
        >>> registry = create_action_registry()
        >>> engine = ReACTEngine(llm_client, registry)
    """
    # Create registry with module config
    registry = ActionRegistry(module_config=SPIDER2V_CONFIG)
    
    # TODO: Register Spider2V-specific actions here
    # registry.register(FetchSchema(...))
    # registry.register(ProcessImage(...))
    # registry.register(GenerateSQL(...))
    # etc.
    
    return registry


def get_module_config() -> ModuleConfig:
    """
    Get the Spider2V module configuration.
    
    Returns:
        ModuleConfig for the Spider2V module
    """
    return SPIDER2V_CONFIG


__all__ = [
    "create_action_registry",
    "get_module_config",
    "SPIDER2V_CONFIG",
]
