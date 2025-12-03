"""
Semantic Parser Framework

A unified framework for converting natural language queries into formal
logical representations (SQL, SPARQL, Cypher, etc.) using LLM-powered
ReACT-style reasoning and recursive task decomposition.
"""

# Core shared components
from semantic_parser.action_protocol import Action, ActionOutput, ActionRegistry, ModuleConfig
from semantic_parser.llm_client import AzureOpenAIClient, Message, LLMResponse

# ReACT components (from src/react/)
from semantic_parser.src.react import (
    ReACTEngine,
    ReasoningStep,
    ReasoningTrace,
    Thought,
    ActionCall,
    Observation,
    PromptBuilder,
)

# Decompose components (from src/decompose/)
from semantic_parser.src.decompose import (
    DecomposeAgent,
    DecomposeTrace,
    TaskComplexity,
    TaskDecomposition,
    SubTask,
    CompositionOperator,
    ControlFlow,
    ControlFlowEdge,
    DecompositionResult,
    DecompositionError,
    CompositionError,
    CompositionExecutor,
    SafeCompositionExecutor,
    format_trace_tree,
    visualize_control_flow,
    analyze_decomposition_stats,
    create_simple_composition,
)

# Domain modules
from semantic_parser.modules.verdant import *

__version__ = "0.1.0"

__all__ = [
    # Core shared components
    "ActionRegistry",
    "ModuleConfig",
    "Action",
    "ActionOutput",
    "AzureOpenAIClient",
    "Message",
    "LLMResponse",
    
    # ReACT components
    "ReACTEngine",
    "PromptBuilder",
    "ReasoningStep",
    "ReasoningTrace",
    "Thought",
    "ActionCall",
    "Observation",
    
    # Decompose components
    "DecomposeAgent",
    "DecomposeTrace",
    "TaskComplexity",
    "TaskDecomposition",
    "SubTask",
    "CompositionOperator",
    "ControlFlow",
    "ControlFlowEdge",
    "DecompositionResult",
    "DecompositionError",
    "CompositionError",
    "CompositionExecutor",
    "SafeCompositionExecutor",
    "format_trace_tree",
    "visualize_control_flow",
    "analyze_decomposition_stats",
    "create_simple_composition",
]
