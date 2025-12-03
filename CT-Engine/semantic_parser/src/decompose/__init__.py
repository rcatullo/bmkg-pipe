"""
Decompose Agent Package

This package implements recursive task decomposition for breaking down
complex tasks into simpler subtasks.
"""

from semantic_parser.src.decompose.decompose_agent import DecomposeAgent, DecomposeTrace
from semantic_parser.src.decompose.decompose_utils import (
    TaskComplexity,
    TaskDecomposition,
    SubTask,
    SubTaskSpec,
    InputSpec,
    OutputSpec,
    CompositionOperator,
    ControlFlow,
    ControlFlowEdge,
    DecompositionResult,
    DecompositionError,
    CompositionError,
    VariableStack,
    TypedVariable,
    TypeCheckError,
    VariableType,
    format_trace_tree,
    visualize_control_flow,
    analyze_decomposition_stats,
)
from semantic_parser.src.decompose.composition_executor import (
    CompositionExecutor,
    SafeCompositionExecutor,
    create_simple_composition,
)

__all__ = [
    # Agent
    "DecomposeAgent",
    "DecomposeTrace",
    
    # Data structures
    "TaskComplexity",
    "TaskDecomposition",
    "SubTask",
    "SubTaskSpec",
    "InputSpec",
    "OutputSpec",
    "CompositionOperator",
    "ControlFlow",
    "ControlFlowEdge",
    "DecompositionResult",
    
    # Variable Stack
    "VariableStack",
    "TypedVariable",
    "VariableType",
    
    # Exceptions
    "DecompositionError",
    "CompositionError",
    "TypeCheckError",
    
    # Utilities
    "format_trace_tree",
    "visualize_control_flow",
    "analyze_decomposition_stats",
    
    # Executor
    "CompositionExecutor",
    "SafeCompositionExecutor",
    "create_simple_composition",
]

