"""
Decompose Utilities

Data structures and utilities for the decompose agent system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, Literal
from enum import Enum
import pandas as pd
import copy

if TYPE_CHECKING:
    from semantic_parser.src.decompose.decompose_agent import DecomposeTrace


# Supported variable types
VariableType = Literal["string", "int", "float", "dataframe"]

# Type mapping for validation
TYPE_MAP = {
    "string": str,
    "int": int,
    "float": (int, float),  # int is also acceptable for float
    "dataframe": pd.DataFrame,
}


class TypeCheckError(Exception):
    """Exception raised when type checking fails"""
    pass


@dataclass
class TypedVariable:
    """
    A variable with a name, type, value, and description.
    """
    name: str
    var_type: VariableType
    value: Any
    description: str = ""
    source: Optional[str] = None  # Which subtask/operator produced this
    
    def validate_type(self) -> bool:
        """
        Validate that the value matches the declared type.
        
        Returns:
            True if valid
            
        Raises:
            TypeCheckError if type doesn't match
        """
        expected_type = TYPE_MAP.get(self.var_type)
        if expected_type is None:
            raise TypeCheckError(f"Unknown type '{self.var_type}' for variable '{self.name}'")
        
        if not isinstance(self.value, expected_type):
            actual_type = type(self.value).__name__
            raise TypeCheckError(
                f"Type mismatch for variable '{self.name}': "
                f"expected {self.var_type}, got {actual_type}"
            )
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without value for serialization)"""
        return {
            "name": self.name,
            "type": self.var_type,
            "description": self.description,
            "source": self.source
        }


class VariableStack:
    """
    A stack of typed variables that tracks inputs/outputs through decomposition.
    
    This replaces the generic 'context' dict with a type-safe variable store.
    """
    
    def __init__(self, parent: Optional['VariableStack'] = None):
        """
        Initialize the variable stack.
        
        Args:
            parent: Optional parent stack (for nested scopes)
        """
        self._variables: Dict[str, TypedVariable] = {}
        self._parent = parent
    
    def push(
        self, 
        name: str, 
        value: Any, 
        var_type: VariableType,
        description: str = "",
        source: Optional[str] = None,
        validate: bool = True
    ) -> None:
        """
        Push a variable onto the stack.
        
        Args:
            name: Variable name
            value: Variable value
            var_type: Expected type
            description: Description of the variable
            source: Which subtask/operator produced this
            validate: Whether to validate type on push
            
        Raises:
            TypeCheckError if validation fails
        """
        var = TypedVariable(
            name=name,
            var_type=var_type,
            value=value,
            description=description,
            source=source
        )
        
        if validate:
            var.validate_type()
        
        self._variables[name] = var
    
    def get(self, name: str) -> Optional[TypedVariable]:
        """
        Get a variable by name.
        
        Searches current scope first, then parent scopes.
        
        Args:
            name: Variable name
            
        Returns:
            TypedVariable or None if not found
        """
        if name in self._variables:
            return self._variables[name]
        if self._parent:
            return self._parent.get(name)
        return None
    
    def get_value(self, name: str) -> Any:
        """
        Get just the value of a variable.
        
        Args:
            name: Variable name
            
        Returns:
            The value or None if not found
        """
        var = self.get(name)
        return var.value if var else None
    
    def has(self, name: str) -> bool:
        """Check if a variable exists"""
        return self.get(name) is not None
    
    def get_all_names(self) -> List[str]:
        """Get all variable names in scope (including parent)"""
        names = set(self._variables.keys())
        if self._parent:
            names.update(self._parent.get_all_names())
        return list(names)
    
    def get_all_variables(self) -> Dict[str, TypedVariable]:
        """Get all variables in scope (including parent)"""
        result = {}
        if self._parent:
            result.update(self._parent.get_all_variables())
        result.update(self._variables)
        return result
    
    def create_child_scope(self) -> 'VariableStack':
        """Create a child scope that inherits from this stack"""
        return VariableStack(parent=self)
    
    def copy(self) -> 'VariableStack':
        """Create a shallow copy of this stack (same parent, copied variables)"""
        new_stack = VariableStack(parent=self._parent)
        new_stack._variables = copy.copy(self._variables)
        return new_stack
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            name: {
                "type": var.var_type,
                "description": var.description,
                "source": var.source,
                "value_preview": str(var.value)[:100] if var.value is not None else None
            }
            for name, var in self.get_all_variables().items()
        }
    
    def format_for_prompt(self) -> str:
        """Format the variable stack for inclusion in LLM prompts"""
        if not self._variables and not self._parent:
            return "No variables available."
        
        lines = []
        for name, var in self.get_all_variables().items():
            type_str = var.var_type
            desc = var.description or "No description"
            source = f" (from {var.source})" if var.source else ""
            
            # Create a preview of the value
            if var.value is None:
                preview = "None"
            elif isinstance(var.value, pd.DataFrame):
                preview = f"DataFrame with {len(var.value)} rows, columns: {list(var.value.columns)[:5]}"
            elif isinstance(var.value, str):
                preview = f'"{var.value[:50]}..."' if len(var.value) > 50 else f'"{var.value}"'
            else:
                preview = str(var.value)[:50]
            
            lines.append(f"- {name} ({type_str}): {desc}{source}")
            lines.append(f"  Value: {preview}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"VariableStack({list(self._variables.keys())})"


@dataclass
class TaskComplexity:
    """
    Result of task complexity judgment.
    """
    is_simple: bool
    reasoning: str
    recommended_action: Optional[str] = None
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputSpec:
    """
    Specification for a subtask input.
    """
    name: str  # Variable name for this input
    var_type: VariableType  # Type of the variable
    description: str  # What this input represents
    source_subtask: Optional[str] = None  # Which subtask/operator provides this (None if from context)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.var_type,
            "description": self.description,
            "source_subtask": self.source_subtask
        }


@dataclass
class OutputSpec:
    """
    Specification for a subtask output.
    """
    name: str  # Variable name for this output
    var_type: VariableType  # Type of the variable
    description: str  # What this output represents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.var_type,
            "description": self.description
        }


@dataclass
class SubTaskSpec:
    """
    Complete specification for a subtask's inputs and outputs.
    
    This enables proper connection between subtasks and composition operators
    by clearly defining variable names and their meanings.
    """
    inputs: List[InputSpec] = field(default_factory=list)
    output: Optional[OutputSpec] = None
    
    def get_input_names(self) -> List[str]:
        """Get list of input variable names"""
        return [inp.name for inp in self.inputs]
    
    def get_output_name(self) -> Optional[str]:
        """Get the output variable name"""
        return self.output.name if self.output else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "inputs": [inp.to_dict() for inp in self.inputs],
            "output": self.output.to_dict() if self.output else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubTaskSpec':
        """Create from dictionary"""
        inputs = [
            InputSpec(
                name=inp["name"],
                var_type=inp.get("type", "string"),  # Default to string if not specified
                description=inp.get("description", ""),
                source_subtask=inp.get("source_subtask")
            )
            for inp in data.get("inputs", [])
        ]
        output = None
        if data.get("output"):
            output = OutputSpec(
                name=data["output"]["name"],
                var_type=data["output"].get("type", "string"),  # Default to string
                description=data["output"].get("description", "")
            )
        return cls(inputs=inputs, output=output)


@dataclass
class SubTask:
    """
    A subtask in the decomposition.
    """
    subtask_id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    spec: Optional[SubTaskSpec] = None  # Input/output specification
    
    def is_independent(self) -> bool:
        """Check if this subtask has no dependencies"""
        return len(self.depends_on) == 0
    
    def get_input_names(self) -> List[str]:
        """Get list of input variable names from spec"""
        if self.spec:
            return self.spec.get_input_names()
        return []
    
    def get_output_name(self) -> Optional[str]:
        """Get the output variable name from spec"""
        if self.spec:
            return self.spec.get_output_name()
        return None


@dataclass
class CompositionOperator:
    """
    A composition operator that transforms/merges outputs.
    Represents an edge in the control flow graph.
    """
    operator_id: str
    code: str
    input_subtasks: List[str]  # IDs of subtasks whose outputs are inputs
    output_name: str  # Name for the output of this operator
    output_type: VariableType = "dataframe"  # Type of the output
    description: str = ""
    
    def get_signature(self) -> str:
        """Get the function signature"""
        return f"def {self.operator_id}({', '.join(self.input_subtasks)}) -> Any"


@dataclass
class ControlFlowEdge:
    """
    An edge in the control flow graph.
    """
    from_node: str  # Source subtask ID or "start"
    to_node: str    # Target subtask ID or "end"
    operator_id: Optional[str] = None  # Composition operator to apply (None = direct pass)
    
    def is_start_edge(self) -> bool:
        """Check if this edge starts from the beginning"""
        return self.from_node == "start"
    
    def is_end_edge(self) -> bool:
        """Check if this edge leads to the end"""
        return self.to_node == "end"


@dataclass
class ControlFlow:
    """
    Control flow as a directed graph.
    Nodes: subtasks (plus "start" and "end")
    Edges: composition operators
    """
    edges: List[ControlFlowEdge]
    node_ids: List[str]  # All subtask IDs
    
    def get_outgoing_edges(self, node_id: str) -> List[ControlFlowEdge]:
        """Get all edges leaving a node"""
        return [e for e in self.edges if e.from_node == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[ControlFlowEdge]:
        """Get all edges entering a node"""
        return [e for e in self.edges if e.to_node == node_id]
    
    def get_start_nodes(self) -> List[str]:
        """Get nodes that can start (no incoming edges or incoming from 'start')"""
        start_nodes = []
        for node_id in self.node_ids:
            incoming = self.get_incoming_edges(node_id)
            if not incoming or all(e.is_start_edge() for e in incoming):
                start_nodes.append(node_id)
        return start_nodes
    
    def get_end_nodes(self) -> List[str]:
        """Get nodes that output to end"""
        return [e.from_node for e in self.edges if e.is_end_edge()]
    
    def can_execute(self, node_id: str, completed_nodes: set) -> bool:
        """
        Check if a node can be executed given completed nodes.
        A node can execute if all its incoming edges have their source nodes completed.
        """
        incoming = self.get_incoming_edges(node_id)
        if not incoming:
            return True
        
        for edge in incoming:
            if edge.is_start_edge():
                continue
            if edge.from_node not in completed_nodes:
                return False
        return True
    
    def topological_sort(self) -> List[str]:
        """
        Get a topological ordering of nodes.
        Returns list of node IDs in execution order.
        """
        in_degree = {node: 0 for node in self.node_ids}
        
        # Calculate in-degrees
        for edge in self.edges:
            if not edge.is_start_edge() and edge.to_node in in_degree:
                in_degree[edge.to_node] += 1
        
        # Start with nodes that have no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Reduce in-degree for neighbors
            for edge in self.get_outgoing_edges(node):
                if edge.to_node in in_degree:
                    in_degree[edge.to_node] -= 1
                    if in_degree[edge.to_node] == 0:
                        queue.append(edge.to_node)
        
        if len(result) != len(self.node_ids):
            raise ValueError("Circular dependency detected in control flow")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "edges": [
                {
                    "from": e.from_node,
                    "to": e.to_node,
                    "operator": e.operator_id
                }
                for e in self.edges
            ],
            "nodes": self.node_ids
        }


@dataclass
class TaskDecomposition:
    """
    Complete decomposition of a task.
    """
    subtasks: List[SubTask]
    composition_operators: List[CompositionOperator]  # Multiple operators
    control_flow: ControlFlow
    reasoning: str = ""
    
    def get_num_subtasks(self) -> int:
        """Get number of subtasks"""
        return len(self.subtasks)
    
    def get_operator_by_id(self, operator_id: str) -> Optional[CompositionOperator]:
        """Get composition operator by ID"""
        for op in self.composition_operators:
            if op.operator_id == operator_id:
                return op
        return None
    
    def get_subtask_by_id(self, subtask_id: str) -> Optional[SubTask]:
        """Get subtask by ID"""
        for st in self.subtasks:
            if st.subtask_id == subtask_id:
                return st
        return None
    
    def validate(self) -> bool:
        """
        Validate the decomposition.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check all subtask IDs are unique
        subtask_ids = [st.subtask_id for st in self.subtasks]
        if len(subtask_ids) != len(set(subtask_ids)):
            raise ValueError("Subtask IDs must be unique")
        
        # Check all operator IDs are unique
        operator_ids = [op.operator_id for op in self.composition_operators]
        if len(operator_ids) != len(set(operator_ids)):
            raise ValueError("Operator IDs must be unique")
        
        # Check control flow references valid nodes and operators
        for edge in self.control_flow.edges:
            # Check from_node
            if edge.from_node not in subtask_ids and edge.from_node != "start":
                raise ValueError(f"Invalid from_node in edge: {edge.from_node}")
            
            # Check to_node
            if edge.to_node not in subtask_ids and edge.to_node != "end":
                raise ValueError(f"Invalid to_node in edge: {edge.to_node}")
            
            # Check operator_id
            if edge.operator_id:
                if not self.get_operator_by_id(edge.operator_id):
                    raise ValueError(f"Invalid operator_id in edge: {edge.operator_id}")
        
        # Check control flow nodes match subtasks
        if set(self.control_flow.node_ids) != set(subtask_ids):
            raise ValueError("Control flow nodes must match subtask IDs")
        
        # Validate subtask specs (warning only, not error)
        output_names = set()
        for subtask in self.subtasks:
            if subtask.spec:
                # Check output name uniqueness
                if subtask.spec.output:
                    if subtask.spec.output.name in output_names:
                        raise ValueError(
                            f"Duplicate output name '{subtask.spec.output.name}' "
                            f"in subtask {subtask.subtask_id}"
                        )
                    output_names.add(subtask.spec.output.name)
        
        # Validate operator output names don't conflict with subtask outputs
        subtask_output_names = output_names.copy()
        for operator in self.composition_operators:
            if operator.output_name in subtask_output_names:
                # Find which subtask has the conflicting name
                conflicting_subtask = None
                for st in self.subtasks:
                    if st.spec and st.spec.output and st.spec.output.name == operator.output_name:
                        conflicting_subtask = st.subtask_id
                        break
                raise ValueError(
                    f"Operator '{operator.operator_id}' output name '{operator.output_name}' "
                    f"conflicts with subtask output name"
                    f"{f' (from {conflicting_subtask})' if conflicting_subtask else ''}. "
                    f"Operator output names must be unique and different from all subtask output names."
                )
            if operator.output_name in output_names:
                raise ValueError(
                    f"Duplicate operator output name '{operator.output_name}' "
                    f"in operator '{operator.operator_id}'"
                )
            output_names.add(operator.output_name)
        
        return True


@dataclass
class DecompositionResult:
    """
    Result of executing a decomposed task.
    """
    final_result: Any
    subtask_results: Dict[str, Any]
    composition_code: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecompositionError(Exception):
    """Exception raised during task decomposition"""
    pass


class CompositionError(Exception):
    """Exception raised during result composition"""
    pass


def format_trace_tree(trace: 'DecomposeTrace', indent: int = 0, show_specs: bool = False) -> str:
    """
    Format a decomposition trace as a tree structure.
    
    Args:
        trace: Decomposition trace
        indent: Current indentation level
        show_specs: Whether to show input/output specs
        
    Returns:
        Formatted tree string
    """
    lines = []
    prefix = "  " * indent
    
    # Current task
    task_type = "SIMPLE" if trace.is_simple else "COMPLEX"
    task_preview = trace.task[:60] + "..." if len(trace.task) > 60 else trace.task
    lines.append(f"{prefix}[{task_type}] {task_preview}")
    
    # Action or subtasks
    if trace.is_simple and trace.action_used:
        lines.append(f"{prefix}  → Action: {trace.action_used}")
    elif trace.subtasks:
        lines.append(f"{prefix}  → {len(trace.subtasks)} subtasks")
        if trace.composition_operators:
            lines.append(f"{prefix}  → {len(trace.composition_operators)} operators: {', '.join(trace.composition_operators)}")
        if trace.control_flow_graph:
            lines.append(f"{prefix}  → Graph: {len(trace.control_flow_graph.get('nodes', []))} nodes, {len(trace.control_flow_graph.get('edges', []))} edges")
        for subtask_trace in trace.subtasks:
            lines.append(format_trace_tree(subtask_trace, indent + 2, show_specs))
    
    # Result status
    if trace.error:
        lines.append(f"{prefix}  ✗ Error: {trace.error}")
    elif trace.result is not None:
        result_preview = str(trace.result)[:40]
        lines.append(f"{prefix}  ✓ Result: {result_preview}...")
    
    return "\n".join(lines)


def format_subtask_specs(subtask: SubTask) -> str:
    """
    Format a subtask's input/output specs.
    
    Args:
        subtask: SubTask with spec
        
    Returns:
        Formatted string showing inputs and outputs
    """
    lines = [f"Subtask: {subtask.subtask_id}"]
    lines.append(f"  Description: {subtask.description}")
    
    if subtask.spec:
        if subtask.spec.inputs:
            lines.append("  Inputs:")
            for inp in subtask.spec.inputs:
                source = f" (from {inp.source_subtask})" if inp.source_subtask else ""
                lines.append(f"    - {inp.name}: {inp.description}{source}")
        else:
            lines.append("  Inputs: (none)")
        
        if subtask.spec.output:
            lines.append(f"  Output: {subtask.spec.output.name}")
            lines.append(f"    {subtask.spec.output.description}")
        else:
            lines.append("  Output: (not specified)")
    else:
        lines.append("  Spec: (not defined)")
    
    return "\n".join(lines)


def format_decomposition_specs(decomposition: 'TaskDecomposition') -> str:
    """
    Format all subtask and operator specs in a decomposition.
    
    Args:
        decomposition: TaskDecomposition to format
        
    Returns:
        Formatted string showing all specs
    """
    lines = ["=== Decomposition Specs ===", ""]
    
    lines.append("--- Subtasks ---")
    for subtask in decomposition.subtasks:
        lines.append(format_subtask_specs(subtask))
        lines.append("")
    
    lines.append("--- Composition Operators ---")
    for op in decomposition.composition_operators:
        lines.append(f"Operator: {op.operator_id}")
        lines.append(f"  Description: {op.description}")
        lines.append(f"  Inputs: {', '.join(op.input_subtasks)}")
        lines.append(f"  Output: {op.output_name}")
        lines.append(f"  Code:\n{op.code}")
        lines.append("")
    
    return "\n".join(lines)


def visualize_control_flow(control_flow: ControlFlow) -> str:
    """
    Create a visual representation of the control flow graph.
    
    Args:
        control_flow: ControlFlow object
        
    Returns:
        String visualization of the graph
    """
    lines = ["Control Flow Graph:", ""]
    
    # Show nodes
    lines.append(f"Nodes ({len(control_flow.node_ids)}):")
    for node_id in control_flow.node_ids:
        incoming = len(control_flow.get_incoming_edges(node_id))
        outgoing = len(control_flow.get_outgoing_edges(node_id))
        lines.append(f"  {node_id}: {incoming} in, {outgoing} out")
    
    lines.append("")
    
    # Show edges
    lines.append(f"Edges ({len(control_flow.edges)}):")
    for edge in control_flow.edges:
        operator_str = f" via {edge.operator_id}" if edge.operator_id else ""
        lines.append(f"  {edge.from_node} → {edge.to_node}{operator_str}")
    
    lines.append("")
    
    # Show execution order
    try:
        order = control_flow.topological_sort()
        lines.append(f"Execution Order: {' → '.join(order)}")
    except ValueError as e:
        lines.append(f"Execution Order: Error - {e}")
    
    return "\n".join(lines)


def analyze_decomposition_stats(trace: 'DecomposeTrace') -> Dict[str, Any]:
    """
    Analyze statistics from a decomposition trace.
    
    Args:
        trace: Decomposition trace
        
    Returns:
        Dictionary with statistics
    """
    def collect_stats(t: 'DecomposeTrace', stats: Dict[str, Any]):
        stats['total_tasks'] += 1
        if t.is_simple:
            stats['simple_tasks'] += 1
            if t.action_used:
                stats['actions_used'][t.action_used] = stats['actions_used'].get(t.action_used, 0) + 1
        else:
            stats['complex_tasks'] += 1
            if t.composition_operators:
                for op_id in t.composition_operators:
                    stats['operators_used'][op_id] = stats['operators_used'].get(op_id, 0) + 1
        
        stats['max_depth'] = max(stats['max_depth'], t.depth)
        
        if t.error:
            stats['failed_tasks'] += 1
        else:
            stats['successful_tasks'] += 1
        
        stats['total_duration'] += t.get_duration()
        
        for subtask in t.subtasks:
            collect_stats(subtask, stats)
    
    stats = {
        'total_tasks': 0,
        'simple_tasks': 0,
        'complex_tasks': 0,
        'successful_tasks': 0,
        'failed_tasks': 0,
        'max_depth': 0,
        'total_duration': 0.0,
        'actions_used': {},
        'operators_used': {}
    }
    
    collect_stats(trace, stats)
    
    return stats

