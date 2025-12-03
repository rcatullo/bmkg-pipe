"""
Decompose Agent - Recursive Task Decomposition

This module implements a sophisticated reasoning agent that recursively decomposes
complex tasks into simpler subtasks, executes them (potentially in parallel),
and composes the results using dynamically generated composition operators.

Key Concepts:
- Simple tasks: Can be solved with a single action call
- Complex tasks: Need to be decomposed into subtasks
- Composition operators: Python code to merge subtask results
- Control flow: Orchestrates parallel or sequential execution
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from semantic_parser.action_protocol import ActionRegistry, ActionOutput
from semantic_parser.llm_client import AzureOpenAIClient, Message
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
    VariableStack,
    TypedVariable,
    TypeCheckError,
    VariableType
)
from semantic_parser.src.decompose.composition_executor import CompositionExecutor


logger = logging.getLogger(__name__)


@dataclass
class DecomposeTrace:
    """
    Trace of the decomposition and execution process.
    
    Contains a variable_stack that tracks all typed variables passed through
    the decomposition. This replaces the generic 'context' dict.
    """
    task: str
    depth: int
    is_simple: bool
    variable_stack: Optional[VariableStack] = field(default=None)  # Stack of typed variables
    action_used: Optional[str] = field(default=None)
    subtasks: List['DecomposeTrace'] = field(default=None)
    composition_operators: List[str] = field(default=None)  # List of operator IDs used
    control_flow_graph: Optional[Dict[str, Any]] = field(default=None)  # Graph structure
    result: Any = field(default=None)
    error: Optional[str] = field(default=None)
    start_time: datetime = field(default=None)
    end_time: datetime = field(default=None)
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.composition_operators is None:
            self.composition_operators = []
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.variable_stack is None:
            self.variable_stack = VariableStack()
    
    def complete(self, result: Any, error: Optional[str] = None):
        """Mark the trace as complete"""
        self.result = result
        self.error = error
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get execution duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def push_variable(
        self, 
        name: str, 
        value: Any, 
        var_type: VariableType,
        description: str = "",
        source: Optional[str] = None
    ) -> None:
        """
        Push a variable onto the stack with type checking.
        
        Args:
            name: Variable name
            value: Variable value
            var_type: Expected type
            description: Description of the variable
            source: Which subtask/operator produced this
        """
        self.variable_stack.push(name, value, var_type, description, source)
    
    def get_variable(self, name: str) -> Optional[TypedVariable]:
        """Get a variable from the stack"""
        return self.variable_stack.get(name)
    
    def get_variable_value(self, name: str) -> Any:
        """Get just the value of a variable"""
        return self.variable_stack.get_value(name)
    
    def create_child_trace(self, task: str) -> 'DecomposeTrace':
        """
        Create a child trace that inherits the variable stack.
        
        Args:
            task: The task for the child trace
            
        Returns:
            New DecomposeTrace with child variable scope
        """
        child_stack = self.variable_stack.create_child_scope()
        return DecomposeTrace(
            task=task,
            depth=self.depth + 1,
            is_simple=False,
            variable_stack=child_stack
        )


class DecomposeAgent:
    """
    Recursive task decomposition agent.
    
    This agent recursively breaks down complex tasks into simpler subtasks,
    executes them (potentially in parallel), and composes the results.
    """
    
    def __init__(
        self,
        llm_client: AzureOpenAIClient,
        action_registry: ActionRegistry,
        max_depth: int = 5,
        max_subtasks: int = 5,
        verbose: bool = False
    ):
        """
        Initialize the decompose agent.
        
        Args:
            llm_client: LLM client for reasoning
            action_registry: Registry of available actions
            max_depth: Maximum recursion depth
            max_subtasks: Maximum number of subtasks per decomposition
            verbose: Whether to print detailed logs
        """
        self.llm_client = llm_client
        self.action_registry = action_registry
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks
        self.verbose = verbose
        self.executor = CompositionExecutor()
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    async def solve(
        self,
        task: str,
        variable_stack: Optional[VariableStack] = None,
        depth: int = 0
    ) -> DecomposeTrace:
        """
        Recursively solve a task using decomposition.
        
        Args:
            task: The task to solve
            variable_stack: Stack of typed variables (replaces context)
            depth: Current recursion depth
            
        Returns:
            DecomposeTrace with execution details and result
        """
        # Create trace with variable stack
        if variable_stack is None:
            variable_stack = VariableStack()
        
        trace = DecomposeTrace(
            task=task, 
            depth=depth, 
            is_simple=False,
            variable_stack=variable_stack
        )
        
        if self.verbose:
            indent = "  " * depth
            print(f"{indent}[Depth {depth}] Solving task: {task[:80]}...")
            if variable_stack.get_all_names():
                print(f"{indent}  Variables in scope: {variable_stack.get_all_names()}")
        
        # Check max depth
        if depth >= self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached")
            trace.complete(None, f"Max depth {self.max_depth} reached")
            return trace
        
        try:
            # Step 1: Judge task complexity
            complexity = await self._judge_complexity(task, trace)
            
            if complexity.is_simple:
                # Simple task: execute directly
                trace.is_simple = True
                result = await self._execute_simple_task(
                    task, 
                    complexity.recommended_action,
                    trace
                )
                trace.complete(result)
                
                if self.verbose:
                    indent = "  " * depth
                    print(f"{indent}✓ Simple task completed with {complexity.recommended_action}")
                
            else:
                # Complex task: decompose and solve recursively
                trace.is_simple = False
                decomposition = await self._decompose_task(task, trace)
                
                if self.verbose:
                    indent = "  " * depth
                    print(f"{indent}↓ Decomposing into {len(decomposition.subtasks)} subtasks")
                    print(f"{indent}  with {len(decomposition.composition_operators)} operators")
                
                # Store decomposition info
                trace.composition_operators = [op.operator_id for op in decomposition.composition_operators]
                trace.control_flow_graph = decomposition.control_flow.to_dict()
                
                # Solve subtasks recursively
                result = await self._solve_decomposed_task(
                    decomposition,
                    depth,
                    trace
                )
                trace.complete(result)
                
                if self.verbose:
                    indent = "  " * depth
                    print(f"{indent}✓ Complex task completed")
            
            return trace
            
        except Exception as e:
            logger.error(f"Error solving task at depth {depth}: {e}")
            trace.complete(None, str(e))
            return trace
    
    async def _judge_complexity(
        self,
        task: str,
        trace: DecomposeTrace
    ) -> TaskComplexity:
        """
        Judge if a task is simple enough to be solved with one action.
        
        Args:
            task: The task to judge
            trace: Current trace with variable stack
            
        Returns:
            TaskComplexity with judgment and recommended action
        """
        # Get available actions
        action_specs = self.action_registry.get_action_specs()
        actions_desc = "\n".join([
            f"- {spec['name']}: {spec['description']}"
            for spec in action_specs
        ])
        
        # Format variable stack for prompt
        variables_str = trace.variable_stack.format_for_prompt()
        
        prompt = f"""You are analyzing whether a task is SIMPLE or COMPLEX.

A task is SIMPLE if:
- It can be completed with exactly ONE action call
- No intermediate steps or composition needed
- The action directly produces the final answer
- The task is atomic and can not be broken down further

A task is COMPLEX if:
- It requires multiple steps or actions
- Results need to be combined or processed
- There are dependencies between subtasks

**Task:** {task}

**Available Variables:**
{variables_str}

**Available Actions:**
{actions_desc}

Analyze this task and respond with a JSON object:
{{
    "is_simple": true/false,
    "reasoning": "Brief explanation of your judgment",
    "recommended_action": "action_name (if simple)" or null,
    "recommended_parameters": {{}} (if simple) or null
}}

Return ONLY the JSON object, no additional text.
"""
        
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.chat_completion(messages)
        
        # Parse response
        try:
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            json_str = response.content[json_start:json_end]
            parsed = json.loads(json_str)
            
            return TaskComplexity(
                is_simple=parsed["is_simple"],
                reasoning=parsed["reasoning"],
                recommended_action=parsed.get("recommended_action"),
                recommended_parameters=parsed.get("recommended_parameters", {})
            )
        except Exception as e:
            logger.error(f"Failed to parse complexity judgment: {e}")
            # Default to complex if parsing fails
            return TaskComplexity(
                is_simple=False,
                reasoning=f"Failed to parse: {e}",
                recommended_action=None,
                recommended_parameters={}
            )
    
    async def _execute_simple_task(
        self,
        task: str,
        action_name: str,
        trace: DecomposeTrace,
        subtask: Optional[SubTask] = None
    ) -> Any:
        """
        Execute a simple task with the recommended action.
        
        Uses the variable stack from the trace to provide inputs to the action.
        Only passes parameters that the action explicitly accepts.
        
        Args:
            task: The task to execute
            action_name: Name of the action to use
            trace: Current trace with variable stack
            subtask: Optional subtask with spec for input/output naming
            
        Returns:
            Result from the action
        """
        action = self.action_registry.get_action(action_name)
        
        if action is None:
            raise ValueError(f"Action '{action_name}' not found")
        
        trace.action_used = action_name
        
        # Get the action's accepted parameters from its input schema
        input_schema = action.get_input_schema()
        accepted_params = set(input_schema.get("properties", {}).keys())
        # Remove 'query' since we pass it explicitly
        accepted_params.discard("query")
        
        # Prepare action kwargs from variable stack
        # Only include parameters that the action accepts
        action_kwargs = {}
        
        # If subtask has spec, use input specs to map variables
        if subtask and subtask.spec and subtask.spec.inputs:
            for input_spec in subtask.spec.inputs:
                var = trace.variable_stack.get(input_spec.name)
                if var is not None:
                    # Validate type matches what the input spec expects
                    if var.var_type != input_spec.var_type:
                        raise TypeCheckError(
                            f"Type mismatch for input '{input_spec.name}': "
                            f"expected {input_spec.var_type}, got {var.var_type}"
                        )
                    # Only pass to action if it accepts this parameter
                    if input_spec.name in accepted_params:
                        action_kwargs[input_spec.name] = var.value
        
        # Also check for any variables that match action parameters
        for param_name in accepted_params:
            if param_name not in action_kwargs:
                var = trace.variable_stack.get(param_name)
                if var is not None:
                    action_kwargs[param_name] = var.value
        
        if self.verbose:
            indent = "  " * trace.depth
            print(f"{indent}  Executing {action_name} with params: {list(action_kwargs.keys())}")
        
        # For simple tasks, we pass the task as the query parameter
        try:
            result = action.execute(query=task, **action_kwargs)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Action failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error executing simple task: {e}")
            raise
    
    async def _decompose_task(
        self,
        task: str,
        trace: DecomposeTrace
    ) -> TaskDecomposition:
        """
        Decompose a complex task into a directed graph.
        
        Nodes: Subtasks
        Edges: Composition operators that transform/merge outputs
        
        Args:
            task: The task to decompose
            trace: Current trace with variable stack
            
        Returns:
            TaskDecomposition with subtasks, operators, and control flow graph
        """
        action_specs = self.action_registry.get_action_specs()
        actions_desc = "\n".join([
            f"- {spec['name']}: {spec['description']}"
            for spec in action_specs
        ])
        
        # Format variable stack for prompt
        variables_str = trace.variable_stack.format_for_prompt()
        
        prompt = f"""You are decomposing a complex task into a directed graph of subtasks and composition operators.

**Task:** {task}

**Available Variables:**
{variables_str}

**Available Actions:**
{actions_desc}

**Decomposition Structure:**

1. **Subtasks (Nodes)**: 2-{self.max_subtasks} subtasks that solve parts of the problem
   - Each subtask should be as simple as possible (ideally solvable with one action)
   - Each subtask produces a typed output
   - **IMPORTANT**: Each subtask must have a clear input/output specification with variable names AND types

2. **Subtask Specs**: For each subtask, define:
   - **inputs**: List of inputs with name, type, description, and source
   - **output**: The output with name, type, and description
   - **Supported types**: "string", "int", "float", "dataframe"
   - Input variable names should match the output names from source subtasks or operators

3. **Composition Operators (Edges)**: Functions that connect subtasks
   - Transform or merge outputs from one or more subtasks
   - Can be: simple pass-through, data transformation, merging, filtering, etc.
   - Written as executable Python functions
   - Parameter names MUST match the output variable names from the source subtasks

4. **Control Flow (Directed Graph)**: 
   - Nodes are ONLY subtasks (plus special "start" and "end" markers)
   - Operators are NOT nodes - they are applied on edges between subtasks
   - Edge format: {{"from": "subtask_X or start", "to": "subtask_Y or end", "operator": "op_id or null"}}
   - **CRITICAL**: "from" and "to" can ONLY be: "start", "end", or subtask IDs like "subtask_1", "subtask_2", etc.
   - **NEVER use operator IDs (like "op_merge") in "from" or "to" fields** - operators go ONLY in the "operator" field
   
   WRONG: {{"from": "subtask_1", "to": "op_merge", ...}}  ← op_merge is an operator, NOT a valid node!
   RIGHT: {{"from": "subtask_1", "to": "subtask_3", "operator": "op_merge"}}  ← operator in "operator" field only

**Example Decomposition:**

Task: "Compare Q4 2024 vs Q4 2023 MOIC and rank by improvement"

Subtasks:
- subtask_1: Get Q4 2024 MOIC data
  - inputs: []
  - output: {{"name": "q4_2024_moic", "type": "dataframe", "description": "MOIC data for Q4 2024"}}
- subtask_2: Get Q4 2023 MOIC data  
  - inputs: []
  - output: {{"name": "q4_2023_moic", "type": "dataframe", "description": "MOIC data for Q4 2023"}}
- subtask_3: Rank the improvements
  - inputs: [{{"name": "merged_moic_data", "type": "dataframe", "description": "Merged MOIC data", "source_subtask": "op_merge"}}]
  - output: {{"name": "ranked_improvements", "type": "dataframe", "description": "Rankings by MOIC improvement"}}

**JSON Response Format:**

{{
    "subtasks": [
        {{
            "subtask_id": "subtask_1",
            "description": "What this subtask does",
            "spec": {{
                "inputs": [
                    {{
                        "name": "input_var_name",
                        "type": "string|int|float|dataframe",
                        "description": "What this input represents",
                        "source_subtask": "subtask_id or operator_id that provides this"
                    }}
                ],
                "output": {{
                    "name": "output_var_name",
                    "type": "string|int|float|dataframe",
                    "description": "What this output represents"
                }}
            }}
        }}
    ],
    "composition_operators": [
        {{
            "operator_id": "op_1",
            "description": "What this operator does",
            "input_subtasks": ["subtask_1", "subtask_2"],
            "output_name": "merged_data",
            "output_type": "dataframe",
            "code": "def op_1(q4_2024_moic, q4_2023_moic):\\n    # Parameter names match output names from subtasks\\n    import pandas as pd\\n    merged = pd.merge(q4_2024_moic, q4_2023_moic, on='fund_id')\\n    return merged"
        }}
    ],
    "control_flow": {{
        "edges": [
            {{"from": "start", "to": "subtask_1", "operator": null}},
            {{"from": "start", "to": "subtask_2", "operator": null}},
            {{"from": "subtask_1", "to": "subtask_3", "operator": "op_merge"}},
            {{"from": "subtask_2", "to": "subtask_3", "operator": null}},
            {{"from": "subtask_3", "to": "end", "operator": null}}
        ],
        "nodes": ["subtask_1", "subtask_2", "subtask_3"]
    }},
    "reasoning": "Subtask 1 and 2 run in parallel. op_merge combines their outputs when flowing to subtask_3."
}}

**Important Rules:**
1. Every subtask must be reachable from "start"
2. At least one subtask must reach "end"  
3. Operator parameter names MUST match the output variable names from source subtasks
4. Subtask input names should match what they will receive (from operators or other subtasks)
5. Use null for operator when just starting a subtask or passing through unchanged
6. Make operators simple and focused (one clear purpose each)
7. Every subtask MUST have a spec with inputs and output INCLUDING types
8. **CRITICAL**: Operator output_name MUST be DIFFERENT from all subtask output names
9. All output names must be unique across the entire decomposition
10. **CRITICAL - VALIDATION WILL FAIL IF VIOLATED**: In control_flow edges:
    - "from" field: ONLY "start" or subtask IDs (subtask_1, subtask_2, etc.)
    - "to" field: ONLY "end" or subtask IDs (subtask_1, subtask_2, etc.)
    - "operator" field: operator IDs go HERE (op_merge, op_filter, etc.) or null
    - NEVER put operator IDs in "from" or "to" - this will cause a validation error!
11. **CRITICAL**: All inputs and outputs MUST have a "type" field with one of: "string", "int", "float", "dataframe"
12. The "nodes" array in control_flow must contain ONLY subtask IDs, never operator IDs

Respond with ONLY the JSON object, no additional text.
"""
        
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.chat_completion(messages)
        
        # Parse response
        try:
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            json_str = response.content[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Create subtasks with specs
            subtasks = []
            for st_data in parsed["subtasks"]:
                # Parse the spec if provided
                spec = None
                if "spec" in st_data and st_data["spec"]:
                    spec = SubTaskSpec.from_dict(st_data["spec"])
                
                subtask = SubTask(
                    subtask_id=st_data["subtask_id"],
                    description=st_data["description"],
                    depends_on=[],  # No longer used, graph defines dependencies
                    spec=spec
                )
                subtasks.append(subtask)
            
            # Create composition operators
            operators = []
            for op_data in parsed["composition_operators"]:
                operator = CompositionOperator(
                    operator_id=op_data["operator_id"],
                    code=op_data["code"],
                    input_subtasks=op_data["input_subtasks"],
                    output_name=op_data["output_name"],
                    output_type=op_data.get("output_type", "dataframe"),  # Default to dataframe
                    description=op_data.get("description", "")
                )
                operators.append(operator)
            
            # Create control flow graph
            edges = []
            for edge_data in parsed["control_flow"]["edges"]:
                edge = ControlFlowEdge(
                    from_node=edge_data["from"],
                    to_node=edge_data["to"],
                    operator_id=edge_data.get("operator")
                )
                edges.append(edge)
            
            control_flow = ControlFlow(
                edges=edges,
                node_ids=parsed["control_flow"]["nodes"]
            )
            
            decomposition = TaskDecomposition(
                subtasks=subtasks,
                composition_operators=operators,
                control_flow=control_flow,
                reasoning=parsed.get("reasoning", "")
            )
            
            # Validate the decomposition
            decomposition.validate()
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Failed to parse decomposition: {e}")
            raise
    
    async def _solve_decomposed_task(
        self,
        decomposition: TaskDecomposition,
        depth: int,
        parent_trace: DecomposeTrace
    ) -> Any:
        """
        Solve a decomposed task by executing the control flow graph.
        
        Algorithm:
        1. Start with nodes that have incoming edges from "start"
        2. Execute nodes when all their dependencies are satisfied
        3. Apply composition operators on edges to transform outputs
        4. Continue until reaching "end"
        
        Results are stored in the variable stack with proper types, enabling
        type-safe connection between subtasks via composition operators.
        
        Args:
            decomposition: Task decomposition with graph structure
            depth: Current depth
            parent_trace: Parent trace with variable stack
            
        Returns:
            Final result (output of the edge leading to "end")
        """
        # Track completed nodes and their results
        completed_nodes = set()
        node_results = {}  # subtask_id -> result (raw result)
        edge_results = {}  # (from, to, operator_id) -> result after operator
        
        # Use the parent trace's variable stack as the base
        # Results will be pushed onto this stack with proper types
        var_stack = parent_trace.variable_stack
        
        if self.verbose:
            indent = "  " * depth
            print(f"{indent}Executing control flow graph:")
            print(f"{indent}  Nodes: {decomposition.control_flow.node_ids}")
            print(f"{indent}  Edges: {len(decomposition.control_flow.edges)}")
        
        # Get nodes that can start immediately (connected from "start")
        ready_nodes = set(decomposition.control_flow.get_start_nodes())
        
        while ready_nodes or len(completed_nodes) < len(decomposition.subtasks):
            if not ready_nodes:
                # Check if we can enable any new nodes
                for node_id in decomposition.control_flow.node_ids:
                    if node_id not in completed_nodes:
                        if decomposition.control_flow.can_execute(node_id, completed_nodes):
                            ready_nodes.add(node_id)
                
                if not ready_nodes:
                    # No progress can be made
                    incomplete = set(decomposition.control_flow.node_ids) - completed_nodes
                    raise Exception(f"Deadlock: Cannot execute remaining nodes {incomplete}")
            
            # Execute all ready nodes in parallel
            tasks = []
            node_list = list(ready_nodes)
            
            for node_id in node_list:
                subtask = decomposition.get_subtask_by_id(node_id)
                
                # Create a child variable stack for this subtask
                # It inherits all variables from the parent scope
                subtask_stack = var_stack.create_child_scope()
                
                # Add any inputs from incoming edges to the subtask's stack
                incoming_edges = decomposition.control_flow.get_incoming_edges(node_id)
                
                for edge in incoming_edges:
                    if edge.is_start_edge():
                        continue
                    
                    # Get the result from the edge
                    edge_key = (edge.from_node, edge.to_node, edge.operator_id)
                    
                    # Determine the result, its name, and type
                    result_value = None
                    result_name = None
                    result_type: VariableType = "string"  # default
                    
                    if edge_key in edge_results:
                        # Use the transformed result from the edge
                        result_value = edge_results[edge_key]
                        # If operator was applied, use the operator's output_name
                        if edge.operator_id:
                            operator = decomposition.get_operator_by_id(edge.operator_id)
                            if operator:
                                result_name = operator.output_name
                                # Get type from operator if available
                                result_type = getattr(operator, 'output_type', 'dataframe')
                    elif edge.from_node in node_results:
                        # Use the raw result if no operator was applied
                        result_value = node_results[edge.from_node]
                        # Get the output name and type from the source subtask's spec
                        source_subtask = decomposition.get_subtask_by_id(edge.from_node)
                        if source_subtask and source_subtask.spec and source_subtask.spec.output:
                            result_name = source_subtask.spec.output.name
                            result_type = source_subtask.spec.output.var_type
                    
                    if result_value is not None and result_name:
                        # Push the result onto the subtask's variable stack
                        # Map to the input name expected by the subtask
                        if subtask.spec and subtask.spec.inputs:
                            for input_spec in subtask.spec.inputs:
                                # Check if this input expects data from this source
                                if input_spec.source_subtask == edge.from_node or \
                                   input_spec.source_subtask == edge.operator_id or \
                                   input_spec.name == result_name:
                                    subtask_stack.push(
                                        name=input_spec.name,
                                        value=result_value,
                                        var_type=input_spec.var_type,
                                        description=input_spec.description,
                                        source=edge.from_node if not edge.operator_id else edge.operator_id,
                                        validate=True  # Enforce type checking
                                    )
                                    break
                            else:
                                # Fallback: use result_name
                                subtask_stack.push(
                                    name=result_name,
                                    value=result_value,
                                    var_type=result_type,
                                    source=edge.from_node,
                                    validate=True
                                )
                        else:
                            # No spec, use result_name
                            subtask_stack.push(
                                name=result_name,
                                value=result_value,
                                var_type=result_type,
                                source=edge.from_node,
                                validate=True
                            )
                
                # Solve the subtask with its own variable stack
                task = self.solve(subtask.description, subtask_stack, depth + 1)
                tasks.append((node_id, subtask, task))
            
            # Wait for all ready nodes to complete
            for node_id, subtask, task in tasks:
                trace = await task
                parent_trace.subtasks.append(trace)
                
                if trace.error:
                    raise Exception(f"Subtask {node_id} failed: {trace.error}")
                
                # Store the result by subtask_id
                node_results[node_id] = trace.result
                
                # Push result onto the main variable stack with proper type
                if subtask.spec and subtask.spec.output:
                    output_spec = subtask.spec.output
                    var_stack.push(
                        name=output_spec.name,
                        value=trace.result,
                        var_type=output_spec.var_type,
                        description=output_spec.description,
                        source=node_id,
                        validate=True  # Enforce type checking
                    )
                    if self.verbose:
                        indent = "  " * depth
                        print(f"{indent}  Stored result as '{output_spec.name}' ({output_spec.var_type})")
                
                completed_nodes.add(node_id)
                ready_nodes.discard(node_id)
                
                if self.verbose:
                    indent = "  " * depth
                    print(f"{indent}  ✓ Completed {node_id}")
            
            # Process outgoing edges from completed nodes
            for node_id in list(completed_nodes):
                outgoing_edges = decomposition.control_flow.get_outgoing_edges(node_id)
                
                for edge in outgoing_edges:
                    edge_key = (edge.from_node, edge.to_node, edge.operator_id)
                    
                    # Skip if already processed
                    if edge_key in edge_results:
                        continue
                    
                    # Check if all inputs for the operator are ready
                    if edge.operator_id:
                        operator = decomposition.get_operator_by_id(edge.operator_id)
                        if not operator:
                            raise Exception(f"Operator {edge.operator_id} not found")
                        
                        # Check if all input subtasks are completed
                        inputs_ready = all(
                            input_id in completed_nodes 
                            for input_id in operator.input_subtasks
                        )
                        
                        if not inputs_ready:
                            continue
                        
                        # Execute the composition operator
                        # Use output names from subtask specs as parameter names
                        operator_inputs = {}
                        for input_id in operator.input_subtasks:
                            input_subtask = decomposition.get_subtask_by_id(input_id)
                            if input_subtask and input_subtask.spec and input_subtask.spec.output:
                                # Use the output name as the parameter name
                                param_name = input_subtask.spec.output.name
                                operator_inputs[param_name] = node_results[input_id]
                            else:
                                # Fallback: use subtask_id as parameter name
                                operator_inputs[input_id] = node_results[input_id]
                        
                        result = self.executor.execute_operator(
                            operator,
                            operator_inputs
                        )
                        
                        edge_results[edge_key] = result
                        
                        # Push operator output onto variable stack with type
                        output_type = getattr(operator, 'output_type', 'dataframe')
                        var_stack.push(
                            name=operator.output_name,
                            value=result,
                            var_type=output_type,
                            description=operator.description,
                            source=operator.operator_id,
                            validate=True
                        )
                        
                        if self.verbose:
                            indent = "  " * depth
                            print(f"{indent}  → Applied {edge.operator_id}: {edge.from_node} → {edge.to_node}")
                            print(f"{indent}    Output stored as '{operator.output_name}' ({output_type})")
                    else:
                        # No operator, just pass through
                        edge_results[edge_key] = node_results[node_id]
                    
                    # If this edge leads to "end", we have the final result
                    if edge.is_end_edge():
                        final_result = edge_results[edge_key]
                        if self.verbose:
                            indent = "  " * depth
                            print(f"{indent}  ✓ Reached end node")
                        return final_result
        
        # If we completed all nodes but didn't reach end, find the end nodes
        end_nodes = decomposition.control_flow.get_end_nodes()
        if not end_nodes:
            raise Exception("No path to end node found")
        
        # Return the result from the first end node
        for edge in decomposition.control_flow.edges:
            if edge.is_end_edge() and edge.from_node in completed_nodes:
                edge_key = (edge.from_node, edge.to_node, edge.operator_id)
                if edge_key in edge_results:
                    return edge_results[edge_key]
                else:
                    return node_results[edge.from_node]
        
        raise Exception("Could not determine final result")
    
    def get_trace_summary(self, trace: DecomposeTrace) -> Dict[str, Any]:
        """
        Get a summary of the decomposition trace.
        
        Args:
            trace: Decomposition trace
            
        Returns:
            Summary dictionary
        """
        return {
            "task": trace.task,
            "depth": trace.depth,
            "is_simple": trace.is_simple,
            "action_used": trace.action_used,
            "num_subtasks": len(trace.subtasks),
            "composition_operators": trace.composition_operators,
            "control_flow": trace.control_flow_graph,
            "duration": trace.get_duration(),
            "success": trace.error is None,
            "error": trace.error,
            "result": trace.result,
            "subtasks": [self.get_trace_summary(st) for st in trace.subtasks]
        }

