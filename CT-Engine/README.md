# Decompose Agent - Recursive Task Decomposition

A sophisticated reasoning agent that recursively decomposes complex tasks into simpler subtasks, executes them (potentially in parallel), and composes the results using dynamically generated composition operators.

---

## 🎯 Key Concepts

### Simple vs Complex Tasks

**Simple Task:**
- Can be solved with exactly ONE action call
- No intermediate steps needed
- The action directly produces the final answer
- Example: "What is the average MOIC for all funds?"

**Complex Task:**
- Requires multiple steps or actions
- Results need to be combined or processed
- May have dependencies between subtasks
- Example: "Compare MOIC performance and rank funds by improvement"

### Task Decomposition

For complex tasks, the agent:
1. **Decomposes** the task into 2-5 subtasks
2. **Generates** a composition operator (Python code) to merge results
3. **Determines** control flow (parallel or sequential)
4. **Recursively solves** each subtask
5. **Composes** the final result

### Control Flow

**Parallel:**
- Subtasks are independent
- Can run concurrently
- Faster execution through parallelization
- Example: "Calculate average MOIC AND identify top funds"

**Sequential:**
- Subtasks have dependencies
- One subtask needs another's output
- Must run in order
- Example: "Find funds with MOIC > 2.0, THEN rank them by cash flow"

### Composition Operators

Dynamically generated Python code that merges subtask results:

```python
def compose(results):
    """
    Merge subtask results into final result.
    
    Args:
        results: Dict mapping subtask_id to result
                 e.g., {"subtask_1": result1, "subtask_2": result2}
    
    Returns:
        Final composed result
    """
    # Generated code here
    # Examples:
    # - Merge dictionaries
    # - Concatenate lists
    # - Combine and rank
    # - Filter and aggregate
    pass
```

---

## 📦 Files Delivered

### Core Files

1. **decompose_agent.py** - Main decompose agent implementation
   - `DecomposeAgent` - Recursive task solver
   - `DecomposeTrace` - Execution trace tracking
   - Complexity judgment logic
   - Task decomposition logic
   - Parallel and sequential execution

2. **decompose_utils.py** - Supporting data structures
   - `TaskComplexity` - Complexity judgment result
   - `SubTask` - Subtask specification
   - `CompositionOperator` - Composition code wrapper
   - `ControlFlow` - Execution control flow
   - `TaskDecomposition` - Complete decomposition
   - Utility functions for analysis

3. **composition_executor.py** - Safe code execution
   - `CompositionExecutor` - Execute composition operators
   - Code validation and safety checks
   - Sandboxed execution environment
   - Error handling

4. **decompose_example.py** - Usage examples
   - Simple task examples
   - Complex task examples
   - Parallel execution examples
   - Sequential execution examples
   - Performance comparisons

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from semantic_parser.modules.verdant.decompose_agent import DecomposeAgent
from semantic_parser import ActionRegistry, AzureOpenAIClient

# Setup
llm_client = AzureOpenAIClient(...)
action_registry = ActionRegistry()
# ... register actions ...

agent = DecomposeAgent(
    llm_client=llm_client,
    action_registry=action_registry,
    max_depth=5,
    verbose=True
)

# Solve a task
async def solve_task():
    trace = await agent.solve("Your complex task here")
    print(f"Result: {trace.result}")
    print(f"Success: {trace.error is None}")

asyncio.run(solve_task())
```

### Run Examples

```bash
# Interactive mode
python decompose_example.py

# Specific example
python decompose_example.py 1  # Simple task
python decompose_example.py 2  # Complex task
python decompose_example.py 4  # Parallel execution
python decompose_example.py 6  # All examples
```

---

## 📊 How It Works

### The Decomposition Process

```
┌─────────────────────────────────────────┐
│         Input: Complex Task             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Step 1: Judge Complexity              │
│   LLM decides: Simple or Complex?       │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │ SIMPLE  │      │ COMPLEX  │
   └────┬────┘      └─────┬────┘
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────────────┐
│ Execute      │   │ Step 2: Decompose    │
│ Action       │   │ - Create subtasks    │
│ Directly     │   │ - Generate composer  │
└──────┬───────┘   │ - Determine flow     │
       │           └─────────┬────────────┘
       │                     │
       │                     ▼
       │           ┌──────────────────────┐
       │           │ Step 3: Solve Each   │
       │           │ Subtask Recursively  │
       │           │ (Back to Step 1)     │
       │           └─────────┬────────────┘
       │                     │
       │                     ▼
       │           ┌──────────────────────┐
       │           │ Step 4: Compose      │
       │           │ Execute composition  │
       │           │ operator on results  │
       │           └─────────┬────────────┘
       │                     │
       └─────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Final Result                    │
└─────────────────────────────────────────┘
```

### Example: Complex Task Flow

**Task:** "Compare MOIC of Q4 2024 vs Q4 2023 and rank funds by improvement"

```
Depth 0: Complex Task
├─ Decompose into:
│  ├─ Subtask 1: "Get MOIC for all funds in Q4 2024"
│  │  └─ Depth 1: Simple → Execute GenerateAndExecuteSQL
│  │
│  ├─ Subtask 2: "Get MOIC for all funds in Q4 2023"
│  │  └─ Depth 1: Simple → Execute GenerateAndExecuteSQL
│  │
│  └─ Subtask 3: "Calculate improvement and rank"
│     └─ Depth 1: Simple → Execute with subtask results
│
├─ Control Flow: Sequential (Subtask 3 depends on 1 & 2)
│
├─ Composition Operator:
│  def compose(results):
│      moic_2024 = results["subtask_1"]
│      moic_2023 = results["subtask_2"]
│      # Calculate improvements and rank
│      ...
│      return ranked_funds
│
└─ Execute and Compose → Final Result
```

---

## 🎯 Key Features

### 1. Recursive Decomposition
- **Automatic breakdown** of complex tasks
- **Inductive reasoning**: Assumes subtasks can be solved perfectly
- **Depth limiting**: Prevents infinite recursion
- **Trace tracking**: Complete execution history

### 2. Parallel Execution
- **Concurrent subtasks**: Independent tasks run in parallel
- **Async/await**: Built on Python asyncio
- **Performance gain**: Faster than sequential for independent tasks

### 3. Dynamic Composition
- **LLM-generated code**: Composition operators created on-the-fly
- **Safe execution**: Sandboxed environment with validation
- **Flexible merging**: Handles different result types

### 4. Control Flow
- **Automatic determination**: LLM decides parallel vs sequential
- **Dependency handling**: Topological sorting for sequential tasks
- **Context passing**: Dependency results available to dependent tasks

---

## 💡 Usage Examples

### Example 1: Simple Task (No Decomposition)

```python
task = "What is the average MOIC for all funds in 2024?"

trace = await agent.solve(task)

# Output:
# [Depth 0] Solving task: What is the average MOIC...
# ✓ Simple task completed with GenerateAndExecuteSQL
# Result: 1.85
```

### Example 2: Complex Task (With Decomposition)

```python
task = """Compare MOIC performance Q4 2024 vs Q4 2023.
Calculate improvement percentage and rank funds."""

trace = await agent.solve(task)

# Output:
# [Depth 0] Solving task: Compare MOIC performance...
# ↓ Decomposing into 3 subtasks
#   [Depth 1] Solving task: Get Q4 2024 MOIC...
#   ✓ Simple task completed
#   [Depth 1] Solving task: Get Q4 2023 MOIC...
#   ✓ Simple task completed
#   [Depth 1] Solving task: Calculate improvements...
#   ✓ Simple task completed
# ✓ Complex task completed
```

### Example 3: Parallel Execution

```python
task = """Provide fund analysis:
1. Average MOIC
2. Top 5 funds by capital
3. Funds with negative change"""

trace = await agent.solve(task)

# Subtasks run in parallel (all independent)
# Faster execution than sequential
```

### Example 4: Sequential with Dependencies

```python
task = """Find funds with MOIC > 2.0,
then calculate their total cash flow,
then rank by cash flow."""

trace = await agent.solve(task)

# Subtasks run sequentially:
# 1. Find funds with MOIC > 2.0
# 2. Calculate cash flow (uses result from step 1)
# 3. Rank (uses result from step 2)
```

---

## 📈 Analyzing Results

### View Decomposition Tree

```python
from semantic_parser.modules.verdant.decompose_utils import format_trace_tree

print(format_trace_tree(trace))
```

**Output:**
```
[COMPLEX] Compare MOIC performance Q4 2024 vs Q4 2023...
  → 3 subtasks (sequential)
    [SIMPLE] Get Q4 2024 MOIC data
      → Action: GenerateAndExecuteSQL
      ✓ Result: | Fund | MOIC_2024 |...
    [SIMPLE] Get Q4 2023 MOIC data
      → Action: GenerateAndExecuteSQL
      ✓ Result: | Fund | MOIC_2023 |...
    [SIMPLE] Calculate improvement and rank
      → Action: GenerateAndExecuteSQL
      ✓ Result: | Fund | Improvement |...
  ✓ Result: Final ranked list...
```

### Get Statistics

```python
from semantic_parser.modules.verdant.decompose_utils import analyze_decomposition_stats

stats = analyze_decomposition_stats(trace)
print(f"Total Tasks: {stats['total_tasks']}")
print(f"Simple Tasks: {stats['simple_tasks']}")
print(f"Complex Tasks: {stats['complex_tasks']}")
print(f"Max Depth: {stats['max_depth']}")
print(f"Total Duration: {stats['total_duration']:.2f}s")
print(f"Actions Used: {stats['actions_used']}")
print(f"Control Flows: {stats['control_flows']}")
```

### Save Trace

```python
import json

def save_trace(trace, filename):
    with open(filename, 'w') as f:
        json.dump(agent.get_trace_summary(trace), f, indent=2, default=str)

save_trace(trace, "decompose_trace.json")
```

---

## ⚙️ Configuration

### Agent Parameters

```python
agent = DecomposeAgent(
    llm_client=llm_client,
    action_registry=action_registry,
    max_depth=5,        # Maximum recursion depth
    max_subtasks=5,     # Max subtasks per decomposition
    verbose=True        # Print detailed logs
)
```

**max_depth:**
- Default: 5
- Prevents infinite recursion
- Increase for very complex tasks
- Decrease to limit computation

**max_subtasks:**
- Default: 5
- Max number of subtasks per decomposition
- More subtasks = finer granularity
- Fewer subtasks = simpler decomposition

**verbose:**
- Default: False
- Set to True to see detailed execution logs
- Shows decomposition decisions
- Displays subtask execution

---

## 🔐 Safety Features

### Composition Code Execution

The composition executor implements multiple safety measures:

**1. Code Validation:**
- AST parsing to check structure
- Blocks import statements
- Blocks exec/eval/compile
- Blocks file operations
- Allows only safe builtins

**2. Sandboxed Execution:**
- Limited namespace
- Restricted builtins only
- No access to system functions
- Timeout protection (30s default)

**3. Safe Builtins:**
```python
# Allowed:
int, float, str, list, dict, set, tuple
len, sum, min, max, sorted, any, all
range, enumerate, zip, map, filter

# Blocked:
import, exec, eval, open, file, __import__
```

---

## 📊 Performance Comparison

### Decompose Agent vs Sequential ReACT

| Aspect | Decompose Agent | Sequential ReACT |
|--------|----------------|------------------|
| **Complex Tasks** | Automatic decomposition | Manual step-by-step |
| **Parallel Tasks** | ✅ Concurrent execution | ❌ Sequential only |
| **Clarity** | ✅ Clear task structure | Reasoning steps |
| **Speed** | ⚡ Faster (parallel) | Slower (sequential) |
| **LLM Calls** | More (decomposition) | Fewer |
| **Best For** | Multi-part queries | Single-path queries |

**When to Use Decompose Agent:**
- Complex multi-part queries
- Tasks with independent subtasks
- When parallelization helps
- When task structure is important

**When to Use ReACT Agent:**
- Simple or single-step queries
- Linear reasoning paths
- When LLM calls should be minimized

---

## 🎓 Advanced Usage

### Custom Composition Templates

```python
from semantic_parser.modules.verdant.composition_executor import create_simple_composition

# Use predefined templates
merge_code = create_simple_composition("merge")
concat_code = create_simple_composition("concatenate")
sum_code = create_simple_composition("sum")
```

### Test Composition Operators

```python
from semantic_parser.modules.verdant.composition_executor import CompositionExecutor

executor = CompositionExecutor()

# Test with sample data
test_results = {
    "subtask_1": [1, 2, 3],
    "subtask_2": [4, 5, 6]
}

result = executor.test_composition(composition_op, test_results)
print(f"Test passed: {result['success']}")
```

### Manual Decomposition

```python
from semantic_parser.modules.verdant.decompose_utils import (
    SubTask, CompositionOperator, ControlFlow, TaskDecomposition
)

# Manually create decomposition
subtasks = [
    SubTask("subtask_1", "Get MOIC 2024", depends_on=[]),
    SubTask("subtask_2", "Get MOIC 2023", depends_on=[]),
    SubTask("subtask_3", "Compare", depends_on=["subtask_1", "subtask_2"])
]

composition_op = CompositionOperator(
    code="def compose(results): ...",
    input_keys=["subtask_1", "subtask_2", "subtask_3"],
    output_key="final"
)

control_flow = ControlFlow(
    type="sequential",
    subtask_order=["subtask_1", "subtask_2", "subtask_3"]
)

decomposition = TaskDecomposition(subtasks, composition_op, control_flow)
decomposition.validate()  # Check for errors
```

---

## 🐛 Troubleshooting

### Issue: Max Depth Reached

**Cause:** Task decomposition too deep

**Solution:**
```python
agent = DecomposeAgent(
    max_depth=10,  # Increase depth limit
    ...
)
```

### Issue: Composition Fails

**Cause:** Generated code has errors

**Solution:**
- Check verbose logs for composition code
- LLM may need better examples
- Manually inspect failing composition
- Use simpler task descriptions

### Issue: Slow Execution

**Cause:** Too many LLM calls for decomposition

**Solutions:**
- Reduce max_subtasks
- Use ReACT agent for simple tasks
- Cache decomposition results

### Issue: Circular Dependencies

**Cause:** Subtasks depend on each other circularly

**Solution:**
- LLM should detect this
- If it occurs, check decomposition logic
- Simplify task description

---

## 🎯 Best Practices

1. **Task Description:**
   - Be specific and clear
   - Break down very complex tasks yourself first
   - Specify parallel vs sequential if known

2. **Depth Management:**
   - Start with max_depth=5
   - Increase only if needed
   - Monitor actual depths used

3. **Action Design:**
   - Design actions to be composable
   - Each action should do one thing well
   - Clear action descriptions help decomposition

4. **Performance:**
   - Use parallel execution when possible
   - Cache decomposition results if repeating
   - Profile to find bottlenecks

5. **Debugging:**
   - Always use verbose=True when debugging
   - Save traces for analysis
   - Check composition operators manually

---

## 📝 Summary

The Decompose Agent provides:

✅ **Recursive task decomposition** - Automatically breaks down complex tasks  
✅ **Parallel execution** - Concurrent processing of independent subtasks  
✅ **Dynamic composition** - LLM-generated code to merge results  
✅ **Safe execution** - Sandboxed environment for generated code  
✅ **Flexible control flow** - Parallel or sequential based on dependencies  
✅ **Complete tracing** - Full execution history and statistics  

Perfect for complex, multi-part queries that benefit from structured decomposition and parallel execution!
