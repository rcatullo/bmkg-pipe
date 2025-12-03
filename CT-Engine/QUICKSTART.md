# Quick Start Guide

Get up and running with the Semantic Parser Framework in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- Azure OpenAI account with o3 model deployment
- pip package manager

## Installation

### Step 1: Install the framework

```bash
cd semantic_parser
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### Step 2: Configure environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
```

## Basic Usage

### Example 1: Simple SQL Parsing

Create a file `my_parser.py`:

```python
from semantic_parser import ReACTEngine, ActionRegistry, AzureOpenAIClient
from semantic_parser.example_actions import FinishAction, ExampleSchemaRetrievalAction

# Initialize
llm_client = AzureOpenAIClient(deployment_name="o3")
registry = ActionRegistry()
registry.register(FinishAction())
registry.register(ExampleSchemaRetrievalAction())

engine = ReACTEngine(llm_client, registry, verbose=True)

# Parse
result = engine.parse(
    query="Find all users older than 18",
    target_format="SQL"
)

print(f"Generated SQL: {result.final_output}")
```

Run it:

```bash
python my_parser.py
```

### Example 2: Using Provided Examples

Run the included example:

```bash
python example_usage.py
```

## Adding Your First Custom Action

### Step 1: Create action file

Create `my_actions.py`:

```python
from semantic_parser.action_protocol import Action, ActionOutput
from typing import Dict, Any

class MySchemaAction(Action):
    def __init__(self):
        super().__init__(
            name="get_my_schema",
            description="Retrieves schema from my database"
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"}
            },
            "required": ["table_name"]
        }
    
    def execute(self, table_name: str, **kwargs) -> ActionOutput:
        # TODO: Replace with actual schema retrieval
        schema = {
            "table": table_name,
            "columns": ["id", "name", "email"]
        }
        return ActionOutput(success=True, result=schema)
```

### Step 2: Register and use it

```python
from my_actions import MySchemaAction

registry = ActionRegistry()
registry.register(FinishAction())
registry.register(MySchemaAction())

engine = ReACTEngine(llm_client, registry)
result = engine.parse("Get all user names", "SQL")
```

## Common Patterns

### Pattern 1: Multiple Target Formats

```python
# SQL
sql_result = engine.parse(query, "SQL")

# SPARQL
sparql_result = engine.parse(query, "SPARQL")

# Cypher
cypher_result = engine.parse(query, "Cypher")
```

### Pattern 2: Streaming Results

```python
for item in engine.run_streaming(query, "SQL"):
    if isinstance(item, ReasoningStep):
        print(f"Step {item.step_number}: {item.action.action_name}")
    else:
        print(f"Final: {item.final_output}")
```

### Pattern 3: Accessing Reasoning History

```python
result = engine.parse(query, "SQL")

# Print each reasoning step
for step in result.steps:
    print(f"\nStep {step.step_number}:")
    print(f"  Thought: {step.thought.content}")
    print(f"  Action: {step.action.action_name}")
    print(f"  Result: {step.observation.result}")

# Get metrics
print(f"Total steps: {result.get_step_count()}")
print(f"Duration: {result.get_duration()} seconds")
```

## Project Structure

```
semantic_parser/
├── semantic_parser/           # Main package
│   ├── __init__.py           # Public API
│   ├── action_protocol.py    # Action interface
│   ├── llm_client.py         # Azure OpenAI client
│   ├── reasoning_step.py     # Data models
│   ├── prompt_builder.py     # Prompt construction
│   ├── react_engine.py       # Main engine
│   └── example_actions.py    # Example actions
├── example_usage.py          # Usage examples
├── custom_actions_template.py # Action templates
├── README.md                 # Full documentation
├── BUILDING_ACTIONS.md       # Action development guide
├── ARCHITECTURE.md           # Architecture details
├── requirements.txt          # Dependencies
└── .env.example             # Environment template
```

## Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Check BUILDING_ACTIONS.md** to learn how to create powerful custom actions
3. **Review ARCHITECTURE.md** to understand the framework design
4. **Look at example_actions.py** for action implementation examples
5. **Explore custom_actions_template.py** for copy-paste templates

## Troubleshooting

### Error: "Azure endpoint and API key must be provided"

**Solution**: Make sure your `.env` file has correct values:
```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=sk-...
```

### Error: "Action not found in registry"

**Solution**: Make sure you registered the action:
```python
registry.register(YourAction())
```

### LLM not selecting correct actions

**Solution**: Improve action descriptions with:
- Clear use cases
- Input/output examples
- When to use vs. when not to use

### Actions failing

**Solution**: Add error handling in `execute()`:
```python
def execute(self, **kwargs):
    try:
        # Your logic
        return ActionOutput(success=True, result=result)
    except Exception as e:
        return ActionOutput(success=False, error=str(e))
```

## Getting Help

- **Documentation**: See README.md, BUILDING_ACTIONS.md, ARCHITECTURE.md
- **Examples**: Check example_usage.py and example_actions.py
- **Templates**: Use custom_actions_template.py as a starting point

## Example Output

When you run the parser, you'll see output like:

```
============================================================
Step 1/10
============================================================

Thought: I need to understand the database schema first
Action: get_schema
Parameters: {'database': 'users'}
Observation: {'tables': ['users', 'orders'], ...}

============================================================
Step 2/10
============================================================

Thought: Now I can write the SQL query
Action: finish
Parameters: {'output': 'SELECT * FROM users WHERE age > 18'}

============================================================
RESULTS
============================================================

Final Output:
SELECT * FROM users WHERE age > 18

Total Steps: 2
Duration: 3.45 seconds
Completed: True
```

Happy parsing! 🚀
