# Building Custom Actions: Complete Guide

This guide provides detailed instructions for creating powerful custom actions for the Semantic Parser Framework.

## Table of Contents

1. [Action Lifecycle](#action-lifecycle)
2. [Action Design Patterns](#action-design-patterns)
3. [Common Use Cases](#common-use-cases)
4. [Best Practices](#best-practices)
5. [Advanced Techniques](#advanced-techniques)

## Action Lifecycle

Understanding how actions work in the framework:

```
1. Registration Phase
   └─> Action added to ActionRegistry
   └─> Action spec generated for LLM

2. Selection Phase
   └─> LLM receives all action specs
   └─> LLM reasons about which action to use
   └─> LLM outputs action name + parameters

3. Validation Phase
   └─> Parameters validated against schema
   └─> Missing required fields flagged

4. Execution Phase
   └─> Action.execute() called with parameters
   └─> Logic runs, results produced
   └─> ActionOutput returned

5. Observation Phase
   └─> Result added to reasoning trace
   └─> Context updated for next step
```

## Action Design Patterns

### Pattern 1: Information Retrieval

Actions that fetch information from external sources.

```python
class SchemaRetrievalAction(Action):
    """Retrieves database schema information"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        super().__init__(
            name="get_schema",
            description="""Get database schema including tables, columns, and relationships.
            
            Use this to understand database structure before writing queries.
            """
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "database": {"type": "string"},
                "include_samples": {"type": "boolean", "default": False}
            },
            "required": ["database"]
        }
    
    def execute(self, database: str, include_samples: bool = False, **kwargs):
        schema = self.db.get_schema(database)
        
        if include_samples:
            schema["samples"] = self.db.get_sample_data(database)
        
        return ActionOutput(
            success=True,
            result=schema,
            metadata={"database": database}
        )
```

### Pattern 2: Validation

Actions that validate outputs before completion.

```python
class SQLValidatorAction(Action):
    """Validates SQL query syntax and semantics"""
    
    def __init__(self, schema_manager):
        self.schema = schema_manager
        super().__init__(
            name="validate_sql",
            description="""Validate SQL query for correctness.
            
            Checks:
            - Syntax validity
            - Table/column existence
            - Type compatibility
            """
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "database": {"type": "string"}
            },
            "required": ["query", "database"]
        }
    
    def execute(self, query: str, database: str, **kwargs):
        errors = []
        warnings = []
        
        # Check syntax
        if not self._is_valid_syntax(query):
            errors.append("Invalid SQL syntax")
        
        # Check schema compliance
        tables = self._extract_tables(query)
        for table in tables:
            if not self.schema.table_exists(database, table):
                errors.append(f"Table '{table}' does not exist")
        
        return ActionOutput(
            success=len(errors) == 0,
            result={
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            },
            error="; ".join(errors) if errors else None
        )
```

### Pattern 3: Transformation

Actions that transform or process data.

```python
class QueryRewriteAction(Action):
    """Rewrites queries to optimize or fix issues"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        super().__init__(
            name="rewrite_query",
            description="""Rewrite a query to fix issues or optimize performance.
            
            Use this when:
            - Query validation fails
            - Query needs optimization
            - Alternative formulation needed
            """
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "goal": {
                    "type": "string",
                    "enum": ["fix_syntax", "optimize", "simplify"]
                }
            },
            "required": ["query", "goal"]
        }
    
    def execute(self, query: str, goal: str, **kwargs):
        if goal == "fix_syntax":
            result = self.optimizer.fix_syntax(query)
        elif goal == "optimize":
            result = self.optimizer.optimize(query)
        else:
            result = self.optimizer.simplify(query)
        
        return ActionOutput(
            success=True,
            result=result,
            metadata={
                "original_query": query,
                "goal": goal
            }
        )
```

### Pattern 4: Search and Retrieval

Actions that search through knowledge bases or examples.

```python
class SimilarQuerySearchAction(Action):
    """Search for similar queries in a database"""
    
    def __init__(self, query_database, embedder):
        self.db = query_database
        self.embedder = embedder
        super().__init__(
            name="search_similar_queries",
            description="""Find similar queries that have been successfully parsed.
            
            Returns examples of:
            - Natural language queries
            - Their corresponding logical forms
            - Similarity scores
            """
        )
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "target_format": {"type": "string"},
                "limit": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, target_format: str = None, limit: int = 5, **kwargs):
        # Get embedding for query
        query_embedding = self.embedder.embed(query)
        
        # Search similar queries
        results = self.db.similarity_search(
            embedding=query_embedding,
            target_format=target_format,
            limit=limit
        )
        
        return ActionOutput(
            success=True,
            result=results,
            metadata={
                "query": query,
                "num_results": len(results)
            }
        )
```

## Common Use Cases

### Use Case 1: SQL Semantic Parser

Complete action set for SQL parsing:

```python
# 1. Schema retrieval
schema_action = DatabaseSchemaAction(connection_string)

# 2. Example search
search_action = SQLExampleSearchAction(example_db)

# 3. Query generation helper
helper_action = SQLHelperAction()

# 4. Validation
validator_action = SQLValidatorAction(schema_manager)

# 5. Execution/testing
test_action = SQLTestAction(connection_string)

# 6. Finish
finish_action = FinishAction()

# Register all
registry = ActionRegistry()
for action in [schema_action, search_action, helper_action, 
               validator_action, test_action, finish_action]:
    registry.register(action)
```

### Use Case 2: SPARQL Semantic Parser

```python
# 1. Ontology retrieval
ontology_action = OntologyRetrievalAction(endpoint)

# 2. Entity linking
entity_action = EntityLinkingAction(entity_db)

# 3. Property mapping
property_action = PropertyMappingAction(ontology)

# 4. SPARQL validation
validator_action = SPARQLValidatorAction()

# 5. Endpoint query
query_action = SPARQLQueryAction(endpoint)

# 6. Finish
finish_action = FinishAction()
```

### Use Case 3: Multi-Format Parser

Support multiple formats with conditional actions:

```python
class AdaptiveSchemaAction(Action):
    """Retrieves schema based on target format"""
    
    def __init__(self):
        self.sql_handler = SQLSchemaHandler()
        self.sparql_handler = SPARQLSchemaHandler()
        self.cypher_handler = CypherSchemaHandler()
        
        super().__init__(
            name="get_schema",
            description="""Get schema for the target format.
            
            Automatically selects the right schema retrieval method based
            on target format (SQL, SPARQL, Cypher).
            """
        )
    
    def execute(self, source: str, target_format: str, **kwargs):
        if target_format == "SQL":
            result = self.sql_handler.get_schema(source)
        elif target_format == "SPARQL":
            result = self.sparql_handler.get_ontology(source)
        elif target_format == "Cypher":
            result = self.cypher_handler.get_graph_schema(source)
        else:
            return ActionOutput(
                success=False,
                result=None,
                error=f"Unsupported format: {target_format}"
            )
        
        return ActionOutput(success=True, result=result)
```

## Best Practices

### 1. Clear Action Descriptions

**Bad:**
```python
description="Gets schema"
```

**Good:**
```python
description="""Retrieve the database schema including all tables, columns, types, and relationships.

Use this action when:
- You need to understand the database structure
- You're unsure about table or column names
- You need to verify data types before writing a query

Input:
- database: Name of the database (required)
- include_samples: Whether to include sample data (optional, default: False)

Output:
- Dictionary with tables, columns, types, and relationships
- Sample data if requested

Example:
Action: get_schema
Action Input: {"database": "ecommerce", "include_samples": true}
"""
```

### 2. Robust Error Handling

```python
def execute(self, **kwargs):
    try:
        # Main logic
        result = self._do_work(kwargs)
        
        return ActionOutput(
            success=True,
            result=result,
            metadata={"execution_time": time.time() - start}
        )
        
    except ValueError as e:
        # Handle expected errors
        return ActionOutput(
            success=False,
            result=None,
            error=f"Invalid input: {str(e)}",
            metadata={"error_type": "validation"}
        )
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in action: {e}", exc_info=True)
        return ActionOutput(
            success=False,
            result=None,
            error=f"Action failed: {str(e)}",
            metadata={"error_type": "unexpected"}
        )
```

### 3. Meaningful Metadata

```python
return ActionOutput(
    success=True,
    result=query_result,
    metadata={
        "execution_time_ms": execution_time,
        "rows_returned": len(query_result),
        "cache_hit": cache_hit,
        "source": data_source,
        "timestamp": datetime.now().isoformat()
    }
)
```

### 4. Input Validation

```python
def execute(self, query: str, database: str, **kwargs):
    # Validate inputs
    if not query or not query.strip():
        return ActionOutput(
            success=False,
            result=None,
            error="Query cannot be empty"
        )
    
    if not self._database_exists(database):
        return ActionOutput(
            success=False,
            result=None,
            error=f"Database '{database}' not found"
        )
    
    # Proceed with execution
    ...
```

## Advanced Techniques

### 1. Stateful Actions

Actions that maintain state across calls:

```python
class ConversationalSchemaAction(Action):
    """Schema action that remembers previous queries"""
    
    def __init__(self):
        self.query_history = []
        self.schema_cache = {}
        super().__init__(name="get_schema", description="...")
    
    def execute(self, database: str, **kwargs):
        # Check cache
        if database in self.schema_cache:
            result = self.schema_cache[database]
            metadata = {"cache_hit": True}
        else:
            result = self._fetch_schema(database)
            self.schema_cache[database] = result
            metadata = {"cache_hit": False}
        
        # Track history
        self.query_history.append({
            "database": database,
            "timestamp": datetime.now()
        })
        
        return ActionOutput(success=True, result=result, metadata=metadata)
```

### 2. Composite Actions

Actions that orchestrate multiple operations:

```python
class ComprehensiveValidationAction(Action):
    """Runs multiple validation checks"""
    
    def __init__(self, validators):
        self.syntax_validator = validators['syntax']
        self.schema_validator = validators['schema']
        self.performance_validator = validators['performance']
        super().__init__(name="comprehensive_validate", description="...")
    
    def execute(self, query: str, database: str, **kwargs):
        results = {
            "syntax": self.syntax_validator.check(query),
            "schema": self.schema_validator.check(query, database),
            "performance": self.performance_validator.estimate(query)
        }
        
        all_valid = all(r["valid"] for r in results.values())
        
        return ActionOutput(
            success=all_valid,
            result=results,
            error="Validation failed" if not all_valid else None
        )
```

### 3. Async Actions

For I/O-bound operations:

```python
class AsyncAPIAction(Action):
    """Async action for external API calls"""
    
    def __init__(self, api_client):
        self.api = api_client
        super().__init__(name="api_call", description="...")
    
    async def execute_async(self, endpoint: str, **kwargs):
        """Async version of execute"""
        result = await self.api.get(endpoint)
        return ActionOutput(success=True, result=result)
    
    def execute(self, endpoint: str, **kwargs):
        """Sync wrapper"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_async(endpoint, **kwargs))
```

### 4. Conditional Actions

Actions that adapt based on context:

```python
class SmartSchemaAction(Action):
    """Retrieves different levels of detail based on query complexity"""
    
    def execute(self, database: str, query_complexity: str = "medium", **kwargs):
        if query_complexity == "simple":
            # Return basic schema only
            result = self._get_basic_schema(database)
        elif query_complexity == "complex":
            # Return detailed schema with examples
            result = self._get_detailed_schema(database)
            result["examples"] = self._get_query_examples(database)
        else:
            # Medium complexity
            result = self._get_standard_schema(database)
        
        return ActionOutput(success=True, result=result)
```

## Testing Your Actions

Always test actions before deployment:

```python
def test_action():
    # Setup
    action = MyCustomAction()
    
    # Test valid input
    result = action.execute(param="valid_value")
    assert result.success
    assert result.result is not None
    
    # Test invalid input
    result = action.execute(param="")
    assert not result.success
    assert result.error is not None
    
    # Test edge cases
    result = action.execute(param=None)
    assert not result.success

if __name__ == "__main__":
    test_action()
```

## Debugging Actions

Enable logging in your actions:

```python
import logging

logger = logging.getLogger(__name__)

class DebugAction(Action):
    def execute(self, **kwargs):
        logger.info(f"Executing action with params: {kwargs}")
        
        try:
            result = self._process(kwargs)
            logger.info(f"Action succeeded: {result}")
            return ActionOutput(success=True, result=result)
        except Exception as e:
            logger.error(f"Action failed: {e}", exc_info=True)
            return ActionOutput(success=False, result=None, error=str(e))
```

## Next Steps

1. Review `custom_actions_template.py` for more examples
2. Implement your first custom action
3. Test it thoroughly
4. Integrate into your semantic parser
5. Monitor performance and iterate

For more help, see the main README or open an issue on GitHub.
