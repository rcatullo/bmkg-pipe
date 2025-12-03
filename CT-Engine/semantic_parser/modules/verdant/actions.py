"""
Actions module for ReACT-style semantic parsing agent.

This module implements concrete actions that follow the Action protocol
defined in action_protocol.py. Each action represents a discrete operation
in the semantic parsing pipeline.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from semantic_parser.action_protocol import Action, ActionInput, ActionOutput
from semantic_parser.modules.verdant.db_utils import DatabaseManager, QueryResult
from semantic_parser.llm_client import AzureOpenAIClient, Message
from semantic_parser.modules.verdant.table_utils import TableInfo, load_table_info, TableDescriber
from semantic_parser.modules.verdant.utils import to_json_string
from pydantic import BaseModel, Field
import os

CACHE_DIR = "/home/jiuding/computational_thinking/semantic_parser/modules/verdant/cache"

# ============================================================================
# Input Models
# ============================================================================

class FetchRelatedColumnsInput(ActionInput):
    """Input schema for FetchRelatedColumns action"""
    query: str = Field(description="The full natural language query")
    column_dictionary: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Optional dictionary mapping table names to column lists. If not provided, searches across all available tables"
    )
    background_knowledge: Optional[str] = Field(
        default=None,
        description="Optional background knowledge about the query domain to help with column selection"
    )

class FetchBackgroundKnowledgeInput(ActionInput):
    """Input schema for FetchBackgroundKnowledge action"""
    query: str = Field(description="The full natural language query")


class SketchSQLInput(ActionInput):
    """Input schema for SketchSQL action"""
    query: str = Field(description="The full natural language query to sketch")
    selected_columns: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Optional dictionary of selected tables and columns"
    )


class GenerateSQLInput(ActionInput):
    """Input schema for GenerateSQL action"""
    natural_language_sketch: str = Field(description="Natural language description of SQL logic")
    selected_columns: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Optional dictionary of selected tables and columns"
    )


class ExecuteSQLInput(ActionInput):
    """Input schema for ExecuteSQL action"""
    sql_query: str = Field(description="SQL query to execute")
    
    
class GenerateAndExecuteSQLInput(ActionInput):
    """Input schema for GenerateAndExecuteSQL action"""
    natural_language_sketch: str = Field(description="Natural language description of SQL logic to generate and execute")
    selected_columns: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Optional dictionary of selected tables and columns"
    )


class FinishInput(ActionInput):
    """Input schema for Finish action"""
    query: str = Field(description="The original natural language query")
    result_table: str = Field(description="The result table from query execution")


# ============================================================================
# Action Implementations
# ============================================================================
    

class FetchRelatedColumns(Action):
    """
    Action to identify and retrieve relevant columns from tables based on a task description.
    
    This action analyzes the task description and selects the most relevant tables and columns
    needed to answer the query, along with reasoning for each selection.
    """
    
    def __init__(self, openai_client: AzureOpenAIClient):
        """
        Initialize FetchRelatedColumns action.
        
        Args:
            openai_client: AzureOpenAIClient for LLM queries
        """
        super().__init__(
            name="FetchRelatedColumns",
            description="""Identify and retrieve relevant columns from database tables based on the task description (and background knowledge if fetched already).
            
            Use this action when you need to:
            - Determine which tables and columns are relevant for a query
            - Get detailed descriptions of selected columns
            - Understand the reasoning behind column selection
            
            The action returns a structured list of relevant columns with their table information,
            descriptions, and reasoning for selection."""
        )
        
        self.client = openai_client
        self.table_descriptions: Dict[str, TableInfo] = {}
        
        self.available_tables = [
            "FundCashFlow",
            "FundInvestmentProperties",
            "FundInvestmentTimeProperties",
            "FundTimeProperties",
            "HandTransformedPortfolioInvestmentswithCFRaw"
        ]
        
        self._initialize_table_descriptions()
        
    def _initialize_table_descriptions(self):
        """Generate and cache descriptions for all available tables."""
        logging.info("Initializing table descriptions...")
        for table_name in self.available_tables:
            
            if os.path.exists(os.path.join(CACHE_DIR, f"{table_name.lower()}")):
                logging.info(f"Loading cached description for table: {table_name}")
                table_info = load_table_info(os.path.join(CACHE_DIR, f"{table_name.lower()}"))
                self.table_descriptions[table_name] = table_info
                continue
                
            try:
                describer = TableDescriber(table_name)
                table_info = describer.describe_table(format_description=False)
                self.table_descriptions[table_name] = table_info
                logging.info(f"Loaded description for table: {table_name}")
            except Exception as e:
                logging.error(f"Failed to describe table {table_name}: {e}")
                
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The full natural language query"
                },
                "column_dictionary": {
                    "type": "object",
                    "description": "Optional dictionary mapping table names to column lists. If not provided, searches all tables",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "background_knowledge": {
                    "type": "string",
                    "description": "Optional background knowledge about the query domain to help with column selection"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, column_dictionary: Optional[Dict[str, List[str]]] = None, background_knowledge: Optional[str] = None) -> ActionOutput:
        """
        Execute column selection based on task description.
        
        Args:
            query: The full natural language query
            column_dictionary: Optional dict of {table_name: [column_names]}
            background_knowledge: Optional background knowledge to help with column selection
            
        Returns:
            ActionOutput with selected columns, descriptions, and reasoning
        """
        try:
            # Determine scope of search
            if column_dictionary:
                search_scope = column_dictionary
            else:
                # Use all available tables
                search_scope = {
                    table_name: [col['column_name'] for col in table_info.columns]
                    for table_name, table_info in self.table_descriptions.items()
                }
            
            # Build context for LLM
            context_parts = []
            for table_name, columns in search_scope.items():
                if table_name not in self.table_descriptions:
                    continue
                    
                table_info = self.table_descriptions[table_name]
                context_parts.append(f"\n**Table: {table_name}**")
                context_parts.append(f"Description: {table_info.overall_description}")
                context_parts.append("Columns:")
                
                for col_name in columns:
                    if col_name in table_info.column_descriptions:
                        col_desc = table_info.column_descriptions[col_name]
                        context_parts.append(f"  - {col_name}: {col_desc}")
            
            context = "\n".join(context_parts)
            
            # Build the prompt with optional background knowledge
            background_section = ""
            if background_knowledge:
                background_section = f"""
**Background Knowledge:**
{background_knowledge}

Use this background knowledge to better understand the query context and select the most relevant columns.
"""
            
            # Create prompt for LLM
            prompt = f"""Given the following task and available database schema, identify the relevant columns needed to complete the task.

**Task Description:**
{query}
{background_section}
**Available Schema:**
{context}

Please analyze and return a JSON object with the following structure:
{{
    "selected_columns": [
        {{
            "table": "table_name",
            "column": "column_name",
            "description": "description of what this column represents",
            "reasoning": "why this column is relevant for the task"
        }},
        ...
    ]
}}

Guidelines:
- Only select columns that are directly relevant to the task
- Provide clear reasoning for each selection
- Consider relationships between tables if multiple tables are involved

Return ONLY the JSON object, no additional text.
"""
            
            system_prompt = """You are a database expert analyzing query requirements. 
Your job is to identify the minimal set of relevant columns needed for a task, 
with clear reasoning for each selection."""
            
            # Query LLM
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt)
            ]
            response = self.client.chat_completion(messages).content
            
            # Parse response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            selected_columns = parsed.get("selected_columns", [])
            
            # Format result
            result = to_json_string(selected_columns, indent=2)
            
            # logging.info(f"Selected {len(selected_columns)} relevant columns from {len(result['tables_involved'])} tables")
            
            return ActionOutput(
                success=True,
                result=result,
                metadata={
                    "search_scope": list(search_scope.keys()),
                    "total_columns_searched": sum(len(cols) for cols in search_scope.values()),
                    "used_background_knowledge": background_knowledge is not None
                }
            )
            
        except Exception as e:
            logging.error(f"FetchRelatedColumns failed: {e}")
            return ActionOutput(
                success=False,
                result=None,
                error=str(e)
            )


class FetchBackgroundKnowledge(Action):
    """
    Action to identify and elaborate on key terminology in a query.
    
    This action helps understand domain-specific terms and concepts that appear
    in queries, providing expert-level explanations.
    """
    
    def __init__(self, openai_client: AzureOpenAIClient):
        """
        Initialize FetchBackgroundKnowledge action.
        
        Args:
            openai_client: AzureOpenAIClient for LLM queries
        """
        super().__init__(
            name="FetchBackgroundKnowledge",
            description="""Identify and explain key terminology and concepts in a query from an expert's perspective.
            
            Use this action when you need to:
            - Understand domain-specific terminology in a query
            - Get expert explanations of financial or technical concepts
            - Clarify ambiguous terms before proceeding with query execution
            
            The action returns detailed explanations of key terms and concepts."""
        )
        self.client = openai_client
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The full natural language query"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str) -> ActionOutput:
        """
        Extract and explain key terminology from task description.
        
        Args:
            query: Description of the task or query
            
        Returns:
            ActionOutput with terminology explanations
        """
        try:
            prompt = f"""Analyze the following task/query and identify key terminology that requires expert explanation.

**Task/Query:**
{query}

Please provide a JSON object with the following structure:
{{
    "key_terms": [
        {{
            "term": "term or concept",
            "definition": "expert-level definition and explanation",
            "context": "how this term relates to the query",
            "importance": "why understanding this term is important for the task"
        }},
        ...
    ],
    "domain_context": "overall domain or field this query relates to (e.g., finance, healthcare, etc.)",
    "assumptions": ["any assumptions or interpretations made about ambiguous terms"]
}}

Guidelines:
- Focus on domain-specific, technical, or potentially ambiguous terms
- Provide clear, expert-level explanations
- Explain how each term relates to the task
- Identify any assumptions needed to interpret the query

Return ONLY the JSON object, no additional text.
"""
            
            system_prompt = """You are a domain expert specializing in financial databases, investment banking, and fund management.
Your job is to identify and explain key terminology that someone needs to understand to properly interpret and execute queries."""
            
            # Query LLM
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt)
            ]
            response = self.client.chat_completion(messages).content
            
            # Parse response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            result = to_json_string({
                "key_terms": parsed.get("key_terms", []),
                "domain_context": parsed.get("domain_context", ""),
                "assumptions": parsed.get("assumptions", []),
            })
            
            # logging.info(f"Identified {result['term_count']} key terms in domain: {result['domain_context']}")
            
            return ActionOutput(
                success=True,
                result=result,
                metadata={"query_length": len(query)}
            )
            
        except Exception as e:
            logging.error(f"FetchBackgroundKnowledge failed: {e}")
            return ActionOutput(
                success=False,
                result=None,
                error=str(e)
            )


class SketchSQL(Action):
    """
    Action to create a natural language sketch of SQL query logic.
    
    This action describes how to approach writing the SQL query without
    generating the actual SQL code, helping to plan the query structure.
    """
    
    def __init__(self, openai_client: AzureOpenAIClient):
        """
        Initialize SketchSQL action.
        
        Args:
            openai_client: AzureOpenAIClient for LLM queries
        """
        super().__init__(
            name="SketchSQL",
            description="""Create a natural language description of SQL query logic and structure.
            
            Use this action when you need to:
            - Plan the approach for a SQL query before writing code
            - Describe the logical steps and operations needed
            - Identify which columns and tables to use
            - Outline joins, filters, aggregations, etc.
            
            The action returns a clear, step-by-step description of the SQL logic."""
        )
        self.client = openai_client
        
        self.table_descriptions: Dict[str, TableInfo] = {}
        
        self.available_tables = [
            "FundCashFlow",
            "FundInvestmentProperties",
            "FundInvestmentTimeProperties",
            "FundTimeProperties",
            "HandTransformedPortfolioInvestmentswithCFRaw"
        ]
        
        self._initialize_table_descriptions()
    
    def _initialize_table_descriptions(self):
        """Generate and cache descriptions for all available tables."""
        logging.info("Initializing table descriptions...")
        for table_name in self.available_tables:
            
            if os.path.exists(os.path.join(CACHE_DIR, f"{table_name.lower()}")):
                logging.info(f"Loading cached description for table: {table_name}")
                table_info = load_table_info(os.path.join(CACHE_DIR, f"{table_name.lower()}"))
                self.table_descriptions[table_name] = table_info
                continue
                
            try:
                describer = TableDescriber(table_name)
                table_info = describer.describe_table(format_description=False)
                self.table_descriptions[table_name] = table_info
                logging.info(f"Loaded description for table: {table_name}")
            except Exception as e:
                logging.error(f"Failed to describe table {table_name}: {e}")
                
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The full natural language query"
                },
                "selected_columns": {
                    "type": "object",
                    "description": "Optional dictionary of selected tables and columns",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, selected_columns: Optional[Dict[str, List[str]]] = None) -> ActionOutput:
        """
        Create natural language sketch of SQL query.
        
        Args:
            query: The full natural language query
            selected_columns: Optional dict of selected tables and columns
            
        Returns:
            ActionOutput with SQL sketch in natural language
        """
        try:
            # Build schema context
            context_parts = []
            
            if selected_columns:
                for table_name, columns in selected_columns.items():
                    if table_name not in self.table_descriptions:
                        continue
                    
                    table_info = self.table_descriptions[table_name]
                    context_parts.append(f"\n**Table: {table_name}**")
                    context_parts.append(f"Description: {table_info.overall_description}")
                    context_parts.append("Available Columns:")
                    
                    for col_name in columns:
                        if col_name in table_info.column_descriptions:
                            col_desc = table_info.column_descriptions[col_name]
                            context_parts.append(f"  - {col_name}: {col_desc}")
            else:
                # Include all tables if no selection provided
                for table_name, table_info in self.table_descriptions.items():
                    context_parts.append(f"\n**Table: {table_name}**")
                    context_parts.append(f"Description: {table_info.overall_description}")
            
            context = "\n".join(context_parts) if context_parts else "No specific schema context provided"
            
            prompt = f"""Given the following task and database schema, create a natural language sketch describing how to write the SQL query.

**Task Description:**
{query}

**Schema Context:**
{context}

Please provide a JSON object with the following structure:
{{
    "approach": "high-level approach to solving this query",
    "sketch": "Step-by-step description of the SQL logic needed",
    "key_operations": [
        "List of key SQL operations needed (e.g., JOIN, GROUP BY, WHERE, etc.)"
    ],
    "columns_needed": {{
        "table_name": ["column1", "column2"],
        ...
    }},
    "potential_challenges": ["Any potential issues or complexities to consider"]
}}

Guidelines:
- Describe the logic in natural language, not SQL code
- Be specific about which columns and tables to use
- Explain any joins, filters, aggregations, or ordering needed
- Identify potential edge cases or challenges

Return ONLY the JSON object, no additional text.
"""
            
            system_prompt = """You are a SQL expert who excels at breaking down complex queries into logical steps.
Your job is to create clear, natural language descriptions of SQL query logic without writing actual SQL code."""
            
            # Query LLM
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt)
            ]
            response = self.client.chat_completion(messages).content
            
            # Parse response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            result = parsed.get("sketch", "")
            
            # logging.info(f"Created SQL sketch with {len(result)} characters")
            
            return ActionOutput(
                success=True,
                result=result,
                metadata={"task_length": len(query)}
            )
            
        except Exception as e:
            logging.error(f"SketchSQL failed: {e}")
            return ActionOutput(
                success=False,
                result=None,
                error=str(e)
            )


class GenerateSQL(Action):
    """
    Action to generate executable SQL code from a natural language sketch.
    
    This action converts a natural language description of SQL logic into
    actual SQL code that can be executed against the database.
    """
    
    def __init__(self, openai_client: AzureOpenAIClient):
        """
        Initialize GenerateSQL action.
        
        Args:
            openai_client: AzureOpenAIClient for LLM queries
        """
        super().__init__(
            name="GenerateSQL",
            description="""Generate executable SQL code from a natural language description of query logic.
            
            Use this action when you need to:
            - Convert a natural language sketch into SQL code
            - Generate a query based on a task description
            - Produce executable SQL that follows best practices
            
            The action returns syntactically correct SQL code ready for execution."""
        )
        self.client = openai_client
        
        self.table_descriptions: Dict[str, TableInfo] = {}
        
        self.available_tables = [
            "FundCashFlow",
            "FundInvestmentProperties",
            "FundInvestmentTimeProperties",
            "FundTimeProperties",
            "HandTransformedPortfolioInvestmentswithCFRaw"
        ]
        
        self._initialize_table_descriptions()
        
        
        
    def _initialize_table_descriptions(self):
        """Generate and cache descriptions for all available tables."""
        logging.info("Initializing table descriptions...")
        for table_name in self.available_tables:
            
            if os.path.exists(os.path.join(CACHE_DIR, f"{table_name.lower()}")):
                logging.info(f"Loading cached description for table: {table_name}")
                table_info = load_table_info(os.path.join(CACHE_DIR, f"{table_name.lower()}"))
                self.table_descriptions[table_name] = table_info
                continue
                
            try:
                describer = TableDescriber(table_name)
                table_info = describer.describe_table(format_description=False)
                self.table_descriptions[table_name] = table_info
                logging.info(f"Loaded description for table: {table_name}")
            except Exception as e:
                logging.error(f"Failed to describe table {table_name}: {e}")
                
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "natural_language_sketch": {
                    "type": "string",
                    "description": "Natural language description of SQL logic"
                },
                "selected_columns": {
                    "type": "object",
                    "description": "Optional dictionary of selected tables and columns",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["natural_language_sketch"]
        }
    
    def execute(self, natural_language_sketch: str, selected_columns: Optional[Dict[str, List[str]]] = None) -> ActionOutput:
        """
        Generate SQL code from natural language sketch.
        
        Args:
            natural_language_sketch: Natural language description of SQL logic
            selected_columns: Optional dict of selected tables and columns
            
        Returns:
            ActionOutput with generated SQL query
        """
        try:
            # Build schema context
            context_parts = []
            
            if selected_columns:
                for table_name, columns in selected_columns.items():
                    if table_name not in self.table_descriptions:
                        continue
                    
                    table_info = self.table_descriptions[table_name]
                    context_parts.append(f"\n**Table: {table_name}**")
                    
                    # Add sample data
                    if hasattr(table_info, 'sample_data') and not table_info.sample_data.empty:
                        sample_str = table_info.sample_data.head(3).to_string(index=False, max_rows=3)
                        context_parts.append(f"Sample Data:\n{sample_str}")
                    
                    context_parts.append("Columns:")
                    for col_name in columns:
                        if col_name in table_info.column_descriptions:
                            col_desc = table_info.column_descriptions[col_name]
                            # Get column type from schema
                            col_type = next((col['data_type'] for col in table_info.columns if col['column_name'] == col_name), 'unknown')
                            context_parts.append(f"  - {col_name} ({col_type}): {col_desc}")
            else:
                # Provide basic table information
                for table_name, table_info in self.table_descriptions.items():
                    context_parts.append(f"\n**Table: {table_name}**")
                    context_parts.append(f"Description: {table_info.overall_description}")
            
            context = "\n".join(context_parts) if context_parts else "No specific schema context provided"
            
            prompt = f"""Generate a PostgreSQL query based on the following natural language description and schema.

**Natural Language Description:**
{natural_language_sketch}

**Database Schema:**
{context}

Please provide a string of the SQL query.

Guidelines:
- Generate syntactically correct PostgreSQL SQL
- Use proper table and column names from the schema (use double quotes for identifiers if needed)
- Follow SQL best practices (proper joins, efficient WHERE clauses, etc.)
- Include comments for complex parts
- Ensure the query is executable as-is

Return ONLY the SQL string, no additional text or markdown.
"""
            
            system_prompt = """You are an expert PostgreSQL developer. Generate clean, efficient, and correct SQL queries.
Always use double quotes around table and column identifiers to handle case sensitivity and special characters."""
            
            # Query LLM
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt)
            ]
            response = self.client.chat_completion(messages).content
            
            # Parse response
            # json_start = response.find('{')
            # json_end = response.rfind('}') + 1
            # json_str = response[json_start:json_end]
            # parsed = json.loads(json_str)
            
            sql_query = response.strip()
            
            print("-----------------------------\n")
            print("Generated SQL Query:\n", sql_query)
            print("-----------------------------\n")
            
            # Clean up SQL query
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            result = sql_query
            
            logging.info(f"Generated SQL query: {sql_query[:100]}...")
            
            return ActionOutput(
                success=True,
                result=result,
                metadata={"query_length": len(sql_query)}
            )
            
        except Exception as e:
            logging.error(f"GenerateSQL failed: {e}")
            return ActionOutput(
                success=False,
                result=None,
                error=str(e)
            )


class ExecuteSQL(Action):
    """
    Action to execute a SQL query against the database.
    
    This action runs the SQL query and returns the results in a structured format.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize ExecuteSQL action.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__(
            name="ExecuteSQL",
            description="""Execute a SQL query against the database and return results.
            
            Use this action when you need to:
            - Execute a SQL query that has been generated
            - Retrieve data from the database
            - Verify query results
            
            The action returns the query results as a structured table."""
        )
        self.db = db_manager
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["sql_query"]
        }
    
    def execute(self, sql_query: str) -> ActionOutput:
        """
        Execute SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            ActionOutput with query results
        """
        try:
            # Execute query
            result = self.db.execute(sql_query).to_markdown()
            
            # Include preview for large results
            """if len(result) > 10:
                result_data["preview"] = result.to_dict_list()[:10]
                result_data["is_truncated"] = True
            else:
                result_data["is_truncated"] = False"""
            
            # logging.info(f"Query executed successfully. Returned {len(result)} rows in {result.execution_time:.3f}s")
            
            return ActionOutput(
                success=True,
                result=result,
                metadata={
                    "execution_time_ms": result.execution_time * 1000
                }
            )
            
        except Exception as e:
            logging.error(f"ExecuteSQL failed: {e}")
            return ActionOutput(
                success=False,
                result=None,
                error=str(e),
                metadata={"sql_query": sql_query[:200]}  # Include query snippet in error
            )


class GenerateAndExecuteSQL(Action):
    """
    Combined action to generate and execute a SQL query in one step.
    
    This action combines the functionality of GenerateSQL and ExecuteSQL,
    generating a SQL query from a natural language description and immediately
    executing it against the database.
    """
    
    def __init__(self, openai_client: AzureOpenAIClient, db_manager: DatabaseManager):
        """
        Initialize GenerateAndExecuteSQL action.
        
        Args:
            openai_client: AzureOpenAIClient for LLM queries
            db_manager: Database manager instance
        """
        super().__init__(
            name="GenerateAndExecuteSQL",
            description="""Generate a SQL query from natural language and execute it against the database in one step.
            
            Use this action when you need to:
            - Convert a natural language query into SQL and get results immediately
            - Skip the intermediate step of reviewing the generated SQL
            - Efficiently process straightforward queries
            
            The action returns both the generated SQL query and the execution results."""
        )
        
        self.table_descriptions: Dict[str, TableInfo] = {}
        
        self.available_tables = [
            "FundCashFlow",
            "FundInvestmentProperties",
            "FundInvestmentTimeProperties",
            "FundTimeProperties",
            "HandTransformedPortfolioInvestmentswithCFRaw"
        ]
        
        self._initialize_table_descriptions()
        
        self.client = openai_client
        self.db = db_manager
        
    def _initialize_table_descriptions(self):
        """Generate and cache descriptions for all available tables."""
        logging.info("Initializing table descriptions...")
        for table_name in self.available_tables:
            
            if os.path.exists(os.path.join(CACHE_DIR, f"{table_name.lower()}")):
                logging.info(f"Loading cached description for table: {table_name}")
                table_info = load_table_info(os.path.join(CACHE_DIR, f"{table_name.lower()}"))
                self.table_descriptions[table_name] = table_info
                continue
                
            try:
                describer = TableDescriber(table_name)
                table_info = describer.describe_table(format_description=False)
                self.table_descriptions[table_name] = table_info
                logging.info(f"Loaded description for table: {table_name}")
            except Exception as e:
                logging.error(f"Failed to describe table {table_name}: {e}")
                
                
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The full natural language query to generate SQL for and execute"
                },
                "selected_columns": {
                    "type": "object",
                    "description": "Optional dictionary of selected tables and columns to use in query generation",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, selected_columns: Optional[Dict[str, List[str]]] = None) -> ActionOutput:
        """
        Generate and execute SQL query.
        
        Args:
            query: The full natural language query
            selected_columns: Optional dict of {table_name: [column_names]}
            
        Returns:
            ActionOutput with generated SQL and execution results
        """
        try:
            # Step 1: Generate SQL (same as GenerateSQL)
            logging.info(f"Generating SQL for query: {query}")
            
            # Build schema context
            if selected_columns:
                context_parts = []
                for table_name, columns in selected_columns.items():
                    if table_name not in self.table_descriptions:
                        continue
                    table_info = self.table_descriptions[table_name]
                    context_parts.append(f"\n**Table: {table_name}**")
                    context_parts.append(f"Description: {table_info.overall_description}")
                    context_parts.append("Columns:")
                    for col_name in columns:
                        if col_name in table_info.column_descriptions:
                            col_desc = table_info.column_descriptions[col_name]
                            context_parts.append(f"  - {col_name}: {col_desc}")
                context = "\n".join(context_parts)
            else:
                context_parts = []
                for table_name, table_info in self.table_descriptions.items():
                    context_parts.append(f"\n**Table: {table_name}**")
                    context_parts.append(f"Description: {table_info.overall_description}")
                    context_parts.append("Columns:")
                    for col in table_info.columns:
                        col_name = col['column_name']
                        col_desc = table_info.column_descriptions.get(col_name, "No description available")
                        context_parts.append(f"  - {col_name}: {col_desc}")
                context = "\n".join(context_parts)
            
            # Create prompt for SQL generation
            prompt = f"""Given the following task and database schema, generate a complete, executable PostgreSQL query.

**Task Description:**
{query}

**Database Schema:**
{context}

**Instructions:**
- Generate a complete, executable PostgreSQL query that answers the task
- Use proper table and column names from the schema (use double quotes for identifiers if needed)
- Follow SQL best practices (proper joins, efficient WHERE clauses, etc.)
- Include comments for complex parts
- Ensure the query is executable as-is

Return ONLY the SQL string, no additional text or markdown.
"""
            
            system_prompt = """You are an expert PostgreSQL developer. Generate clean, efficient, and correct SQL queries.
Always use double quotes around table and column identifiers to handle case sensitivity and special characters."""
            
            # Query LLM
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt)
            ]
            response = self.client.chat_completion(messages).content
            
            # Clean up SQL query
            sql_query = response.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            logging.info(f"Generated SQL query: {sql_query[:100]}...")
            print("-----------------------------\n")
            print("Generated SQL Query:\n", sql_query)
            print("-----------------------------\n")
            
            # Step 2: Execute SQL (same as ExecuteSQL)
            logging.info("Executing generated SQL query...")
            result = self.db.execute(sql_query).to_markdown()
            
            logging.info(f"Query executed successfully")
            
            return ActionOutput(
                success=True,
                result=result,
                metadata={
                    "generated_sql": sql_query,
                    "query_length": len(sql_query),
                }
            )
            
        except Exception as e:
            logging.error(f"GenerateAndExecuteSQL failed: {e}")
            
            # Try to provide helpful error context
            error_metadata = {"query": query}
            if 'sql_query' in locals():
                error_metadata["generated_sql"] = sql_query[:200]
            
            return ActionOutput(
                success=False,
                result=None,
                error=str(e),
                metadata=error_metadata
            )
            
class Finish(Action):
    """
    Action to generate a natural language response from query results and complete the task.
    
    This action takes the correct result table from query execution and generates
    a short, natural language response to answer the original query.
    """
    
    def __init__(self, openai_client: AzureOpenAIClient):
        """
        Initialize Finish action.
        
        Args:
            openai_client: AzureOpenAIClient for LLM queries
        """
        super().__init__(
            name="Finish",
            description="""Generate a natural language response from query results and complete the task.
            
            Use this action when:
            - You have obtained the correct result table from executing the query
            - All necessary processing is complete
            - You are ready to present the final answer in natural language
            
            This action will generate a concise, natural language response based on the result table
            and terminate the reasoning loop."""
        )
        self.client = openai_client
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original natural language query"
                },
                "result_table": {
                    "type": "string",
                    "description": "The result table from query execution (as markdown or string format)"
                }
            },
            "required": ["query", "result_table"]
        }
    
    def execute(self, query: str, result_table: str) -> ActionOutput:
        """
        Generate natural language response from query results.
        
        Args:
            query: The original natural language query
            result_table: The result table from query execution
            
        Returns:
            ActionOutput with the natural language response
        """
        try:
            logging.info(f"Generating natural language response for query: {query[:100]}...")
            
            # Create prompt for natural language generation
            prompt = f"""Given the original query and its result table, generate a concise natural language response that directly answers the query.

**Original Query:**
{query}

**Result Table:**
{result_table}

Please provide a short, clear natural language response that:
1. Directly answers the query based on the result table
2. Highlights the key findings or insights
3. Is concise (2-4 sentences maximum)
4. Uses specific numbers/values from the result table when relevant

Return ONLY the natural language response, no additional formatting or explanation.
"""
            
            system_prompt = """You are an expert at interpreting database query results and communicating findings clearly.
Generate concise, accurate natural language responses that directly answer the user's question based on the result table."""
            
            # Query LLM for natural language response
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt)
            ]
            nl_response = self.client.chat_completion(messages).content
            
            logging.info(f"Generated natural language response: {nl_response[:100]}...")
            
            return ActionOutput(
                success=True,
                result={"natural_language_response": nl_response.strip(), "result_table": result_table},
                metadata={
                    "action": "finish",
                    "query": query,
                    "response_length": len(nl_response)
                }
            )
            
        except Exception as e:
            logging.error(f"Finish action failed: {e}")
            return ActionOutput(
                success=False,
                result=None,
                error=str(e)
            )