from semantic_parser.modules.verdant.db_utils import *
from semantic_parser.llm_client import AzureOpenAIClient, Message
import os
import json


@dataclass
class TableInfo:
    name: str
    overall_description: str
    row_count: int
    columns: List[Dict[str, Any]]
    sample_data: pd.DataFrame
    column_descriptions: Dict[str, str]
    column_format_description: Dict[str, str]
    
    
def save_table_info(table_info: TableInfo, directory: str):
    """
    Save table information to a specified directory.
    
    Args:
        table_info: TableInfo object containing table metadata and descriptions
        directory: Directory path to save the information
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save overall description
    with open(os.path.join(directory, f"{table_info.name.lower()}_description.txt"), "w") as f:
        f.write(table_info.overall_description)
    
    # Save column descriptions
    with open(os.path.join(directory, f"{table_info.name.lower()}_column_descriptions.json"), "w") as f:
        json.dump(table_info.column_descriptions, f, indent=4)
    
    # Save column format descriptions
    with open(os.path.join(directory, f"{table_info.name.lower()}_column_format_descriptions.json"), "w") as f:
        json.dump(table_info.column_format_description, f, indent=4)
    
    # Save sample data
    table_info.sample_data.to_csv(os.path.join(directory, f"{table_info.name.lower()}_sample_data.csv"), index=False)
    # Save schema
    with open(os.path.join(directory, f"{table_info.name.lower()}_schema.json"), "w") as f:
        json.dump(table_info.columns, f, indent=4)
        

def load_table_info(directory: str) -> TableInfo:
    """
    Load table information from a specified directory.
    
    Args:
        directory: Directory path where the information is saved

    Returns:
        TableInfo object loaded from the directory
    """
    name = os.path.basename(directory)
    with open(os.path.join(directory, f"{name}_description.txt"), "r") as f:
        overall_description = f.read()
    with open(os.path.join(directory, f"{name}_column_descriptions.json"), "r") as f:
        column_descriptions = json.load(f)
    with open(os.path.join(directory, f"{name}_column_format_descriptions.json"), "r") as f:
        column_format_description = json.load(f)
    sample_data = pd.read_csv(os.path.join(directory, f"{name}_sample_data.csv"))
    with open(os.path.join(directory, f"{name}_schema.json"), "r") as f:
        columns = json.load(f)

    return TableInfo(
        name=name,
        overall_description=overall_description,
        row_count=len(sample_data),
        columns=columns,
        sample_data=sample_data,
        column_descriptions=column_descriptions,
        column_format_description=column_format_description
    )


class TableDescriber:
    
    def __init__(self, table_name: str, deployment_name: str = "o3"):
        self.db = DatabaseManager()
        self.client = AzureOpenAIClient(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=deployment_name,
        )
        
        if not self.db.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist in the database.")
        self.table_name = table_name

    def _generate_column_format_description(self, column_name: str, column_description: str) -> str:
        """
        Generate a detailed format description for a column based on its unique values.
        
        Args:
            column_name: Name of the column
            column_description: Generated description of the column
            
        Returns:
            Detailed format/content description of the column
        """
        # Get unique values from the column (limit to 10 for conciseness)
        unique_values = self.db.get_unique_values(self.table_name, column_name, limit=10)
        
        # Convert set to sorted list for consistent output
        unique_values_list = sorted(list(unique_values), key=str)
        
        prompt = f"""Given a database column with the following information:

Column Name: {column_name}
Column Description: {column_description}
Sample Unique Values: {unique_values_list}

Please provide a concise format description that explains:
1. The data format/pattern (e.g., date format, currency, code pattern)
2. The type of values stored (e.g., categorical, numerical ranges, text patterns)
3. Any specific conventions or standards used

Keep the description to 1-2 sentences and focus on the format and content structure.
"""
        
        system_prompt = "You are a database schema expert specializing in financial data. Provide clear, concise descriptions of data formats."
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt)
        ]
        return self.client.chat_completion(messages).content

    def _generate_overall_description(self, column_descriptions: Dict[str, str]) -> str:
        """
        Generate an overall description of the table based on column descriptions.
        
        Args:
            column_descriptions: Dictionary mapping column names to their descriptions
            
        Returns:
            Overall table description
        """
        columns_info = "\n".join([f"- {col}: {desc}" for col, desc in column_descriptions.items()])
        
        prompt = f"""Given a financial database table named '{self.table_name}' with the following columns:

{columns_info}

Please provide a concise overall description of this table in 2-3 sentences. Explain:
1. What the table represents
2. Its primary purpose in a financial/investment context
3. The type of data it contains

Focus on the business meaning and use case of this table.
"""
        
        system_prompt = "You are a financial database expert. Provide clear, business-focused descriptions of database tables in investment banking contexts."
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt)
        ]
        return self.client.chat_completion(messages).content
    
    def describe_table(self, format_description: bool = False, included_columns: List[str] = None) -> TableInfo:
        """
        Generate comprehensive table information including schema and descriptions.
        
        Args:
            format_description: If True, generate detailed format descriptions for each column
            included_columns: List of column names to include. If None, all columns are included.
            
        Returns:
            TableInfo object with complete table metadata and descriptions
        """
        # Get basic table information
        columns = self.db.get_table_schema(self.table_name)
        row_count = self.db.get_row_count(self.table_name)
        
        # Filter columns if included_columns is specified
        if included_columns is not None:
            # Validate that all included columns exist
            all_column_names = {col['column_name'] for col in columns}
            invalid_columns = set(included_columns) - all_column_names
            if invalid_columns:
                raise ValueError(f"The following columns do not exist in table {self.table_name}: {invalid_columns}")
            
            # Filter to only included columns
            columns = [col for col in columns if col['column_name'] in included_columns]
            
            # Build SELECT statement with only included columns
            column_names_str = ', '.join([f'"{col}"' for col in included_columns])
            sample_query = f'SELECT {column_names_str} FROM "{self.table_name}" LIMIT 5;'
        else:
            # Get all columns
            sample_query = f'SELECT * FROM "{self.table_name}" LIMIT 5;'
        
        # Get sample data (5 rows)
        sample_result = self.db.execute(sample_query)
        sample_data = sample_result.to_dataframe()
        
        # Prepare schema information for LLM prompt
        schema_info = []
        for col in columns:
            col_info = f"- {col['column_name']} ({col['data_type']})"
            if col['is_nullable'] == 'NO':
                col_info += " [NOT NULL]"
            schema_info.append(col_info)
        
        schema_str = "\n".join(schema_info)
        
        # Convert sample data to a readable format
        sample_str = sample_data.to_string(index=False, max_rows=5)
        
        # Prompt LLM for column descriptions
        prompt = f"""You are analyzing a financial database table from an investment bank. The table is named '{self.table_name}'.

**Table Schema:**
{schema_str}

**Sample Data (first 5 rows):**
{sample_str}

Please provide descriptions for this table in JSON format with the following structure:
{{
    "column_descriptions": {{
        "column_name_1": "description of what this column represents",
        "column_name_2": "description of what this column represents",
        ...
    }}
}}

Guidelines:
- Provide a clear, concise description for EACH column (1 sentence)
- Focus on what the data represents in a financial/investment context
- Consider the column name, data type, and sample values
- Be specific about financial terminology (e.g., fund ID, ticker symbol, asset class, etc.)

Return ONLY the JSON object, no additional text.
"""
        
        system_prompt = """You are a financial database expert with deep knowledge of investment banking, fund management, and financial data structures. 
Provide accurate, professional descriptions of database columns in a financial context."""
        
        # Get LLM response
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt)
        ]
        llm_response = self.client.chat_completion(messages).content
        
        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            json_str = llm_response[json_start:json_end]
            
            parsed_response = json.loads(json_str)
            column_descriptions = parsed_response.get("column_descriptions", {})
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: create basic descriptions
            column_descriptions = {col['column_name']: f"{col['column_name']} ({col['data_type']})" 
                                 for col in columns}
        
        # Generate overall table description
        overall_description = self._generate_overall_description(column_descriptions)
        
        # Generate format descriptions if requested
        column_format_descriptions = {}
        if format_description:
            for col_name in column_descriptions.keys():
                try:
                    format_desc = self._generate_column_format_description(
                        col_name, 
                        column_descriptions[col_name]
                    )
                    column_format_descriptions[col_name] = format_desc
                except Exception as e:
                    logging.error(f"Failed to generate format description for {col_name}: {e}")
                    column_format_descriptions[col_name] = "Format description unavailable"
        
        # Create and return TableInfo object
        return TableInfo(
            name=self.table_name,
            overall_description=overall_description,
            row_count=row_count,
            columns=columns,
            sample_data=sample_data,
            column_descriptions=column_descriptions,
            column_format_description=column_format_descriptions
        )