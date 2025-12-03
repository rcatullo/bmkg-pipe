"""
Database utility module for managing queries and schema information.
Provides a clean interface for database operations using SUQL PostgreSQL connection.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from suql.postgresql_connection import execute_sql
import pandas as pd


@dataclass
class QueryResult:
    """Structured representation of query results."""
    rows: List[Tuple]
    columns: List[str]
    execution_time: float
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert rows to list of dictionaries with column names as keys."""
        return [dict(zip(self.columns, row)) for row in self.rows]
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary with columns as keys and values as lists."""
        return {col: [row[i] for row in self.rows] for i, col in enumerate(self.columns)}
    
    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.rows)
    
    def __iter__(self):
        """Iterate over rows as dictionaries."""
        return iter(self.to_dict_list())
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame(self.rows, columns=self.columns)
    
    def to_markdown(self) -> str:
        """Convert results to a markdown table string."""
        df = self.to_dataframe()
        return df.to_markdown(index=False)


class DatabaseManager:
    """Manager for database queries and schema operations."""
    
    def __init__(self, visible_tables: Optional[List[str]] = None):
        self._schema_cache: Optional[Dict[str, Any]] = None
        self.visible_tables = visible_tables
        self.temp_tables = set()

    def execute(self, sql_query: str, commit_in_lieu_fetch: bool=False) -> QueryResult:
        """
        Execute a SQL query and return structured results.
        
        Args:
            sql_query: SQL query string to execute
            
        Returns:
            QueryResult object with rows, columns, and execution time
        """
        rows, columns, exec_time = execute_sql(
            user="creator_role",
            password="creator_role",
            database='verdant_db_new',
            sql_query=sql_query,
            commit_in_lieu_fetch=commit_in_lieu_fetch,
        )
        return QueryResult(rows=rows, columns=columns, execution_time=exec_time)

    def execute_raw(self, sql_query: str, commit_in_lieu_fetch: bool=False) -> Tuple[List[Tuple], List[str], float]:
        """
        Execute a SQL query and return raw results.
        
        Args:
            sql_query: SQL query string to execute
            
        Returns:
            Tuple of (rows, columns, execution_time)
        """
        return execute_sql(
            user="creator_role",
            password="creator_role",
            database='verdant_db_new',
            sql_query=sql_query,
            commit_in_lieu_fetch=commit_in_lieu_fetch,
        )
        
    def return_all_table_names(self) -> List[str]:
        return self.execute_raw("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE';
        """)[0]
        
    def create_temp_table(self, table_name: str, create_sql: str):
        """
        Create a temporary table in the database.
        
        Args:
            table_name: Name of the temporary table
            create_sql: SQL statement to create the table
        """
        create_sql = f"CREATE TABLE \"{table_name}\" AS {create_sql}"
        self.execute(create_sql, commit_in_lieu_fetch=True)
        self.temp_tables.add(table_name)
        
    def drop_temp_tables(self):
        """Drop all temporary tables created during the session."""
        for table in self.temp_tables:
            self.execute(f'DROP TABLE IF EXISTS "{table}";')
        self.temp_tables.clear()
    
    def get_table_schema(self, table_name: str, schema: str = 'public') -> List[Dict[str, Any]]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            schema: Database schema name (default: 'public')
            
        Returns:
            List of dictionaries containing column information
        """
        query = f"""
            SELECT 
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = '{schema}' 
            AND table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        result = self.execute(query)
        return result.to_dict_list()
    
    def get_column_names(self, table_name: str, schema: str = 'public') -> List[str]:
        """
        Get column names for a specific table.
        
        Args:
            table_name: Name of the table
            schema: Database schema name (default: 'public')
        Returns:

            List of column names
        """
        schema_info = self.get_table_schema(table_name, schema)
        return [col['column_name'] for col in schema_info]
    
    @property
    def temp_table_names(self) -> List[str]:
        """Get a list of temporary table names created during the session."""
        return list(self.temp_tables)
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table
            schema: Database schema name (default: 'public')
            
        Returns:
            True if table exists, False otherwise
        """
        query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            );
        """
        result = self.execute(query)
        return result.rows[0][0] if result.rows else False
    
    def get_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            schema: Database schema name (default: 'public')
            
        Returns:
            Number of rows in the table
        """
        query = f'SELECT COUNT(*) FROM \"{table_name}\";'
        result = self.execute(query)
        return result.rows[0][0] if result.rows else 0
    
    def get_unique_values(self, table_name: str, column_name: str, limit: int = None) -> List[Any]:
        """
        Get unique values from a specific column in a table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            limit: Maximum number of unique values to return (default: 10)
        Returns:
            List of unique values
        """
        limit_clause = f'LIMIT {limit}' if limit is not None else ''
        query = f'SELECT DISTINCT "{column_name}" FROM \"{table_name}\" {limit_clause};'
        result = self.execute(query)
        return set([row[0] for row in result.rows])