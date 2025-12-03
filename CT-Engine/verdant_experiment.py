"""
Verdant Experiment - Batch Query Processing with Async

This script demonstrates how to:
1. Load queries from a CSV file
2. Process queries asynchronously in parallel
3. Save results to a CSV file
4. Track progress and performance metrics
"""

import os
import asyncio
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from dotenv import load_dotenv

from semantic_parser import (
    ReACTEngine,
    AzureOpenAIClient
)
from semantic_parser.modules.verdant import create_action_registry, DatabaseManager, OpenAIUtils
from semantic_parser.src.react import ReasoningTrace


@dataclass
class ExperimentResult:
    """Data class to store results of each query experiment"""
    query_id: int
    question: str
    ground_truth_answer: Optional[str]
    previous_ct_answer: Optional[str]
    success: bool
    generated_sql: Optional[str]
    natural_language_response: Optional[str]
    result_table: Optional[str]
    final_output: Optional[str]
    total_steps: int
    duration_seconds: float
    error_message: Optional[str]
    timestamp: str
    metadata: Dict[str, Any]


class VerdantExperiment:
    """
    Async batch experiment runner for Verdant semantic parser.
    
    Features:
    - Load queries from CSV
    - Process queries asynchronously
    - Track progress and metrics
    - Save detailed results
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        max_steps: int = 10,
        verbose: bool = False,
        results_dir: str = "./experiment_results"
    ):
        """
        Initialize the experiment runner.
        
        Args:
            max_concurrent: Maximum number of concurrent query processing
            max_steps: Maximum steps per query in ReACT loop
            verbose: Whether to print detailed logs
            results_dir: Directory to save experiment results
        """
        self.max_concurrent = max_concurrent
        self.max_steps = max_steps
        self.verbose = verbose
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize engines (one per concurrent task to avoid conflicts)
        self.engines: List[ReACTEngine] = []
        
        # Initialize OpenAI client for actions
        self.client = OpenAIUtils(api_key=os.getenv("AZURE_OPENAI_API_KEY"))
        
        # Initialize database manager
        self.db_manager = DatabaseManager(
            visible_tables=[
                "FundCashFlow",
                "FundInvestmentProperties",
                "FundInvestmentTimeProperties",
                "FundTimeProperties",
                "HandTransformedPortfolioInvestmentswithCFRaw"
            ]
        )
        
        # Load environment variables
        load_dotenv()
        
    def _setup_engine(self) -> ReACTEngine:
        """
        Set up a ReACT engine instance.
        
        Returns:
            Configured ReACTEngine instance
        """
        # Initialize the LLM client
        llm_client = AzureOpenAIClient(
            azure_endpoint="https://ovalnairr.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            deployment_name="o3",
        )
        
        # Use the module factory to create a configured ActionRegistry
        # This automatically sets up all actions, target format, and predecided actions
        action_registry = create_action_registry(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Create engine - no need to specify target_format or predecided_actions
        # They are provided by the module configuration
        engine = ReACTEngine(
            llm_client=llm_client,
            action_registry=action_registry,
            max_steps=self.max_steps,
            verbose=self.verbose
        )
        
        return engine
    
    def _disambiguate_query(self, query: str) -> str:
        """
        Disambiguate the user query if needed.
        
        Args:
            query: Original user query
        Returns:

            Disambiguated query
        """
        
        
        
        entity_labels = [r[0] for r in self.db_manager.execute("""
           SELECT DISTINCT "fund" FROM "FundCashFlow";
           """).rows]
        # entity_labels = [r[0] for r in entity_labels]
        disambiguate_prompt = f"""The following SQL query may contain ambiguous references to entities. Given the list of known entity labels, rewrite the query to clarify any ambiguous references.
Known entity labels:
{', '.join(f"'{label}'" for label in entity_labels)}
Query:
{query}
If the entity mentioned matches the database label, return it unchanged. Otherwise, replace ambiguous references with specific entity label (exact) from the known list.
"""
        system_prompt = "You are an expert at clarifying ambiguous queries based on known entities."
        response = self.client._query_azure_openai(disambiguate_prompt, system_prompt)
        return response.strip()
    
    
    def load_queries_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load queries from a CSV file.
        
        Expected CSV format:
        - Question: The natural language query text (REQUIRED)
        - Answer: Ground truth answer for reference (OPTIONAL)
        - CT: Previous CT (Computational Thinking) answer for reference (OPTIONAL)
        - query_id: Unique identifier for each query (OPTIONAL, will auto-generate if missing)
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with queries
        """
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        if 'Question' not in df.columns:
            raise ValueError("CSV must contain a 'Question' column")
        else:
            print("Disambiguating the query...")
            df['Question'] = df['Question'].apply(self._disambiguate_query)
        
        # Add query_id if not present
        if 'query_id' not in df.columns:
            df['query_id'] = range(1, len(df) + 1)
        
        # Ensure optional columns exist (fill with None if missing)
        if 'Answer' not in df.columns:
            df['Answer'] = None
        
        if 'CT' not in df.columns:
            df['CT'] = None
        
        print(f"Loaded {len(df)} queries from {csv_path}")
        print(f"Columns: {', '.join(df.columns)}")
        return df
    
    async def process_single_query(
        self,
        query_id: int,
        question: str,
        ground_truth_answer: Optional[str],
        previous_ct_answer: Optional[str],
        semaphore: asyncio.Semaphore
    ) -> ExperimentResult:
        """
        Process a single query asynchronously.
        
        Args:
            query_id: Unique identifier for the query
            question: Natural language question
            ground_truth_answer: Ground truth answer for reference
            previous_ct_answer: Previous CT answer for reference
            semaphore: Semaphore to limit concurrent executions
            
        Returns:
            ExperimentResult with processing details
        """
        async with semaphore:
            start_time = datetime.now()
            
            if self.verbose:
                print(f"\n[Query {query_id}] Starting: {question[:50]}...")
            
            try:
                # Create a dedicated engine for this task
                engine = self._setup_engine()
                
                # Run the parsing in a thread pool to avoid blocking
                # Always use SQL as target format
                loop = asyncio.get_event_loop()
                trace = await loop.run_in_executor(
                    None,
                    lambda: engine.parse(
                        query=question,
                        target_format="SQL",  # Always SQL
                        initial_context={"query_id": query_id}
                    )
                )
                
                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds()
                
                # Extract generated SQL, natural language response, and result table
                generated_sql = None
                natural_language_response = None
                result_table = None
                
                # Extract SQL from steps (look for GenerateAndExecuteSQL action)
                for step in trace.steps:
                    if step.action.action_name == "GenerateAndExecuteSQL":
                        if step.observation and step.observation.success:
                            # SQL is stored in observation metadata
                            if "generated_sql" in step.observation.metadata:
                                generated_sql = step.observation.metadata["generated_sql"]
                        break
                
                # Extract natural language response and result table from final output
                if trace.final_output:
                    # Try to parse if it's a JSON string or dict
                    try:
                        if isinstance(trace.final_output, str):
                            if trace.final_output.startswith('{'):
                                output_dict = json.loads(trace.final_output)
                                natural_language_response = output_dict.get('natural_language_response')
                                result_table = output_dict.get('result_table')
                            else:
                                natural_language_response = trace.final_output
                        elif isinstance(trace.final_output, dict):
                            natural_language_response = trace.final_output.get('natural_language_response')
                            result_table = trace.final_output.get('result_table')
                        else:
                            natural_language_response = str(trace.final_output)
                    except json.JSONDecodeError:
                        natural_language_response = trace.final_output
                
                # Create result object
                result = ExperimentResult(
                    query_id=query_id,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    previous_ct_answer=previous_ct_answer,
                    success=trace.is_complete,
                    generated_sql=generated_sql,
                    natural_language_response=natural_language_response,
                    result_table=result_table,
                    final_output=str(trace.final_output) if trace.final_output else None,
                    total_steps=trace.get_step_count(),
                    duration_seconds=duration,
                    error_message=None,
                    timestamp=datetime.now().isoformat(),
                    metadata=trace.metadata
                )
                
                if self.verbose:
                    print(f"[Query {query_id}] ✓ Completed in {duration:.2f}s")
                    if generated_sql:
                        print(f"[Query {query_id}]   SQL: {generated_sql[:100]}...")
                    if natural_language_response:
                        print(f"[Query {query_id}]   Response: {natural_language_response[:100]}...")
                
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                
                if self.verbose:
                    print(f"[Query {query_id}] ✗ Error: {str(e)}")
                
                # Create error result
                result = ExperimentResult(
                    query_id=query_id,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    previous_ct_answer=previous_ct_answer,
                    success=False,
                    generated_sql=None,
                    natural_language_response=None,
                    result_table=None,
                    final_output=None,
                    total_steps=0,
                    duration_seconds=duration,
                    error_message=str(e),
                    timestamp=datetime.now().isoformat(),
                    metadata={}
                )
                
                return result
    
    async def run_experiment(
        self,
        queries_df: pd.DataFrame,
        experiment_name: str = "experiment"
    ) -> List[ExperimentResult]:
        """
        Run the experiment on all queries asynchronously.
        
        Args:
            queries_df: DataFrame with queries (must have 'Question' column)
            experiment_name: Name for this experiment (used in output files)
            
        Returns:
            List of ExperimentResult objects
        """
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {experiment_name}")
        print(f"Total Queries: {len(queries_df)}")
        print(f"Max Concurrent: {self.max_concurrent}")
        print(f"{'='*80}\n")
        
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create tasks for all queries
        tasks = []
        for _, row in queries_df.iterrows():
            task = self.process_single_query(
                query_id=row['query_id'],
                question=row['Question'],
                ground_truth_answer=row.get('Answer'),
                previous_ct_answer=row.get('CT'),
                semaphore=semaphore
            )
            tasks.append(task)
        
        # Run all tasks concurrently
        experiment_start = datetime.now()
        results = await asyncio.gather(*tasks)
        experiment_duration = (datetime.now() - experiment_start).total_seconds()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Experiment Complete: {experiment_name}")
        print(f"{'='*80}")
        print(f"Total Duration: {experiment_duration:.2f} seconds")
        print(f"Total Queries: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r.success)}")
        print(f"Failed: {sum(1 for r in results if not r.success)}")
        print(f"Average Duration: {sum(r.duration_seconds for r in results) / len(results):.2f}s")
        print(f"{'='*80}\n")
        
        return results
    
    def save_results(
        self,
        results: List[ExperimentResult],
        experiment_name: str = "experiment"
    ):
        """
        Save experiment results to CSV and JSON files.
        
        Args:
            results: List of ExperimentResult objects
            experiment_name: Name for this experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        results_data = [asdict(r) for r in results]
        df_results = pd.DataFrame(results_data)
        
        # Save full results to CSV (excluding complex metadata)
        csv_full_path = self.results_dir / f"{experiment_name}_{timestamp}_results_full.csv"
        df_results.drop(columns=['metadata'], errors='ignore').to_csv(csv_full_path, index=False)
        print(f"Full results saved to: {csv_full_path}")
        
        # Create a simplified CSV with key columns (question, ground truth, CT, SQL, NL response)
        csv_simple_path = self.results_dir / f"{experiment_name}_{timestamp}_results.csv"
        df_simple = df_results[[
            'query_id', 
            'question', 
            'ground_truth_answer', 
            'previous_ct_answer',
            'generated_sql', 
            'natural_language_response', 
            'success', 
            'duration_seconds'
        ]]
        df_simple.to_csv(csv_simple_path, index=False)
        print(f"Simplified results (question + references + SQL + response) saved to: {csv_simple_path}")
        
        # Save full results with metadata to JSON
        json_path = self.results_dir / f"{experiment_name}_{timestamp}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"Full results with metadata saved to: {json_path}")
        
        # Create individual files for each query (SQL and NL response)
        queries_dir = self.results_dir / f"{experiment_name}_{timestamp}_queries"
        queries_dir.mkdir(exist_ok=True)
        
        for result in results:
            query_dir = queries_dir / f"query_{result.query_id}"
            query_dir.mkdir(exist_ok=True)
            
            # Save question text
            with open(query_dir / "question.txt", 'w') as f:
                f.write(result.question)
            
            # Save ground truth answer (if available)
            if result.ground_truth_answer:
                with open(query_dir / "ground_truth_answer.txt", 'w') as f:
                    f.write(str(result.ground_truth_answer))
            
            # Save previous CT answer (if available)
            if result.previous_ct_answer:
                with open(query_dir / "previous_ct_answer.txt", 'w') as f:
                    f.write(str(result.previous_ct_answer))
            
            # Save generated SQL
            if result.generated_sql:
                with open(query_dir / "generated.sql", 'w') as f:
                    f.write(result.generated_sql)
            
            # Save natural language response
            if result.natural_language_response:
                with open(query_dir / "response.txt", 'w') as f:
                    f.write(result.natural_language_response)
            
            # Save result table if available
            if result.result_table:
                with open(query_dir / "result_table.txt", 'w') as f:
                    f.write(result.result_table)
        
        print(f"Individual query files saved to: {queries_dir}")
        
        # Create summary statistics
        summary = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "total_queries": len(results),
            "successful_queries": sum(1 for r in results if r.success),
            "failed_queries": sum(1 for r in results if not r.success),
            "queries_with_sql": sum(1 for r in results if r.generated_sql),
            "queries_with_nl_response": sum(1 for r in results if r.natural_language_response),
            "queries_with_ground_truth": sum(1 for r in results if r.ground_truth_answer),
            "queries_with_ct_answer": sum(1 for r in results if r.previous_ct_answer),
            "average_duration_seconds": sum(r.duration_seconds for r in results) / len(results),
            "total_duration_seconds": sum(r.duration_seconds for r in results),
            "average_steps": sum(r.total_steps for r in results if r.total_steps > 0) / max(sum(1 for r in results if r.total_steps > 0), 1),
        }
        
        summary_path = self.results_dir / f"{experiment_name}_{timestamp}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
        
        print(f"\n{'='*80}")
        print("SAVED FILES SUMMARY:")
        print(f"{'='*80}")
        print(f"1. Simplified CSV (question + references + SQL + response): {csv_simple_path.name}")
        print(f"2. Full CSV (all columns): {csv_full_path.name}")
        print(f"3. JSON (complete data): {json_path.name}")
        print(f"4. Individual query files: {queries_dir.name}/")
        print(f"   - question.txt")
        print(f"   - ground_truth_answer.txt (if available)")
        print(f"   - previous_ct_answer.txt (if available)")
        print(f"   - generated.sql")
        print(f"   - response.txt")
        print(f"   - result_table.txt")
        print(f"5. Summary statistics: {summary_path.name}")
        print(f"{'='*80}\n")
        
        return csv_simple_path, csv_full_path, json_path, queries_dir, summary_path


# ============================================================================
# Example Usage Functions
# ============================================================================

async def run_from_csv(csv_path: str, experiment_name: str = "verdant_experiment"):
    """
    Run experiment from a CSV file.
    
    Args:
        csv_path: Path to CSV file with queries
        experiment_name: Name for this experiment
    """
    # Initialize experiment runner
    experiment = VerdantExperiment(
        max_concurrent=25,  # Process 5 queries at a time
        max_steps=8,
        verbose=True,
        results_dir="./experiment_results"
    )
    
    # Load queries from CSV
    queries_df = experiment.load_queries_from_csv(csv_path)
    
    # Run the experiment
    results = await experiment.run_experiment(queries_df, experiment_name)
    
    # Save results
    experiment.save_results(results, experiment_name)
    
    return results


async def run_sample_experiment():
    """
    Run a sample experiment with hardcoded queries.
    """
    # Create sample queries DataFrame
    sample_queries = pd.DataFrame([
        {
            "query_id": 1,
            "Question": "On a year-on-year basis (4Q24 vs 4Q23), which funds saw an uplift to overall MOIC in 2024?",
            "Answer": "Fund A (1.45 to 1.82), Fund B (2.10 to 2.35), Fund C (1.78 to 1.95)",
            "CT": "Previous computational thinking response placeholder"
        },
        {
            "query_id": 2,
            "Question": "What is the average MOIC for all funds in 2024?",
            "Answer": "1.85",
            "CT": None
        },
        {
            "query_id": 3,
            "Question": "Show the top 5 funds by total invested capital.",
            "Answer": "Fund A ($100M), Fund B ($85M), Fund C ($70M), Fund D ($65M), Fund E ($60M)",
            "CT": None
        },
        {
            "query_id": 4,
            "Question": "Which funds have MOIC greater than 2.0 in Q4 2024?",
            "Answer": "Fund B (2.35), Fund F (2.15), Fund G (2.10)",
            "CT": None
        },
        {
            "query_id": 5,
            "Question": "Compare the MOIC trends for Fund A and Fund B over the past year.",
            "Answer": "Fund A: 1.45 -> 1.82 (+25.5%), Fund B: 2.10 -> 2.35 (+11.9%)",
            "CT": None
        }
    ])
    
    # Save sample queries to CSV
    sample_csv = "/home/jiuding/computational_thinking/verdant_revised.csv"
    
    # Run experiment
    results = await run_from_csv(sample_csv, experiment_name="sample_experiment")
    
    return results


def create_sample_csv(output_path: str = "./sample_queries.csv"):
    """
    Create a sample CSV file with example queries.
    
    Args:
        output_path: Path where to save the CSV file
    """
    sample_queries = pd.DataFrame([
        {
            "query_id": 1,
            "Question": "On a year-on-year basis (4Q24 vs 4Q23), which funds saw an uplift to overall MOIC in 2024?",
            "Answer": "Fund A, Fund B, Fund C",
            "CT": "Previous CT answer for query 1"
        },
        {
            "query_id": 2,
            "Question": "What is the average MOIC for all funds in 2024?",
            "Answer": "1.85",
            "CT": None
        },
        {
            "query_id": 3,
            "Question": "Show the top 5 funds by total invested capital.",
            "Answer": None,
            "CT": None
        },
        {
            "query_id": 4,
            "Question": "Which funds have MOIC greater than 2.0 in Q4 2024?",
            "Answer": "Fund B, Fund F, Fund G",
            "CT": None
        },
        {
            "query_id": 5,
            "Question": "Compare the MOIC trends for Fund A and Fund B over the past year.",
            "Answer": None,
            "CT": "Previous CT comparison analysis"
        },
        {
            "query_id": 6,
            "Question": "What was the total cash flow for all funds in Q3 2024?",
            "Answer": "$25M",
            "CT": None
        },
        {
            "query_id": 7,
            "Question": "List all funds with negative MOIC change from 2023 to 2024.",
            "Answer": None,
            "CT": None
        },
        {
            "query_id": 8,
            "Question": "Show the distribution of fund types in the portfolio.",
            "Answer": None,
            "CT": None
        }
    ])
    
    sample_queries.to_csv(output_path, index=False)
    print(f"Sample CSV created at: {output_path}")
    print(f"Contains {len(sample_queries)} sample queries")
    print(f"Format: Question, Answer, CT columns")
    return output_path


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("VERDANT EXPERIMENT - Async Batch Query Processing")
    print("="*80)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Run with provided CSV file
        csv_path = sys.argv[1]
        experiment_name = sys.argv[2] if len(sys.argv) > 2 else "verdant_experiment"
        
        print(f"\nRunning experiment with CSV: {csv_path}")
        print(f"Experiment name: {experiment_name}\n")
        
        # Run async experiment
        asyncio.run(run_from_csv(csv_path, experiment_name))
        
    else:
        # No CSV provided - create sample and run with it
        print("\nNo CSV file provided. Running sample experiment...")
        print("Usage: python verdant_experiment.py <csv_path> [experiment_name]\n")
        
        # Create sample CSV
        sample_csv = "/home/jiuding/computational_thinking/verdant_revised.csv"
        
        # Ask user if they want to proceed
        response = input("\nRun sample experiment with 8 queries? (y/n): ").lower()
        if response == 'y':
            print("\nRunning sample experiment...\n")
            asyncio.run(run_sample_experiment())
        else:
            print("\nExperiment cancelled.")
            print(f"Sample queries saved to: {sample_csv}")
            print("\nTo run later, use:")
            print(f"  python verdant_experiment.py {sample_csv}")