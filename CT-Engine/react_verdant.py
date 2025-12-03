"""
Example Usage of the Semantic Parser Framework

This script demonstrates how to:
1. Set up the framework with custom actions
2. Register actions with the ActionRegistry
3. Run the ReACT engine to parse natural language queries
4. Handle the results
"""

import os
from dotenv import load_dotenv

from semantic_parser import (
    ReACTEngine,
    ActionRegistry,
    AzureOpenAIClient
)
from semantic_parser.modules.verdant.actions import (
    FetchRelatedColumns,
    FetchBackgroundKnowledge,
    SketchSQL,
    GenerateSQL,
    ExecuteSQL,
    GenerateAndExecuteSQL,
    Finish
)

from semantic_parser.reasoning_step import ReasoningStep

from semantic_parser.modules.verdant.db_utils import (
    DatabaseManager
)
from semantic_parser.modules.verdant.openai_utils import OpenAIUtils


def setup_framework():
    """
    Set up the semantic parser framework with all components.
    
    Returns:
        Configured ReACTEngine instance
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize the LLM client
    llm_client = AzureOpenAIClient(
        azure_endpoint="https://ovalnairr.openai.azure.com/",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        deployment_name="o3",  # Your o3 deployment name
    )
    
    
    client = OpenAIUtils(api_key=os.getenv("AZURE_OPENAI_API_KEY"))
    db_manager = DatabaseManager(
        visible_tables=[
            "FundCashFlow",
            "FundInvestmentProperties",
            "FundInvestmentTimeProperties",
            "FundTimeProperties",
            "HandTransformedPortfolioInvestmentswithCFRaw"
        ]
    )
    
    
    # Create the action registry
    action_registry = ActionRegistry()
    
    # Register built-in and example actions
    action_registry.register(FetchRelatedColumns(openai_client=client))
    action_registry.register(FetchBackgroundKnowledge(openai_client=client))
    action_registry.register(SketchSQL(openai_client=client))
    # action_registry.register(GenerateSQL(openai_client=client))
    # action_registry.register(ExecuteSQL(db_manager=db_manager))
    action_registry.register(GenerateAndExecuteSQL(openai_client=client, db_manager=db_manager))
    action_registry.register(Finish(openai_client=client))  # Updated to include openai_client
        
    # Create the ReACT engine with predecided actions
    engine = ReACTEngine(
        llm_client=llm_client,
        action_registry=action_registry,
        max_steps=10,
        verbose=True,
        predecided_actions=["FetchBackgroundKnowledge"]  # Always execute this first
    )
    
    return engine



def parse_query_example():
    
    client = OpenAIUtils(api_key=os.getenv("AZURE_OPENAI_API_KEY"))
    db_manager = DatabaseManager(
        visible_tables=[
            "FundCashFlow",
            "FundInvestmentProperties",
            "FundInvestmentTimeProperties",
            "FundTimeProperties",
            "HandTransformedPortfolioInvestmentswithCFRaw"
        ]
    )
    
    
    def _disambiguate_query(query: str) -> str:
        """
        Disambiguate the user query if needed.
        
        Args:
            query: Original user query
        Returns:

            Disambiguated query
        """
        
        
        
        entity_labels = [r[0] for r in db_manager.execute("""
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
        response = client._query_azure_openai(disambiguate_prompt, system_prompt)
        return response.strip()
    
    """
    Example of parsing a natural language query into SQL.
    """
    # Set up the framework
    engine = setup_framework()
    
    # Natural language query
    query = "How much has Ascendent Capital Partners I distributed to investors?"
    query = _disambiguate_query(query)
    target_format = "SQL"
    
    print(f"\nParsing query: {query}")
    print(f"Target format: {target_format}")
    print("="*80)
    
    # Run the parsing
    trace = engine.parse(
        query=query,
        target_format=target_format,
        initial_context={"database": "ecommerce"}
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nFinal Output:\n{trace.final_output}")
    print(f"\nTotal Steps: {trace.get_step_count()}")
    print(f"Duration: {trace.get_duration():.2f} seconds")
    print(f"Completed: {trace.is_complete}")
    
    # Display step-by-step reasoning
    print("\n" + "="*80)
    print("REASONING TRACE")
    print("="*80)
    for step in trace.steps:
        print(f"\nStep {step.step_number}:")
        print(f"  Thought: {step.thought.content}")
        print(f"  Action: {step.action.action_name}")
        print(f"  Parameters: {step.action.parameters}")
        if step.observation:
            print(f"  Result: {step.observation.result}")
    
    return trace


def streaming_example():
    """
    Example of using streaming mode to see reasoning in real-time.
    """
    engine = setup_framework()
    
    query = "Get the total revenue by product category for the last quarter"
    target_format = "SQL"
    
    print(f"\nStreaming parse for: {query}")
    print("="*80)
    
    # Use streaming mode
    for item in engine.run_streaming(
        query=query,
        target_format=target_format
    ):
        # Check if it's a step or the final trace
        if isinstance(item, ReasoningStep):
            print(f"\n[Step {item.step_number}]")
            print(f"Thought: {item.thought.content}")
            print(f"Action: {item.action.action_name}")
            if item.observation:
                print(f"Result: {item.observation.result}")
        else:
            # Final trace
            print(f"\n[COMPLETE]")
            print(f"Final Output: {item.final_output}")


def multi_format_example():
    """
    Example of parsing queries into different logical forms.
    """
    engine = setup_framework()
    
    queries = [
        ("Find all companies founded after 2000", "SQL"),
        ("Get all people who work for tech companies", "SPARQL"),
        ("Find the shortest path between two users in the social network", "Cypher"),
    ]
    
    results = []
    
    for query, format_type in queries:
        print(f"\nParsing: {query} → {format_type}")
        trace = engine.parse(query, format_type)
        results.append((query, format_type, trace.final_output))
    
    # Display all results
    print("\n" + "="*80)
    print("ALL RESULTS")
    print("="*80)
    for query, format_type, output in results:
        print(f"\nQuery: {query}")
        print(f"Format: {format_type}")
        print(f"Output: {output}")


if __name__ == "__main__":
    # Run the basic example
    print("="*80)
    print("BASIC USAGE EXAMPLE")
    print("="*80)
    
    try:
        parse_query_example()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set up your .env file with:")
        print("  AZURE_OPENAI_ENDPOINT=your_endpoint")
        print("  AZURE_OPENAI_API_KEY=your_api_key")
    
    # Uncomment to run other examples:
    # streaming_example()
    # multi_format_example()