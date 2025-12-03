"""
Graph-Based Decomposition Example

This example demonstrates the new directed graph control flow
where composition operators are edges between subtasks.
"""

from semantic_parser.decompose_utils import (
    SubTask,
    CompositionOperator,
    ControlFlow,
    ControlFlowEdge,
    TaskDecomposition,
    visualize_control_flow
)


def create_example_decomposition():
    """
    Create an example decomposition for:
    "Compare MOIC Q4 2024 vs Q4 2023 and rank by improvement"
    
    Graph Structure:
    
        start
          ↓
      ┌───┴───┐
      ↓       ↓
    Task1   Task2
    (2024)  (2023)
      ↓       ↓
      └───┬───┘
          ↓ (via op_merge)
        Task3
        (rank)
          ↓ (via op_pass)
         end
    """
    
    # Define subtasks (nodes)
    subtasks = [
        SubTask(
            subtask_id="get_moic_2024",
            description="Get MOIC data for all funds in Q4 2024"
        ),
        SubTask(
            subtask_id="get_moic_2023",
            description="Get MOIC data for all funds in Q4 2023"
        ),
        SubTask(
            subtask_id="calculate_and_rank",
            description="Calculate MOIC improvements and rank funds"
        )
    ]
    
    # Define composition operators (edge transformations)
    operators = [
        CompositionOperator(
            operator_id="op_merge",
            description="Merge 2024 and 2023 data for comparison",
            input_subtasks=["get_moic_2024", "get_moic_2023"],
            output_name="merged_moic_data",
            code="""
def op_merge(get_moic_2024, get_moic_2023):
    \"\"\"Merge 2024 and 2023 MOIC data.\"\"\"
    # Assuming both are dictionaries: {fund_id: moic_value}
    merged = {}
    for fund_id in get_moic_2024:
        if fund_id in get_moic_2023:
            merged[fund_id] = {
                'moic_2024': get_moic_2024[fund_id],
                'moic_2023': get_moic_2023[fund_id],
                'improvement': get_moic_2024[fund_id] - get_moic_2023[fund_id]
            }
    return merged
"""
        ),
        CompositionOperator(
            operator_id="op_pass",
            description="Pass through ranked results",
            input_subtasks=["calculate_and_rank"],
            output_name="final_result",
            code="""
def op_pass(calculate_and_rank):
    \"\"\"Pass through the result unchanged.\"\"\"
    return calculate_and_rank
"""
        )
    ]
    
    # Define control flow (directed graph)
    edges = [
        # Start edges (no operator needed)
        ControlFlowEdge(from_node="start", to_node="get_moic_2024", operator_id=None),
        ControlFlowEdge(from_node="start", to_node="get_moic_2023", operator_id=None),
        
        # Merge edges (both feed into Task 3 via merge operator)
        ControlFlowEdge(from_node="get_moic_2024", to_node="calculate_and_rank", operator_id="op_merge"),
        ControlFlowEdge(from_node="get_moic_2023", to_node="calculate_and_rank", operator_id="op_merge"),
        
        # End edge (pass through to final result)
        ControlFlowEdge(from_node="calculate_and_rank", to_node="end", operator_id="op_pass")
    ]
    
    control_flow = ControlFlow(
        edges=edges,
        node_ids=["get_moic_2024", "get_moic_2023", "calculate_and_rank"]
    )
    
    # Create full decomposition
    decomposition = TaskDecomposition(
        subtasks=subtasks,
        composition_operators=operators,
        control_flow=control_flow,
        reasoning="Split into parallel data retrieval, then merge and rank"
    )
    
    # Validate
    decomposition.validate()
    
    return decomposition


def create_sequential_example():
    """
    Create an example with sequential dependencies:
    "Find funds with MOIC > 2.0, then get their cash flow, then rank by cash flow"
    
    Graph Structure:
    
        start
          ↓
        Task1 (find MOIC > 2.0)
          ↓ (via op_extract_ids)
        Task2 (get cash flow for those funds)
          ↓ (via op_attach_cashflow)
        Task3 (rank by cash flow)
          ↓ (via op_pass)
         end
    """
    
    subtasks = [
        SubTask(
            subtask_id="find_high_moic",
            description="Find all funds with MOIC > 2.0"
        ),
        SubTask(
            subtask_id="get_cashflow",
            description="Get cash flow data for specified funds"
        ),
        SubTask(
            subtask_id="rank_by_cashflow",
            description="Rank funds by their cash flow"
        )
    ]
    
    operators = [
        CompositionOperator(
            operator_id="op_extract_ids",
            description="Extract fund IDs from MOIC query result",
            input_subtasks=["find_high_moic"],
            output_name="fund_ids",
            code="""
def op_extract_ids(find_high_moic):
    \"\"\"Extract fund IDs from query result.\"\"\"
    # Assuming result is a list of dicts with 'fund_id' key
    return [fund['fund_id'] for fund in find_high_moic]
"""
        ),
        CompositionOperator(
            operator_id="op_attach_cashflow",
            description="Attach cash flow data to fund IDs",
            input_subtasks=["get_cashflow"],
            output_name="funds_with_cashflow",
            code="""
def op_attach_cashflow(get_cashflow):
    \"\"\"Prepare data for ranking.\"\"\"
    return get_cashflow
"""
        ),
        CompositionOperator(
            operator_id="op_pass",
            description="Pass final ranking",
            input_subtasks=["rank_by_cashflow"],
            output_name="final_ranking",
            code="""
def op_pass(rank_by_cashflow):
    \"\"\"Pass through unchanged.\"\"\"
    return rank_by_cashflow
"""
        )
    ]
    
    edges = [
        ControlFlowEdge(from_node="start", to_node="find_high_moic", operator_id=None),
        ControlFlowEdge(from_node="find_high_moic", to_node="get_cashflow", operator_id="op_extract_ids"),
        ControlFlowEdge(from_node="get_cashflow", to_node="rank_by_cashflow", operator_id="op_attach_cashflow"),
        ControlFlowEdge(from_node="rank_by_cashflow", to_node="end", operator_id="op_pass")
    ]
    
    control_flow = ControlFlow(
        edges=edges,
        node_ids=["find_high_moic", "get_cashflow", "rank_by_cashflow"]
    )
    
    decomposition = TaskDecomposition(
        subtasks=subtasks,
        composition_operators=operators,
        control_flow=control_flow,
        reasoning="Sequential pipeline with data transformation at each stage"
    )
    
    decomposition.validate()
    return decomposition


def create_complex_example():
    """
    Create a complex example with both parallel and sequential:
    "Analyze funds: get average MOIC, find top 5 by capital, and compare Q4 performance"
    
    Graph Structure:
    
            start
              ↓
        ┌─────┼─────┐
        ↓     ↓     ↓
      Task1 Task2 Task3
      (avg) (top5) (Q4 comp)
        ↓     ↓     ↓
        └─────┼─────┘
              ↓ (via op_merge_all)
            Task4 (format report)
              ↓ (via op_pass)
             end
    """
    
    subtasks = [
        SubTask("calc_avg_moic", "Calculate average MOIC for all funds"),
        SubTask("find_top5", "Find top 5 funds by invested capital"),
        SubTask("compare_q4", "Compare Q4 2024 vs Q4 2023 performance"),
        SubTask("format_report", "Format comprehensive analysis report")
    ]
    
    operators = [
        CompositionOperator(
            operator_id="op_merge_all",
            description="Merge all analysis results",
            input_subtasks=["calc_avg_moic", "find_top5", "compare_q4"],
            output_name="merged_analysis",
            code="""
def op_merge_all(calc_avg_moic, find_top5, compare_q4):
    \"\"\"Merge all analysis results into one structure.\"\"\"
    return {
        'average_moic': calc_avg_moic,
        'top_funds': find_top5,
        'q4_comparison': compare_q4
    }
"""
        ),
        CompositionOperator(
            operator_id="op_pass",
            description="Pass formatted report",
            input_subtasks=["format_report"],
            output_name="final_report",
            code="""
def op_pass(format_report):
    \"\"\"Pass through report.\"\"\"
    return format_report
"""
        )
    ]
    
    edges = [
        # Parallel start
        ControlFlowEdge("start", "calc_avg_moic", None),
        ControlFlowEdge("start", "find_top5", None),
        ControlFlowEdge("start", "compare_q4", None),
        
        # Merge into report
        ControlFlowEdge("calc_avg_moic", "format_report", "op_merge_all"),
        ControlFlowEdge("find_top5", "format_report", "op_merge_all"),
        ControlFlowEdge("compare_q4", "format_report", "op_merge_all"),
        
        # Final output
        ControlFlowEdge("format_report", "end", "op_pass")
    ]
    
    control_flow = ControlFlow(
        edges=edges,
        node_ids=["calc_avg_moic", "find_top5", "compare_q4", "format_report"]
    )
    
    decomposition = TaskDecomposition(
        subtasks=subtasks,
        composition_operators=operators,
        control_flow=control_flow,
        reasoning="Parallel analysis tasks merged into comprehensive report"
    )
    
    decomposition.validate()
    return decomposition


if __name__ == "__main__":
    print("="*80)
    print("GRAPH-BASED DECOMPOSITION EXAMPLES")
    print("="*80)
    
    print("\n1. PARALLEL WITH MERGE")
    print("-"*80)
    decomp1 = create_example_decomposition()
    print(f"Subtasks: {len(decomp1.subtasks)}")
    print(f"Operators: {len(decomp1.composition_operators)}")
    print(f"Edges: {len(decomp1.control_flow.edges)}")
    print("\n" + visualize_control_flow(decomp1.control_flow))
    
    print("\n\n2. SEQUENTIAL PIPELINE")
    print("-"*80)
    decomp2 = create_sequential_example()
    print(f"Subtasks: {len(decomp2.subtasks)}")
    print(f"Operators: {len(decomp2.composition_operators)}")
    print(f"Edges: {len(decomp2.control_flow.edges)}")
    print("\n" + visualize_control_flow(decomp2.control_flow))
    
    print("\n\n3. COMPLEX (PARALLEL + SEQUENTIAL)")
    print("-"*80)
    decomp3 = create_complex_example()
    print(f"Subtasks: {len(decomp3.subtasks)}")
    print(f"Operators: {len(decomp3.composition_operators)}")
    print(f"Edges: {len(decomp3.control_flow.edges)}")
    print("\n" + visualize_control_flow(decomp3.control_flow))
    
    print("\n" + "="*80)
    print("All decompositions validated successfully! ✓")
    print("="*80)