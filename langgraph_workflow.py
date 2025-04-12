"""
LangGraph Workflow for Lab Report Extraction

This module defines and executes the workflow for extracting biomarker data
from lab reports using a directed graph of processing steps.
"""

import logging
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from utils
from utils.state_utils import WorkflowState, create_initial_state
from utils.workflow_nodes import extract_all_agents, llm_consensus, validate, normalize
from utils.response_utils import create_workflow_response
from utils.file_utils import ensure_output_directories


def run_langgraph_workflow(pdf_path: str):
    """
    Run the lab report extraction workflow using LangGraph

    Args:
        pdf_path: Path to the PDF file to process

    Returns:
        Dictionary containing the workflow results
    """
    # Ensure output directories exist
    ensure_output_directories()

    # Create initial state
    initial_state = create_initial_state(pdf_path)

    # Build LangGraph with proper state schema
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("extract_all_agents", extract_all_agents)
    workflow.add_node("llm_consensus", llm_consensus)
    workflow.add_node("validate", validate)
    workflow.add_node("normalize", normalize)

    # Define edges
    workflow.set_entry_point("extract_all_agents")
    workflow.add_edge("extract_all_agents", "llm_consensus")
    workflow.add_edge("llm_consensus", "validate")
    workflow.add_edge("validate", "normalize")
    workflow.add_edge("normalize", END)

    # Compile and run the workflow
    graph = workflow.compile()
    final_state = graph.invoke(initial_state)

    # Create and return the final response
    response = create_workflow_response(
        status=final_state["status"],
        pdf_path=final_state["pdf_path"],
        metadata=final_state["metadata"],
        extracted_data=final_state["extracted_data"],
        consensus_data=final_state["consensus_data"],
        llm_consensus_data=final_state["llm_consensus_data"],
        validated_data=final_state["validated_data"],
        normalized_data=final_state["normalized_data"],
        errors=final_state.get("errors", []),
    )

    return response
