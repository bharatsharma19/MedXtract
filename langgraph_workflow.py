from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    pdf_path: str
    extracted_data: List[Dict[str, Any]]
    validated_data: Dict[str, Any]
    normalized_data: Dict[str, Any]
    consensus_data: Dict[str, Any]
    llm_consensus_data: Dict[str, Any]
    errors: List[str]
    status: str
    metadata: Dict[str, Any]


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_langgraph_workflow(pdf_path: str):
    # Define initial state
    initial_state = WorkflowState(
        pdf_path=pdf_path,
        extracted_data=[],
        validated_data={},
        normalized_data={},
        consensus_data={},
        llm_consensus_data={},
        errors=[],
        status="started",
        metadata={
            "start_time": get_timestamp(),
            "total_models": 0,
            "successful_models": 0,
            "failed_models": 0,
        },
    )

    def extract_all_agents(state: WorkflowState):
        try:
            from utils.extractors import run_all_extractions

            # Validate PDF path
            if not os.path.exists(state["pdf_path"]):
                state["errors"].append(f"PDF file not found: {state['pdf_path']}")
                state["status"] = "failed"
                return {"next": END}

            # Create output directories
            os.makedirs("outputs/raw_extractions", exist_ok=True)
            os.makedirs("outputs/final_extraction", exist_ok=True)

            # Run extractions
            results, consensus_data = run_all_extractions(state["pdf_path"])
            state["extracted_data"] = results
            state["consensus_data"] = consensus_data

            # Update metadata
            state["metadata"]["total_models"] = len(results)
            state["metadata"]["successful_models"] = len(
                [r for r in results if r.get("status") == "success"]
            )
            state["metadata"]["failed_models"] = len(
                [r for r in results if r.get("status") == "failed"]
            )

            # Check if we have any successful extractions
            successful_extractions = [
                r for r in results if r.get("status") == "success"
            ]
            if not successful_extractions:
                state["errors"].append("No successful extractions from any model")
                state["status"] = "failed"
                return {"next": END}

            state["status"] = "extracted"
            return {"next": "llm_consensus"}
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            state["errors"].append(f"Extraction error: {str(e)}")
            state["status"] = "failed"
            return {"next": END}

    def llm_consensus(state: WorkflowState):
        try:
            from utils.consensus_agent import run_consensus_analysis

            successful_extractions = [
                r for r in state["extracted_data"] if r.get("status") == "success"
            ]
            if not successful_extractions:
                state["errors"].append(
                    "No successful extractions for LLM consensus analysis"
                )
                state["status"] = "failed"
                return {"next": END}

            # Run LLM consensus analysis
            llm_consensus_data = run_consensus_analysis(
                [result["output"] for result in successful_extractions],
                model_type="gpt-4",
            )

            if llm_consensus_data:
                state["llm_consensus_data"] = llm_consensus_data
                state["status"] = "consensus_analyzed"
                return {"next": "validate"}
            else:
                state["errors"].append("Failed to generate LLM consensus")
                state["status"] = "failed"
                return {"next": END}
        except Exception as e:
            logger.error(f"LLM consensus error: {str(e)}")
            state["errors"].append(f"LLM consensus error: {str(e)}")
            state["status"] = "failed"
            return {"next": END}

    def validate(state: WorkflowState):
        try:
            from utils.validators import validate_agents_data

            successful_extractions = [
                r for r in state["extracted_data"] if r.get("status") == "success"
            ]
            if not successful_extractions:
                state["errors"].append("No successful extractions to validate")
                state["status"] = "failed"
                return {"next": END}

            state["validated_data"] = validate_agents_data(
                [result["output"] for result in successful_extractions]
            )
            state["status"] = "validated"
            return {"next": "normalize"}
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            state["errors"].append(f"Validation error: {str(e)}")
            state["status"] = "failed"
            return {"next": END}

    def normalize(state: WorkflowState):
        try:
            from utils.normalizer import normalize_and_map

            if not state["validated_data"]:
                state["errors"].append("No validated data to normalize")
                state["status"] = "failed"
                return {"next": END}

            state["normalized_data"] = normalize_and_map(state["validated_data"])
            state["status"] = "completed"
            return {"next": END}
        except Exception as e:
            logger.error(f"Normalization error: {str(e)}")
            state["errors"].append(f"Normalization error: {str(e)}")
            state["status"] = "failed"
            return {"next": END}

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

    # Prepare the final response
    response = {
        "id": get_timestamp(),
        "status": final_state["status"],
        "data": {
            "normalized_data": final_state["normalized_data"],
            "statistical_consensus": final_state["consensus_data"],
            "llm_consensus": final_state["llm_consensus_data"],
            "raw_extractions": final_state["extracted_data"],
        },
        "metadata": {
            "total_models": final_state["metadata"]["total_models"],
            "successful_models": final_state["metadata"]["successful_models"],
            "failed_models": final_state["metadata"]["failed_models"],
            "start_time": final_state["metadata"]["start_time"],
            "end_time": get_timestamp(),
            "duration": (
                datetime.strptime(get_timestamp(), "%Y%m%d_%H%M%S")
                - datetime.strptime(
                    final_state["metadata"]["start_time"], "%Y%m%d_%H%M%S"
                )
            ).total_seconds(),
        },
    }

    if final_state["errors"]:
        response["errors"] = final_state["errors"]

    # Save final response
    output_path = f"outputs/final_extraction/final_response_{get_timestamp()}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(response, f, indent=2)
    logger.info(f"Saved final response to {output_path}")

    return response
