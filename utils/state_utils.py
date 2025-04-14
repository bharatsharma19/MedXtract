from typing import Dict, Any, List, TypedDict
import logging
from utils.file_utils import get_timestamp

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """TypedDict defining the structure of the workflow state"""

    pdf_path: str
    extracted_data: List[Dict[str, Any]]
    extraction_by_agent: Dict[str, Dict[str, Any]]  # Store data by agent name
    successful_extractions: List[Dict[str, Any]]  # Store successful extractions
    validated_data: Dict[str, Any]
    normalized_data: Dict[str, Any]
    consensus_data: Dict[str, Any]
    llm_consensus_data: Dict[str, Any]
    errors: List[str]
    status: str
    metadata: Dict[str, Any]


def create_initial_state(pdf_path: str) -> WorkflowState:
    """Create and return an initial state for the workflow

    Args:
        pdf_path: Path to the PDF file to process

    Returns:
        Initial workflow state with default values
    """
    return WorkflowState(
        pdf_path=pdf_path,
        extracted_data=[],
        extraction_by_agent={},  # Initialize empty dict for extraction by agent
        successful_extractions=[],  # Initialize empty list for successful extractions
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


def update_state_metadata(
    state: WorkflowState, metadata_updates: Dict[str, Any]
) -> WorkflowState:
    """Update the metadata in the state

    Args:
        state: Current workflow state
        metadata_updates: Dictionary of metadata to update

    Returns:
        Updated workflow state
    """
    # Create a new copy of the state to avoid modifying the original
    updated_state = state.copy()

    # Update the metadata
    if "metadata" not in updated_state:
        updated_state["metadata"] = {}

    updated_state["metadata"].update(metadata_updates)
    return updated_state


def set_state_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """Add an error message to the state and set status to failed

    Args:
        state: Current workflow state
        error_message: Error message to add

    Returns:
        Updated workflow state with error
    """
    updated_state = state.copy()

    if "errors" not in updated_state:
        updated_state["errors"] = []

    updated_state["errors"].append(error_message)
    updated_state["status"] = "failed"

    return updated_state
