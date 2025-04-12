import logging
from typing import Dict, Any
from langgraph.graph import END
import os

from utils.state_utils import WorkflowState
from utils.extraction_utils import run_extraction_pipeline
from utils.consensus_utils import run_llm_consensus_pipeline
from utils.validation_utils import run_validation_pipeline, run_normalization_pipeline
from utils.file_utils import ensure_output_directories

logger = logging.getLogger(__name__)


def extract_all_agents(state: WorkflowState) -> Dict[str, Any]:
    """Extract biomarker data from PDF using all available extraction agents.

    Args:
        state: Current workflow state

    Returns:
        Dict with next node to execute
    """
    try:
        # Convert pdf_path to string if needed and validate
        pdf_path = state["pdf_path"]
        if hasattr(pdf_path, "__fspath__"):  # Check if it's Path-like
            pdf_path = str(pdf_path)
            state["pdf_path"] = pdf_path

        if not os.path.exists(pdf_path):
            state["errors"].append(f"PDF file not found: {pdf_path}")
            state["status"] = "failed"
            return {"next": END}

        # Create output directories
        ensure_output_directories()

        # Run extraction pipeline
        results, successful_extractions, consensus_data, metadata = (
            run_extraction_pipeline(pdf_path)
        )

        # Check for errors
        if "error" in metadata:
            state["errors"].append(metadata["error"])
            state["status"] = "failed"
            return {"next": END}

        if not successful_extractions:
            state["errors"].append("No successful extractions with valid biomarkers")
            state["status"] = "failed"
            return {"next": END}

        # Update state
        state["extracted_data"] = results
        state["consensus_data"] = consensus_data
        state["metadata"].update(metadata)
        state["status"] = "extracted"

        return {"next": "llm_consensus"}

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        state["errors"].append(f"Extraction error: {str(e)}")
        state["status"] = "failed"
        return {"next": END}


def llm_consensus(state: WorkflowState) -> Dict[str, Any]:
    """Generate LLM consensus from extraction data.

    Args:
        state: Current workflow state

    Returns:
        Dict with next node to execute
    """
    try:
        # Get successful extractions with valid biomarkers
        successful_extractions = []
        for result in state["extracted_data"]:
            try:
                output_data = result.get("output", {})
                if (
                    result.get("status") == "success"
                    and isinstance(output_data, dict)
                    and isinstance(output_data.get("biomarkers"), list)
                    and output_data["biomarkers"]
                ):
                    successful_extractions.append(output_data)
            except Exception as e:
                logger.error(f"Error processing extraction for LLM consensus: {str(e)}")
                continue

        if not successful_extractions:
            state["errors"].append("No valid extractions for LLM consensus analysis")
            state["status"] = "failed"
            return {"next": END}

        # Run LLM consensus
        llm_consensus_data = run_llm_consensus_pipeline(
            successful_extractions, model_type="gpt-4"
        )

        # Check if we have valid LLM consensus data
        if not llm_consensus_data.get("biomarkers"):
            logger.warning(
                "No biomarkers in LLM consensus data, using statistical consensus"
            )

            # Try to use statistical consensus instead
            if state["consensus_data"] and state["consensus_data"].get("biomarkers"):
                state["llm_consensus_data"] = state["consensus_data"]
                state["status"] = "consensus_analyzed_with_fallback"
            else:
                state["errors"].append("Failed to generate consensus data")
                state["status"] = "failed"
                return {"next": END}
        else:
            # Update state with LLM consensus data
            state["llm_consensus_data"] = llm_consensus_data
            state["status"] = "consensus_analyzed"

            # Update metadata
            if "metadata" not in state:
                state["metadata"] = {}

            if llm_consensus_data.get("metadata"):
                consensus_file = llm_consensus_data.get("metadata").get(
                    "consensus_file"
                )
                if consensus_file:
                    state["metadata"]["llm_consensus_file"] = consensus_file

        return {"next": "validate"}

    except Exception as e:
        logger.error(f"LLM consensus error: {str(e)}")
        state["errors"].append(f"LLM consensus error: {str(e)}")

        # Try to continue with statistical consensus if available
        if state["consensus_data"] and state["consensus_data"].get("biomarkers"):
            state["llm_consensus_data"] = state["consensus_data"]
            state["status"] = "consensus_analyzed_with_errors"
            return {"next": "validate"}
        else:
            state["status"] = "failed"
            return {"next": END}


def validate(state: WorkflowState) -> Dict[str, Any]:
    """Validate extracted biomarker data.

    Args:
        state: Current workflow state

    Returns:
        Dict with next node to execute
    """
    try:
        successful_extractions = []
        for result in state["extracted_data"]:
            if result.get("status") == "success" and "output" in result:
                extraction_data = result["output"]
                if (
                    isinstance(extraction_data, dict)
                    and "biomarkers" in extraction_data
                ):
                    successful_extractions.append(extraction_data)

        if not successful_extractions:
            state["errors"].append("No successful extractions to validate")
            state["status"] = "failed"
            return {"next": END}

        # Run validation pipeline
        validated_data, metadata = run_validation_pipeline(successful_extractions)

        # Check for errors
        if "error" in metadata:
            state["errors"].append(metadata["error"])
            state["status"] = "failed"
            return {"next": END}

        # Update state
        state["validated_data"] = validated_data
        state["metadata"].update(metadata)
        state["status"] = "validated"

        return {"next": "normalize"}

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        state["errors"].append(f"Validation error: {str(e)}")
        state["status"] = "failed"
        return {"next": END}


def normalize(state: WorkflowState) -> Dict[str, Any]:
    """Normalize validated biomarker data.

    Args:
        state: Current workflow state

    Returns:
        Dict with next node to execute
    """
    try:
        if not state["validated_data"]:
            state["errors"].append("No validated data to normalize")
            state["status"] = "failed"
            return {"next": END}

        # Run normalization pipeline
        normalized_data, metadata = run_normalization_pipeline(state["validated_data"])

        # Check for errors
        if "error" in metadata:
            state["errors"].append(metadata["error"])
            state["status"] = "failed"
            return {"next": END}

        # Update state
        state["normalized_data"] = normalized_data
        state["metadata"].update(metadata)
        state["status"] = "completed"

        return {"next": END}

    except Exception as e:
        logger.error(f"Normalization error: {str(e)}")
        state["errors"].append(f"Normalization error: {str(e)}")
        state["status"] = "failed"
        return {"next": END}
