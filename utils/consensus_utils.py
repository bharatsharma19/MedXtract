import logging
from typing import Dict, Any, List
from utils.file_utils import get_timestamp, save_json_data, sanitize_biomarkers
from agents.consensus_agent import run_consensus_analysis

logger = logging.getLogger(__name__)


def run_llm_consensus_pipeline(
    extractions: List[Dict[str, Any]], model_type: str = "gpt-4"
) -> Dict[str, Any]:
    """Run the LLM consensus pipeline on extraction data

    Args:
        extractions: List of extraction results
        model_type: LLM model to use for consensus (default: gpt-4)

    Returns:
        LLM consensus data
    """
    if not extractions:
        logger.error("No extractions provided for LLM consensus")
        return {
            "biomarkers": [],
            "notes": ["No extractions provided for LLM consensus"],
        }

    timestamp = get_timestamp()

    try:
        # Sanitize extractions
        sanitized_extractions = []
        for extraction in extractions:
            sanitized_extraction = sanitize_biomarkers(extraction)
            if sanitized_extraction and sanitized_extraction.get("biomarkers"):
                sanitized_extractions.append(sanitized_extraction)

        if not sanitized_extractions:
            logger.error("No valid sanitized extractions for LLM consensus")
            return {
                "biomarkers": [],
                "notes": ["No valid sanitized extractions for LLM consensus"],
            }

        # Run consensus analysis
        llm_consensus_data = run_consensus_analysis(
            sanitized_extractions, model_type=model_type
        )

        if not llm_consensus_data or not isinstance(
            llm_consensus_data.get("biomarkers"), list
        ):
            logger.error("Invalid LLM consensus data structure")
            return {
                "biomarkers": [],
                "notes": ["Failed to generate LLM consensus - invalid data structure"],
                "_timestamp": timestamp,
            }

        if not llm_consensus_data["biomarkers"]:
            logger.error("No biomarkers in LLM consensus data")
            return {
                "biomarkers": [],
                "notes": ["Failed to generate LLM consensus - no biomarkers"],
                "_timestamp": timestamp,
            }

        # Add timestamp to metadata if not present
        if "_timestamp" not in llm_consensus_data:
            llm_consensus_data["_timestamp"] = timestamp

        # Ensure metadata is present
        if "metadata" not in llm_consensus_data:
            llm_consensus_data["metadata"] = {}

        # Add additional metadata
        llm_consensus_data["metadata"].update(
            {
                "total_models": len(sanitized_extractions),
                "successful_models": len(sanitized_extractions),
                "model_type": model_type,
                "timestamp": timestamp,
            }
        )

        # Save LLM consensus data
        consensus_file = f"llm_consensus_{timestamp}.json"
        save_json_data(llm_consensus_data, "outputs/consensus_data", consensus_file)
        logger.info(f"Saved LLM consensus to outputs/consensus_data/{consensus_file}")

        return llm_consensus_data

    except Exception as e:
        logger.error(f"Error in LLM consensus generation: {str(e)}")

        # Create default LLM consensus data for error case
        default_consensus = {
            "biomarkers": [],
            "notes": [f"LLM consensus error: {str(e)}"],
            "metadata": {"error": str(e), "timestamp": timestamp},
        }

        # Save error consensus data for debugging
        error_file = f"llm_consensus_error_{timestamp}.json"
        save_json_data(default_consensus, "outputs/consensus_data", error_file)
        logger.info(f"Saved error LLM consensus to outputs/consensus_data/{error_file}")

        return default_consensus
