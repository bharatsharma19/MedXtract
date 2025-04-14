import logging
from typing import Dict, Any, List
from datetime import datetime
import os
from pathlib import Path
from utils.file_utils import get_timestamp, save_json_data
import json

logger = logging.getLogger(__name__)


def create_workflow_response(
    status: str,
    pdf_path: str,
    metadata: Dict[str, Any],
    extracted_data: List[Dict[str, Any]] = None,
    extraction_by_agent: Dict[str, Dict[str, Any]] = None,
    successful_extractions: List[Dict[str, Any]] = None,
    consensus_data: Dict[str, Any] = None,
    llm_consensus_data: Dict[str, Any] = None,
    validated_data: Dict[str, Any] = None,
    normalized_data: Dict[str, Any] = None,
    errors: List[str] = None,
) -> Dict[str, Any]:
    """Create a formatted response from workflow results

    Args:
        status: Status of the workflow
        pdf_path: Path to the processed PDF
        metadata: Metadata from the workflow
        extracted_data: Extraction results
        extraction_by_agent: Extraction data organized by agent name
        successful_extractions: List of successful extractions
        consensus_data: Statistical consensus data
        llm_consensus_data: LLM consensus data
        validated_data: Validated data
        normalized_data: Normalized data
        errors: List of errors that occurred

    Returns:
        Formatted response dictionary
    """
    # Get timestamp for final response
    final_timestamp = get_timestamp()

    # Convert pdf_path to string if it's a Path object
    if isinstance(pdf_path, Path):
        pdf_path = str(pdf_path)

    # Try to load consensus data from file if it's not provided but metadata has the file path
    if not consensus_data and metadata.get("consensus_file"):
        consensus_file_path = os.path.join(
            "outputs/consensus_data", metadata["consensus_file"]
        )
        if os.path.exists(consensus_file_path):
            try:
                with open(consensus_file_path, "r") as f:
                    consensus_data = json.load(f)
                logger.info(f"Loaded consensus data from {consensus_file_path}")
            except Exception as e:
                logger.error(f"Error loading consensus data: {str(e)}")

    # Build file paths dictionary
    file_paths = {}
    for key in [
        "consensus_file",
        "llm_consensus_file",
        "validated_file",
        "normalized_file",
        "final_file",
        "final_csv",
    ]:
        if key in metadata:
            file_name = metadata[key]
            if file_name:
                if key.startswith("consensus"):
                    file_paths[key] = f"outputs/consensus_data/{file_name}"
                else:
                    file_paths[key] = f"outputs/final_extraction/{file_name}"

    # Extract raw extraction paths
    raw_extraction_paths = []
    if extracted_data:
        for result in extracted_data:
            model_name = result.get("model", "unknown")
            timestamp = result.get(
                "timestamp", metadata.get("extraction_timestamp", "unknown")
            )
            raw_extraction_paths.append(
                f"outputs/raw_extractions/{model_name}/{timestamp}.json"
            )

    # Calculate duration
    start_time = metadata.get("start_time", "")
    duration_seconds = 0
    if start_time:
        try:
            duration_seconds = (
                datetime.strptime(final_timestamp, "%Y%m%d_%H%M%S")
                - datetime.strptime(start_time, "%Y%m%d_%H%M%S")
            ).total_seconds()
        except Exception as e:
            logger.warning(f"Could not calculate duration: {str(e)}")

    # Generate statistics to include in the response
    stats = generate_extraction_statistics(
        extracted_data=extracted_data,
        extraction_by_agent=extraction_by_agent,
        successful_extractions=successful_extractions,
        consensus_data=consensus_data,
        llm_consensus_data=llm_consensus_data,
        validated_data=validated_data,
        normalized_data=normalized_data,
    )

    # Prepare the data section of the response
    # Prioritize including actual data rather than empty objects
    response_data = {}

    # Include normalized data if available
    if (
        normalized_data
        and isinstance(normalized_data, dict)
        and normalized_data.get("biomarkers")
    ):
        response_data["normalized_data"] = _ensure_json_serializable(normalized_data)

    # Include consensus data if available - check both sources
    if (
        consensus_data
        and isinstance(consensus_data, dict)
        and consensus_data.get("biomarkers")
    ):
        response_data["statistical_consensus"] = _ensure_json_serializable(
            consensus_data
        )

    # Include LLM consensus if available
    if (
        llm_consensus_data
        and isinstance(llm_consensus_data, dict)
        and llm_consensus_data.get("biomarkers")
    ):
        response_data["llm_consensus"] = _ensure_json_serializable(llm_consensus_data)

    # Include validated data if available
    if (
        validated_data
        and isinstance(validated_data, dict)
        and validated_data.get("biomarkers")
    ):
        response_data["validated_data"] = _ensure_json_serializable(validated_data)

    # Include extraction by agent if available
    if extraction_by_agent and extraction_by_agent:
        response_data["extraction_by_agent"] = _ensure_json_serializable(
            extraction_by_agent
        )

    # Include successful extractions if available
    if successful_extractions and successful_extractions:
        response_data["successful_extractions"] = _ensure_json_serializable(
            successful_extractions
        )

    # If we still have no data but statistics show biomarkers, try to use what we have
    if (not response_data or response_data == {}) and stats["biomarker_stats"][
        "consensus_biomarkers"
    ] > 0:
        # Try to include consensus data since statistics indicate it exists
        if consensus_data and isinstance(consensus_data, dict):
            response_data["statistical_consensus"] = _ensure_json_serializable(
                consensus_data
            )

    # If no data was included, use empty objects
    if not response_data:
        response_data = {
            "normalized_data": {},
            "statistical_consensus": {},
            "llm_consensus": {},
            "validated_data": {},
            "extraction_by_agent": {},
            "successful_extractions": [],
        }

    # Prepare the response
    response = {
        "id": final_timestamp,
        "status": status,
        "data": response_data,
        "statistics": stats,  # Add statistics to response
        "file_paths": file_paths,
        "raw_extraction_paths": raw_extraction_paths,
        "metadata": {
            "pdf_path": pdf_path,
            "total_models": metadata.get("total_models", 0),
            "successful_models": metadata.get("successful_models", 0),
            "failed_models": metadata.get("failed_models", 0),
            "start_time": start_time,
            "end_time": final_timestamp,
            "duration_seconds": duration_seconds,
            "extraction_timestamp": metadata.get("extraction_timestamp", ""),
            "parallel_extraction": True,  # Flag to indicate parallel extraction was used
        },
    }

    # Add errors if any
    if errors:
        response["errors"] = errors

    # Save final response
    final_response_file = f"complete_response_{final_timestamp}.json"
    save_json_data(response, "outputs/final_extraction", final_response_file)
    logger.info(
        f"Saved final complete response to outputs/final_extraction/{final_response_file}"
    )

    return response


def generate_extraction_statistics(
    extracted_data=None,
    extraction_by_agent=None,
    successful_extractions=None,
    consensus_data=None,
    llm_consensus_data=None,
    validated_data=None,
    normalized_data=None,
):
    """Generate statistics about the extraction process and results

    Args:
        extracted_data: Raw extraction results
        extraction_by_agent: Extraction data organized by agent
        successful_extractions: List of successful extractions
        consensus_data: Statistical consensus data
        llm_consensus_data: LLM consensus data
        validated_data: Validated data
        normalized_data: Normalized data

    Returns:
        Dictionary of statistics
    """
    stats = {
        "extraction_stats": {
            "total_extractions": len(extracted_data) if extracted_data else 0,
            "successful_extractions": (
                len(successful_extractions) if successful_extractions else 0
            ),
            "success_rate": round(
                (
                    (len(successful_extractions) / len(extracted_data) * 100)
                    if extracted_data and len(extracted_data) > 0
                    else 0
                ),
                2,
            ),
        },
        "biomarker_stats": {
            "total_biomarkers_extracted": (
                sum(len(ext.get("biomarkers", [])) for ext in successful_extractions)
                if successful_extractions
                else 0
            ),
            "unique_biomarkers": (
                len(
                    {
                        biomarker.get("test_name", "").lower()
                        for ext in successful_extractions or []
                        for biomarker in ext.get("biomarkers", [])
                        if biomarker.get("test_name")
                    }
                )
                if successful_extractions
                else 0
            ),
            "consensus_biomarkers": (
                len(consensus_data.get("biomarkers", [])) if consensus_data else 0
            ),
            "llm_consensus_biomarkers": (
                len(llm_consensus_data.get("biomarkers", []))
                if llm_consensus_data
                else 0
            ),
            "validated_biomarkers": (
                len(validated_data.get("biomarkers", []))
                if validated_data and isinstance(validated_data, dict)
                else 0
            ),
            "normalized_biomarkers": (
                len(normalized_data.get("biomarkers", []))
                if normalized_data and isinstance(normalized_data, dict)
                else 0
            ),
        },
        "agent_stats": {
            "total_agents": len(extraction_by_agent) if extraction_by_agent else 0,
            "biomarkers_by_agent": {
                agent: len(data.get("biomarkers", []))
                for agent, data in (extraction_by_agent or {}).items()
            },
        },
        "confidence_stats": {
            "avg_confidence": _calculate_avg_confidence(consensus_data),
            "biomarkers_high_confidence": _count_high_confidence_biomarkers(
                consensus_data
            ),
        },
    }

    return stats


def _calculate_avg_confidence(consensus_data):
    """Calculate average confidence score from consensus data"""
    if not consensus_data or not isinstance(consensus_data, dict):
        return 0

    confidence_scores = consensus_data.get("_confidence_scores", {})
    if not confidence_scores:
        return 0

    values = list(confidence_scores.values())
    return round(sum(values) / len(values), 2) if values else 0


def _count_high_confidence_biomarkers(consensus_data, threshold=0.8):
    """Count biomarkers with high confidence scores"""
    if not consensus_data or not isinstance(consensus_data, dict):
        return 0

    confidence_scores = consensus_data.get("_confidence_scores", {})
    if not confidence_scores:
        return 0

    return sum(1 for score in confidence_scores.values() if score >= threshold)


def _ensure_json_serializable(data: Any) -> Any:
    """Ensure data is JSON serializable by converting Path objects to strings

    Args:
        data: Data to make JSON serializable

    Returns:
        JSON serializable version of the data
    """
    if data is None:
        return {}

    if isinstance(data, Path):
        return str(data)

    if isinstance(data, dict):
        return {k: _ensure_json_serializable(v) for k, v in data.items()}

    if isinstance(data, list):
        return [_ensure_json_serializable(item) for item in data]

    return data
