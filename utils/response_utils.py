import logging
from typing import Dict, Any, List
from datetime import datetime
import os
from pathlib import Path
from utils.file_utils import get_timestamp, save_json_data

logger = logging.getLogger(__name__)


def create_workflow_response(
    status: str,
    pdf_path: str,
    metadata: Dict[str, Any],
    extracted_data: List[Dict[str, Any]] = None,
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

    # Build file paths dictionary
    file_paths = {}
    for key in [
        "consensus_file",
        "llm_consensus_file",
        "validated_file",
        "normalized_file",
        "final_file",
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
                f"outputs/raw_extractions/{model_name}/{timestamp}_{model_name}.json"
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

    # Prepare the response
    response = {
        "id": final_timestamp,
        "status": status,
        "data": {
            "normalized_data": _ensure_json_serializable(normalized_data or {}),
            "statistical_consensus": _ensure_json_serializable(consensus_data or {}),
            "llm_consensus": _ensure_json_serializable(llm_consensus_data or {}),
            "validated_data": _ensure_json_serializable(validated_data or {}),
        },
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
