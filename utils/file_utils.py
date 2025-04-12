import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Return a formatted timestamp string for file naming and tracking"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json_data(data: Dict[str, Any], directory: str, filename: str) -> str:
    """Save JSON data to a file with proper directory structure

    Args:
        data: The data to save as JSON
        directory: The directory path to save to
        filename: The filename to use

    Returns:
        The full output path where the file was saved
    """
    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(directory, filename)

    # Convert any Path objects to strings before saving
    data_serializable = _convert_paths_to_str(data)

    with open(output_path, "w") as f:
        json.dump(data_serializable, f, indent=2)
    logger.info(f"Saved data to {output_path}")
    return output_path


def _convert_paths_to_str(data: Any) -> Any:
    """Recursively convert any Path objects to strings

    Args:
        data: The data structure to convert

    Returns:
        The same data structure with Path objects converted to strings
    """
    if isinstance(data, Path):
        return str(data)
    elif isinstance(data, dict):
        return {k: _convert_paths_to_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_paths_to_str(item) for item in data]
    else:
        return data


def ensure_output_directories():
    """Create all necessary output directories for the workflow"""
    directories = [
        "outputs/raw_extractions",
        "outputs/consensus_data",
        "outputs/final_extraction",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def sanitize_biomarkers(biomarkers_data):
    """Sanitize biomarker data by ensuring no None values and proper structure

    Args:
        biomarkers_data: List of biomarker dictionaries or a dictionary with a biomarkers key

    Returns:
        Sanitized biomarkers data with the same structure as input but with safe values
    """
    if not biomarkers_data:
        return []

    # Handle both list and dict with biomarkers key
    if isinstance(biomarkers_data, dict) and "biomarkers" in biomarkers_data:
        biomarkers_list = biomarkers_data["biomarkers"]
        if not isinstance(biomarkers_list, list):
            return {"biomarkers": []}

        safe_biomarkers = []
        for biomarker in biomarkers_list:
            if not isinstance(biomarker, dict):
                continue

            safe_biomarker = biomarker.copy()

            # Ensure test_name is never None
            if safe_biomarker.get("test_name") is None:
                safe_biomarker["test_name"] = "Unknown Test"

            # Ensure other fields are not None
            for field in ["value", "unit", "reference_range"]:
                if safe_biomarker.get(field) is None:
                    safe_biomarker[field] = ""

            safe_biomarkers.append(safe_biomarker)

        result = biomarkers_data.copy()
        result["biomarkers"] = safe_biomarkers
        return result

    # Handle direct list of biomarkers
    elif isinstance(biomarkers_data, list):
        safe_biomarkers = []
        for biomarker in biomarkers_data:
            if not isinstance(biomarker, dict):
                continue

            safe_biomarker = biomarker.copy()

            # Ensure test_name is never None
            if safe_biomarker.get("test_name") is None:
                safe_biomarker["test_name"] = "Unknown Test"

            # Ensure other fields are not None
            for field in ["value", "unit", "reference_range"]:
                if safe_biomarker.get(field) is None:
                    safe_biomarker[field] = ""

            safe_biomarkers.append(safe_biomarker)

        return safe_biomarkers

    # Return empty list for any other case
    return []
