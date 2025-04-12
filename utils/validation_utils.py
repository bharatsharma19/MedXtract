import logging
from typing import Dict, Any, List, Union, Tuple
from utils.file_utils import get_timestamp, save_json_data, sanitize_biomarkers
from utils.validators import validate_agents_data
from utils.normalizer import normalize_and_map

logger = logging.getLogger(__name__)


def run_validation_pipeline(
    extractions: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the validation pipeline on extraction data

    Args:
        extractions: List of extraction results

    Returns:
        Tuple containing:
            - Validated data
            - Metadata dictionary
    """
    if not extractions:
        logger.error("No extractions provided for validation")
        return {}, {"error": "No extractions provided for validation"}

    timestamp = get_timestamp()

    try:
        # Sanitize extractions
        sanitized_extractions = []
        for extraction in extractions:
            sanitized_extraction = sanitize_biomarkers(extraction)
            if sanitized_extraction and sanitized_extraction.get("biomarkers"):
                sanitized_extractions.append(sanitized_extraction)

        if not sanitized_extractions:
            logger.error("No valid sanitized extractions for validation")
            return {}, {"error": "No valid sanitized extractions for validation"}

        # Run validation
        validated_data = validate_agents_data(sanitized_extractions)

        if not validated_data:
            logger.error("Validation failed - no valid data")
            return {}, {"error": "Validation failed - no valid data"}

        # Add metadata to validated data
        if not isinstance(validated_data, dict):
            validated_data = {
                "biomarkers": validated_data if isinstance(validated_data, list) else []
            }

        if "_metadata" not in validated_data:
            validated_data["_metadata"] = {}

        validated_data["_metadata"].update(
            {"timestamp": timestamp, "source": "validation", "validated": True}
        )

        # Save validated data
        validated_file = f"validated_data_{timestamp}.json"
        save_json_data(validated_data, "outputs/final_extraction", validated_file)
        logger.info(
            f"Saved validated data to outputs/final_extraction/{validated_file}"
        )

        return validated_data, {"validated_file": validated_file}

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {}, {"error": f"Validation error: {str(e)}"}


def run_normalization_pipeline(
    validated_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the normalization pipeline on validated data

    Args:
        validated_data: Validated biomarker data

    Returns:
        Tuple containing:
            - Normalized data
            - Metadata dictionary
    """
    if not validated_data:
        logger.error("No validated data provided for normalization")
        return {}, {"error": "No validated data provided for normalization"}

    timestamp = get_timestamp()
    validation_source = validated_data.get("_metadata", {}).get(
        "validated_file", "unknown"
    )

    try:
        # Run normalization
        normalized_data = normalize_and_map(validated_data)

        if not normalized_data:
            logger.error("Normalization failed - no valid data")
            return {}, {"error": "Normalization failed - no valid data"}

        # Add metadata to normalized data
        if isinstance(normalized_data, dict):
            if "_metadata" not in normalized_data:
                normalized_data["_metadata"] = {}

            normalized_data["_metadata"].update(
                {
                    "timestamp": timestamp,
                    "source": "normalization",
                    "normalized": True,
                    "validation_source": validation_source,
                }
            )
        else:
            # Convert to dict if it's not already
            normalized_data = {
                "biomarkers": (
                    normalized_data if isinstance(normalized_data, list) else []
                ),
                "_metadata": {
                    "timestamp": timestamp,
                    "source": "normalization",
                    "normalized": True,
                    "validation_source": validation_source,
                },
            }

        # Save normalized data
        normalized_file = f"normalized_data_{timestamp}.json"
        save_json_data(normalized_data, "outputs/final_extraction", normalized_file)
        logger.info(
            f"Saved normalized data to outputs/final_extraction/{normalized_file}"
        )

        # Save a copy as final result
        final_file = f"final_result_{timestamp}.json"
        save_json_data(normalized_data, "outputs/final_extraction", final_file)
        logger.info(f"Saved final result to outputs/final_extraction/{final_file}")

        return normalized_data, {
            "normalized_file": normalized_file,
            "final_file": final_file,
        }

    except Exception as e:
        logger.error(f"Normalization error: {str(e)}")
        return {}, {"error": f"Normalization error: {str(e)}"}
