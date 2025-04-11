from statistics import mode, mean, stdev
from typing import List, Dict, Any, Union, Optional
import re


def is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def clean_numeric(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        # Remove any non-numeric characters except . and -
        cleaned = re.sub(r"[^\d.-]", "", str(value))
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def get_consensus_value(values: List[Any]) -> Any:
    if not values:
        return None

    # For numeric values, use statistical consensus
    numeric_values = [clean_numeric(v) for v in values if clean_numeric(v) is not None]
    if numeric_values:
        if len(numeric_values) >= 3:
            # Remove outliers (values more than 2 standard deviations from mean)
            mean_val = mean(numeric_values)
            std_val = stdev(numeric_values)
            filtered_values = [
                v for v in numeric_values if abs(v - mean_val) <= 2 * std_val
            ]
            return mean(filtered_values) if filtered_values else numeric_values[0]
        return mean(numeric_values)

    # For non-numeric values, use mode with string normalization
    try:
        normalized_values = [str(v).strip().lower() for v in values if v]
        return mode(normalized_values)
    except:
        return values[0] if values else None


def validate_agents_data(agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates and combines outputs from multiple extraction agents using consensus mechanisms
    and data quality checks.
    """
    if not agent_outputs:
        return {}

    validated = {}
    confidence_scores = {}

    # Collect all possible keys
    all_keys = set()
    for output in agent_outputs:
        if isinstance(output, dict):
            all_keys.update(output.keys())

    # Validate each key
    for key in all_keys:
        values = []
        for output in agent_outputs:
            if isinstance(output, dict) and key in output:
                values.append(output[key])

        if values:
            consensus_value = get_consensus_value(values)
            validated[key] = consensus_value

            # Calculate confidence score based on agreement
            if consensus_value is not None:
                matching_values = sum(
                    1
                    for v in values
                    if str(v).strip().lower() == str(consensus_value).strip().lower()
                )
                confidence_scores[key] = matching_values / len(values)

    # Add confidence scores to output
    validated["_confidence_scores"] = confidence_scores

    return validated
