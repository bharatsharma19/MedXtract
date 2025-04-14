from typing import List, Dict, Any
from datetime import datetime
import logging
from config import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def calculate_statistical_consensus(
    extractions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate statistical consensus from multiple extractions"""
    if not extractions:
        return {}

    # Group biomarkers by test name
    biomarker_groups = {}
    for extraction in extractions:
        if "biomarkers" in extraction:
            for biomarker in extraction["biomarkers"]:
                test_name = biomarker.get("test_name", "").strip().lower()
                if test_name:
                    if test_name not in biomarker_groups:
                        biomarker_groups[test_name] = []
                    biomarker_groups[test_name].append(biomarker)

    # Calculate consensus for each biomarker
    consensus_data = {"biomarkers": [], "notes": []}
    confidence_scores = {}

    for test_name, biomarkers in biomarker_groups.items():
        if len(biomarkers) < 2:
            # Add source model info before adding to consensus
            source_model = biomarkers[0].get(
                "source_model",
                biomarkers[0].get("_metadata", {}).get("model_type", "unknown"),
            )
            consensus_biomarker = biomarkers[0].copy()
            consensus_biomarker["source_models"] = [source_model]
            consensus_data["biomarkers"].append(consensus_biomarker)
            confidence_scores[test_name] = 0.5
            continue

        # Calculate value consensus
        values = [b.get("value") for b in biomarkers if b.get("value") is not None]
        if values:
            try:
                numeric_values = [
                    float(v)
                    for v in values
                    if isinstance(v, (int, float))
                    or (isinstance(v, str) and v.replace(".", "").isdigit())
                ]
                if numeric_values:
                    avg_value = sum(numeric_values) / len(numeric_values)
                    std_dev = (
                        sum((x - avg_value) ** 2 for x in numeric_values)
                        / len(numeric_values)
                    ) ** 0.5
                    confidence = 1 - (std_dev / avg_value if avg_value != 0 else 1)
                else:
                    avg_value = values[0]
                    confidence = 0.5
            except:
                avg_value = values[0]
                confidence = 0.5
        else:
            avg_value = None
            confidence = 0

        # Calculate unit consensus
        units = [b.get("unit", "").strip().lower() for b in biomarkers]
        unit_consensus = max(set(units), key=units.count) if units else ""

        # Calculate reference range consensus
        ranges = [b.get("reference_range", "").strip() for b in biomarkers]
        range_consensus = max(set(ranges), key=ranges.count) if ranges else ""

        # Get source models for this biomarker
        source_models = []
        for biomarker in biomarkers:
            # First check if source_model was directly added
            source_model = biomarker.get("source_model")
            if not source_model:
                # Try to get from metadata
                source_model = biomarker.get("_metadata", {}).get("model_type")

            if source_model and source_model not in source_models:
                source_models.append(source_model)

        # Only include if confidence meets threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            consensus_data["biomarkers"].append(
                {
                    "test_name": biomarkers[0]["test_name"],  # Use original casing
                    "value": avg_value,
                    "unit": unit_consensus,
                    "reference_range": range_consensus,
                    "confidence": confidence,
                    "source_models": source_models if source_models else ["unknown"],
                }
            )
            confidence_scores[test_name] = confidence

    consensus_data["_confidence_scores"] = confidence_scores
    consensus_data["_model_count"] = len(extractions)
    consensus_data["_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    return consensus_data
