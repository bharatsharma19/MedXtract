import os
import logging
from typing import Dict, Any, List, Tuple
from utils.file_utils import get_timestamp, save_json_data, sanitize_biomarkers
from utils.statistical_utils import calculate_statistical_consensus
from agents.extraction_agent import run_all_extractions

logger = logging.getLogger(__name__)


def process_extraction_results(
    results: List[Dict[str, Any]], pdf_path: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Process extraction results and save necessary files

    Args:
        results: List of extraction results from all models
        pdf_path: Path to the PDF file

    Returns:
        Tuple containing:
            - List of successfully processed extraction results
            - List of successful extractions (contains biomarker data)
            - Consensus data
            - Metadata dictionary with extraction metrics
    """
    if not results:
        logger.error("No results from any extraction model")
        return [], [], {}, {"error": "No results from any extraction model"}

    timestamp = get_timestamp()
    successful_extractions = []

    # Save individual raw extractions
    for result in results:
        model_name = result.get("model", "unknown")
        output_data = result.get("output", {})

        if not isinstance(output_data, dict):
            logger.warning(f"Invalid output data format from {model_name}")
            continue

        # Check if extraction was successful and has biomarkers
        if (
            result.get("status") == "success"
            and isinstance(output_data.get("biomarkers"), list)
            and output_data["biomarkers"]
        ):
            # Sanitize biomarkers using the utility function
            output_data = sanitize_biomarkers(output_data)
            successful_extractions.append(output_data)

            # Save raw extraction
            model_dir = f"outputs/raw_extractions/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            raw_file = f"{timestamp}_{model_name}.json"
            save_json_data(output_data, model_dir, raw_file)
            logger.info(f"Saved raw extraction to {model_dir}/{raw_file}")

            logger.info(
                f"Successful extraction from {model_name}: {len(output_data['biomarkers'])} biomarkers"
            )
        else:
            logger.warning(f"No valid biomarkers from {model_name}")

    # Calculate consensus if we have successful extractions
    consensus_data = {}
    if successful_extractions:
        # Sanitize extractions
        sanitized_extractions = []
        for extraction in successful_extractions:
            sanitized_extraction = sanitize_biomarkers(extraction)
            if sanitized_extraction and sanitized_extraction.get("biomarkers"):
                sanitized_extractions.append(sanitized_extraction)

        if sanitized_extractions:
            logger.info(
                f"Calculating consensus with {len(sanitized_extractions)} extractions"
            )

            consensus_data = calculate_statistical_consensus(sanitized_extractions)

            if not consensus_data or not isinstance(
                consensus_data.get("biomarkers"), list
            ):
                logger.error("Invalid consensus data structure")
                # Create a default consensus structure
                consensus_data = {
                    "biomarkers": [],
                    "notes": ["Failed to calculate proper consensus"],
                    "_timestamp": timestamp,
                }

            # Save statistical consensus
            consensus_file = f"statistical_consensus_{timestamp}.json"
            save_json_data(consensus_data, "outputs/consensus_data", consensus_file)
            logger.info(
                f"Saved statistical consensus to outputs/consensus_data/{consensus_file}"
            )
        else:
            logger.warning("No valid sanitized extractions for consensus")
            consensus_data = {
                "biomarkers": [],
                "notes": ["No valid sanitized extractions for consensus"],
                "_timestamp": timestamp,
            }
    else:
        logger.warning("No successful extractions with valid biomarkers")
        consensus_data = {
            "biomarkers": [],
            "notes": ["No successful extractions with valid biomarkers"],
            "_timestamp": timestamp,
        }

    # Create metadata
    metadata = {
        "total_models": len(results),
        "successful_models": len(successful_extractions),
        "failed_models": len(results) - len(successful_extractions),
        "extraction_timestamp": timestamp,
        "consensus_file": (
            f"statistical_consensus_{timestamp}.json"
            if successful_extractions
            else None
        ),
    }

    return results, successful_extractions, consensus_data, metadata


def run_extraction_pipeline(
    pdf_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Run the full extraction pipeline on a PDF

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple containing:
            - List of all extraction results
            - List of successful extractions
            - Consensus data
            - Metadata dictionary
    """
    # Validate PDF path
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return [], [], {}, {"error": f"PDF file not found: {pdf_path}"}

    try:
        # Run extractions
        results, initial_consensus = run_all_extractions(pdf_path)

        # Process results
        all_results, successful_extractions, consensus_data, metadata = (
            process_extraction_results(results, pdf_path)
        )

        # If we have initial consensus but no calculated consensus, use the initial one
        if initial_consensus and not consensus_data.get("biomarkers"):
            consensus_data = initial_consensus

            # Save this consensus
            timestamp = metadata.get("extraction_timestamp", get_timestamp())
            consensus_file = f"statistical_consensus_{timestamp}.json"
            save_json_data(consensus_data, "outputs/consensus_data", consensus_file)
            logger.info(
                f"Saved initial consensus to outputs/consensus_data/{consensus_file}"
            )

            # Update metadata
            metadata["consensus_file"] = consensus_file

        return all_results, successful_extractions, consensus_data, metadata

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return [], [], {}, {"error": f"Extraction error: {str(e)}"}
