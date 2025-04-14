import os
import logging
import csv
from typing import Dict, Any, List, Tuple
from utils.file_utils import (
    get_timestamp,
    save_json_data,
    sanitize_biomarkers,
    clean_old_csv_files,
)
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
    extraction_by_agent = {}

    # Process raw extractions but save only to outputs folder
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
            # Add source model information to each biomarker
            if "biomarkers" in output_data and isinstance(
                output_data["biomarkers"], list
            ):
                for biomarker in output_data["biomarkers"]:
                    if isinstance(biomarker, dict):
                        # Add source model information
                        biomarker["source_model"] = model_name

            # Sanitize biomarkers using the utility function
            output_data = sanitize_biomarkers(output_data)
            successful_extractions.append(output_data)

            # Store in extraction_by_agent dictionary for easier access
            extraction_by_agent[model_name] = output_data

            # Save raw extraction (only once with timestamp)
            # IMPORTANT: This is the only place where raw extraction files should be saved
            model_dir = f"outputs/raw_extractions/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            raw_file = (
                f"{timestamp}.json"  # Using only timestamp since folder has model name
            )
            save_json_data(output_data, model_dir, raw_file)
            logger.info(f"Saved raw extraction to {model_dir}/{raw_file}")

            # No CSV for raw extractions

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

            # Pass model information to the consensus calculation
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
            consensus_dir = "outputs/consensus_data"
            os.makedirs(consensus_dir, exist_ok=True)
            consensus_file = f"statistical_consensus_{timestamp}.json"
            save_json_data(consensus_data, consensus_dir, consensus_file)
            logger.info(
                f"Saved statistical consensus to {consensus_dir}/{consensus_file}"
            )

            # No CSV for consensus data - only generating JSON
            # CSV will only be generated for final extraction results
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
        "extraction_by_agent_count": len(extraction_by_agent),
        "parallel_extraction": True,
    }

    # Return all the processed data
    return results, successful_extractions, consensus_data, metadata


def save_biomarkers_as_csv(biomarkers, directory, filename):
    """Save biomarkers as CSV file for easier viewing

    Args:
        biomarkers: List of biomarker dictionaries
        directory: Directory to save to
        filename: Filename to use
    """
    if not biomarkers:
        return

    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    # Extract all possible field names from biomarkers
    fieldnames = set()
    for biomarker in biomarkers:
        if isinstance(biomarker, dict):
            fieldnames.update(biomarker.keys())

    # Ensure critical fields are first in order
    ordered_fields = ["test_name", "value", "unit", "reference_range"]
    # Add remaining fields
    ordered_fields.extend([f for f in fieldnames if f not in ordered_fields])

    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ordered_fields)
            writer.writeheader()
            for biomarker in biomarkers:
                if isinstance(biomarker, dict):
                    writer.writerow(biomarker)
        logger.info(f"Successfully saved biomarkers to CSV: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")


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
    # Clean up any old CSV files from raw extractions and consensus data
    clean_old_csv_files()

    # Validate PDF path
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return [], [], {}, {"error": f"PDF file not found: {pdf_path}"}

    try:
        logger.info(f"Starting parallel extraction on {pdf_path}")
        # Run extractions in parallel
        results, initial_consensus = run_all_extractions(pdf_path)
        logger.info(f"Completed parallel extraction with {len(results)} results")

        # Process results
        all_results, successful_extractions, consensus_data, metadata = (
            process_extraction_results(results, pdf_path)
        )

        # If we have initial consensus but no calculated consensus, use the initial one
        if initial_consensus and not consensus_data.get("biomarkers"):
            consensus_data = initial_consensus

            # Save this consensus
            timestamp = metadata.get("extraction_timestamp", get_timestamp())
            consensus_dir = "outputs/consensus_data"
            os.makedirs(consensus_dir, exist_ok=True)
            consensus_file = f"statistical_consensus_{timestamp}.json"
            save_json_data(consensus_data, consensus_dir, consensus_file)
            logger.info(f"Saved initial consensus to {consensus_dir}/{consensus_file}")

            # No CSV for consensus data - only for final extraction

            # Update metadata
            metadata["consensus_file"] = consensus_file
        else:
            # Update metadata
            metadata["consensus_file"] = consensus_data.get("consensus_file")

        # Create a final combined output file with all the data
        final_output = {
            "status": "success",
            "timestamp": metadata.get("extraction_timestamp", get_timestamp()),
            "pdf_path": pdf_path,
            "extracted_data": successful_extractions,
            "consensus_data": consensus_data,
            "metadata": metadata,
        }

        # Save final output to outputs/final_extraction
        final_dir = "outputs/final_extraction"
        os.makedirs(final_dir, exist_ok=True)
        final_file = f"final_extraction_{metadata.get('extraction_timestamp', get_timestamp())}.json"
        save_json_data(final_output, final_dir, final_file)
        logger.info(f"Saved final combined output to {final_dir}/{final_file}")

        # Generate CSV for final biomarkers
        if consensus_data and consensus_data.get("biomarkers"):
            csv_dir = "outputs/final_extraction/csv"
            os.makedirs(csv_dir, exist_ok=True)
            final_csv = f"final_extraction_{metadata.get('extraction_timestamp', get_timestamp())}.csv"
            save_biomarkers_as_csv(consensus_data["biomarkers"], csv_dir, final_csv)
            logger.info(f"Saved final combined CSV to {csv_dir}/{final_csv}")
            metadata["final_csv"] = (
                f"csv/{final_csv}"  # Use relative path to final_extraction
            )

        return all_results, successful_extractions, consensus_data, metadata

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return [], [], {}, {"error": f"Extraction error: {str(e)}"}
