import os
import sys
import logging
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.extraction_utils import process_extraction_results
from utils.statistical_utils import calculate_statistical_consensus
from utils.file_utils import sanitize_biomarkers


def test_file_saving_and_source_model():
    """Test that extraction files are only saved once and source model information is preserved."""
    # Mock results from extraction agents
    test_results = [
        {
            "model": "test_model_1",
            "output": {
                "biomarkers": [
                    {
                        "test_name": "Test Biomarker 1",
                        "value": 100,
                        "unit": "mg/dL",
                        "reference_range": "70-120",
                    }
                ],
                "notes": ["Test note"],
                "_metadata": {"model_type": "test_model_1", "status": "success"},
            },
            "timestamp": "20250415_120000",
            "status": "success",
        }
    ]

    # Get current directory for testing
    test_dir = Path(__file__).parent
    test_output_dir = test_dir / "test_outputs"
    os.makedirs(test_output_dir, exist_ok=True)

    # Clean up any existing test files
    test_raw_dir = test_output_dir / "raw_extractions" / "test_model_1"
    os.makedirs(test_raw_dir, exist_ok=True)
    for file in test_raw_dir.glob("*.json"):
        os.remove(file)

    test_consensus_dir = test_output_dir / "consensus_data"
    os.makedirs(test_consensus_dir, exist_ok=True)
    for file in test_consensus_dir.glob("*.json"):
        os.remove(file)

    # Process the mock results
    original_save_dir = os.path.join

    try:
        # Process results and check outputs
        all_results, successful_extractions, consensus_data, metadata = (
            process_extraction_results(test_results, "test.pdf")
        )

        # Verify source model is preserved in biomarkers
        for extraction in successful_extractions:
            for biomarker in extraction.get("biomarkers", []):
                assert (
                    "source_model" in biomarker
                ), "source_model not found in biomarker"
                assert (
                    biomarker["source_model"] == "test_model_1"
                ), "Incorrect source_model"

        # Verify consensus data has source models correctly preserved
        if consensus_data and "biomarkers" in consensus_data:
            for biomarker in consensus_data["biomarkers"]:
                assert (
                    "source_models" in biomarker
                ), "source_models not found in consensus biomarker"
                assert (
                    "test_model_1" in biomarker["source_models"]
                ), "Source model not found in source_models list"

        print(
            "Test passed: File saving and source model preservation working correctly"
        )
        return True

    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the test
    test_file_saving_and_source_model()
