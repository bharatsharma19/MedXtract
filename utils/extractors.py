import os
import json
import base64
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import fitz  # PyMuPDF
from PIL import Image
import io
import logging

from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    ANTHROPIC_API_KEY,
    CONFIDENCE_THRESHOLD,
)

EXTRACTION_PROMPT = """
You are a highly trained medical AI assistant. Your job is to extract **biomarker lab test results** from the attached lab report PDF. These reports may be text-based or scanned images.

IMPORTANT: You must return ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

---

Step 1: Deep Scan
- Read the full document thoroughly (do at least 3 full passes).
- Detect:
  - Report structure and test sections
  - Languages used (handle multilingual inputs)
  - Image-based vs text-based layout
  - Formatting issues (decimal symbols, spacing, etc)

---

Step 2: Extract Biomarker Data
- Extract **all available test results** as structured data.
- Convert everything to **standard English** medical terminology.
- Include units, reference ranges, and original values.
- Handle data in images, screenshots, or unusual layouts (e.g. tables, charts).
- Convert commas used as decimal separators into periods.

---

Step 3: Output Format
Return ONLY this exact JSON structure (no other text):

{
  "biomarkers": [
    {
      "test_name": "Hemoglobin",
      "value": 13.5,
      "unit": "g/dL",
      "reference_range": "13.0 - 17.0"
    }
  ],
  "notes": [
    "Detected language: French",
    "Converted commas to periods",
    "Interpreted table from scanned image",
    "Standardized test names to English"
  ]
}
"""

logger = logging.getLogger(__name__)


def convert_pdf_to_images(pdf_path: str, max_pages: int = 5) -> List[str]:
    """Convert PDF to base64 encoded images, limiting to first few pages"""
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images.append(img_str)

    doc.close()
    return images


def clean_json_string(json_str: str) -> str:
    """Clean and fix common JSON formatting issues"""
    try:
        # Remove any text before the first { and after the last }
        json_str = re.sub(r"^[^{]*", "", json_str)
        json_str = re.sub(r"[^}]*$", "", json_str)

        # Fix specific issues from Gemini responses
        json_str = json_str.replace('" [', "[")  # Fix array start
        json_str = json_str.replace('] "', "]")  # Fix array end
        json_str = json_str.replace('" "', '"')  # Fix double quotes
        json_str = json_str.replace('"null', "null")  # Fix null values
        json_str = json_str.replace('null"', "null")  # Fix null values

        # Remove extra quotes around values
        json_str = re.sub(r':\s*"([^"]*)"', r': "\1"', json_str)

        # Fix trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix unquoted keys
        json_str = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str)

        # Fix string values that should be numbers
        json_str = re.sub(r':\s*"(\d+\.?\d*)"\s*([,}])', r": \1\2", json_str)

        # Fix null values in reference ranges
        json_str = re.sub(r':\s*"null"', ": null", json_str)

        # Remove extra spaces
        json_str = re.sub(r"\s+", " ", json_str)

        # Parse and re-stringify to validate
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        print(f"Error cleaning JSON: {str(e)}")
        return json_str


def create_model_messages(pdf_path: str, model_type: str) -> List[Dict[str, Any]]:
    """Create appropriate messages based on model type"""
    if model_type == "gemini":
        # For Gemini, we'll use text extraction first
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # For Gemini, be more explicit about JSON formatting
        gemini_prompt = (
            EXTRACTION_PROMPT
            + "\n\nIMPORTANT: Return ONLY the JSON object, with no additional text or formatting."
        )

        return [
            SystemMessage(content=gemini_prompt),
            HumanMessage(
                content=f"Please analyze this lab report text and return ONLY a valid JSON object:\n\n{text[:8000]}"
            ),
        ]
    else:
        # For other models, use images
        images = convert_pdf_to_images(pdf_path)
        messages = [{"type": "text", "text": EXTRACTION_PROMPT}]

        for img in images:
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
            )

        return messages


def extract_with_model(model, pdf_path: str) -> Dict[str, Any]:
    """Extract data from PDF using specified model"""
    try:
        # Validate PDF file
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")

        # Get file size
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise ValueError("PDF file is empty")

        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        if not images:
            raise ValueError("No valid pages found in PDF")

        # Create model messages
        model_type = "gemini" if isinstance(model, ChatGoogleGenerativeAI) else "other"
        messages = create_model_messages(pdf_path, model_type)

        # Get response from model
        try:
            response = model.invoke(messages)
            if not response:
                raise ValueError("Empty response from model")
        except Exception as e:
            logger.error(f"Model invocation failed: {str(e)}")
            raise

        # Parse the response
        result = parse_model_response(response)

        # Validate the result
        if not result or not isinstance(result, dict):
            raise ValueError("Invalid response format")

        if "biomarkers" not in result:
            result["biomarkers"] = []

        if "notes" not in result:
            result["notes"] = []

        # Add extraction metadata
        result["_metadata"] = {
            "model_type": model_type,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "success" if result["biomarkers"] else "no_data",
            "page_count": len(images),
            "file_size": file_size,
        }

        return result

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return {
            "biomarkers": [],
            "notes": [f"Extraction failed: {str(e)}"],
            "_metadata": {
                "model_type": "unknown",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "status": "failed",
                "error": str(e),
            },
        }


def parse_model_response(response) -> Dict[str, Any]:
    """Parse and validate model response"""
    try:
        # Handle different response formats
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict):
            content = response.get("text", "")
        else:
            content = str(response)

        # Try to find JSON in the response
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1

        if start_idx != -1 and end_idx != 0:
            json_str = content[start_idx:end_idx]
            # Clean and fix JSON formatting
            cleaned_json = clean_json_string(json_str)
            try:
                parsed = json.loads(cleaned_json)
                # Ensure the response has the required structure
                if not isinstance(parsed, dict):
                    return {"biomarkers": [], "notes": ["Invalid response format"]}

                # Ensure biomarkers array exists
                if "biomarkers" not in parsed:
                    parsed["biomarkers"] = []

                # Ensure notes array exists
                if "notes" not in parsed:
                    parsed["notes"] = []

                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Cleaned JSON string: {cleaned_json}")
                return {"biomarkers": [], "notes": [f"JSON parsing error: {str(e)}"]}
        else:
            logger.error("No valid JSON found in response")
            logger.error(f"Raw response: {content}")
            return {"biomarkers": [], "notes": ["No valid JSON found in response"]}
    except Exception as e:
        logger.error(f"Error parsing model response: {str(e)}")
        return {"biomarkers": [], "notes": [f"Error parsing response: {str(e)}"]}


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_consensus(extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate consensus between different model extractions"""
    if not extractions:
        return {}

    # Group biomarkers by test name
    biomarker_groups = {}
    for extraction in extractions:
        if not extraction or not isinstance(extraction, dict):
            continue

        biomarkers = extraction.get("biomarkers", [])
        if not isinstance(biomarkers, list):
            continue

        for biomarker in biomarkers:
            if not biomarker or not isinstance(biomarker, dict):
                continue

            test_name = biomarker.get("test_name")
            if test_name is None:
                continue

            # Clean and normalize test name
            try:
                test_name = str(test_name).strip().lower()
                if not test_name:
                    continue
            except (AttributeError, TypeError):
                continue

            if test_name not in biomarker_groups:
                biomarker_groups[test_name] = []
            biomarker_groups[test_name].append(biomarker)

    # Calculate consensus for each biomarker
    consensus_data = {"biomarkers": [], "notes": []}
    confidence_scores = {}

    for test_name, biomarkers in biomarker_groups.items():
        if len(biomarkers) < 2:  # Need at least 2 models to calculate consensus
            if biomarkers:
                consensus_data["biomarkers"].append(biomarkers[0])
                confidence_scores[test_name] = 0.5
            continue

        # Calculate value consensus
        values = []
        for b in biomarkers:
            try:
                value = b.get("value")
                if value is not None:
                    values.append(value)
            except (AttributeError, TypeError):
                continue

        if values:
            try:
                numeric_values = []
                for v in values:
                    try:
                        if isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        elif isinstance(v, str):
                            cleaned = v.replace(".", "").replace(",", ".")
                            if cleaned.replace(".", "").isdigit():
                                numeric_values.append(float(cleaned))
                    except (ValueError, TypeError):
                        continue

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
            except Exception as e:
                logger.error(f"Error calculating value consensus: {str(e)}")
                avg_value = values[0]
                confidence = 0.5
        else:
            avg_value = None
            confidence = 0

        # Calculate unit consensus
        units = []
        for b in biomarkers:
            try:
                unit = b.get("unit")
                if unit is not None:
                    units.append(str(unit).strip().lower())
            except (AttributeError, TypeError):
                continue
        unit_consensus = max(set(units), key=units.count) if units else ""

        # Calculate reference range consensus
        ranges = []
        for b in biomarkers:
            try:
                ref_range = b.get("reference_range")
                if ref_range is not None:
                    ranges.append(str(ref_range).strip())
            except (AttributeError, TypeError):
                continue
        range_consensus = max(set(ranges), key=ranges.count) if ranges else ""

        # Only include if confidence meets threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            try:
                consensus_data["biomarkers"].append(
                    {
                        "test_name": biomarkers[0].get("test_name", test_name),
                        "value": avg_value,
                        "unit": unit_consensus,
                        "reference_range": range_consensus,
                        "confidence": confidence,
                        "model_count": len(biomarkers),
                    }
                )
                confidence_scores[test_name] = confidence
            except Exception as e:
                logger.error(f"Error creating consensus biomarker: {str(e)}")

    consensus_data["_confidence_scores"] = confidence_scores
    consensus_data["_model_count"] = len(extractions)
    consensus_data["_timestamp"] = get_timestamp()

    return consensus_data


def run_all_extractions(pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run extractions with all available models and calculate consensus"""
    agents = []
    results = []
    timestamp = get_timestamp()

    # Create output directories
    os.makedirs("outputs/raw_extractions", exist_ok=True)
    os.makedirs("outputs/final_extraction", exist_ok=True)

    # Initialize models only if API keys are available
    try:
        if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_google_api_key":
            gemini_models = ["gemini-1.5-pro", "gemini-2.0-flash"]
            for model_id in gemini_models:
                try:
                    agents.append(
                        {
                            "name": f"google_{model_id}",
                            "model": ChatGoogleGenerativeAI(
                                model=model_id,
                                google_api_key=GOOGLE_API_KEY,
                                temperature=0,
                                max_output_tokens=8192,
                            ),
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize Gemini model {model_id}: {str(e)}"
                    )

        if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key":
            try:
                agents.append(
                    {
                        "name": "openai_gpt4_vision",
                        "model": ChatOpenAI(
                            model="gpt-4-vision-preview",
                            api_key=OPENAI_API_KEY,
                            temperature=0,
                            max_tokens=8192,
                        ),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to initialize GPT-4 Vision: {str(e)}")

        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key":
            try:
                agents.append(
                    {
                        "name": "claude_opus",
                        "model": ChatAnthropic(
                            model="claude-3-opus",
                            api_key=ANTHROPIC_API_KEY,
                            temperature=0,
                            max_tokens=8192,
                        ),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {str(e)}")

        if not agents:
            raise ValueError(
                "No valid API keys provided. Please check your configuration."
            )

        # Run extractions with available models
        for agent in agents:
            model_name = agent["name"]
            model = agent["model"]

            logger.info(f"Running extraction with {model_name}")
            try:
                output = extract_with_model(model, pdf_path)

                # Create result object
                result = {
                    "model": model_name,
                    "output": output,
                    "timestamp": timestamp,
                    "status": output.get("_metadata", {}).get("status", "unknown"),
                }

                # Add error if present
                if "error" in output.get("_metadata", {}):
                    result["error"] = output["_metadata"]["error"]

                results.append(result)

                # Create model-specific directory
                model_dir = f"outputs/raw_extractions/{model_name}"
                os.makedirs(model_dir, exist_ok=True)

                # Save output to file with timestamp
                output_path = f"{model_dir}/{timestamp}.json"
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
                logger.info(f"Saved output to {output_path}")

            except Exception as e:
                logger.error(f"Error in {model_name} extraction: {str(e)}")
                results.append(
                    {
                        "model": model_name,
                        "output": {
                            "biomarkers": [],
                            "notes": [f"Extraction error: {str(e)}"],
                            "_metadata": {"status": "failed", "error": str(e)},
                        },
                        "timestamp": timestamp,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Calculate consensus if we have multiple successful models
        successful_extractions = [
            r["output"] for r in results if r.get("status") == "success"
        ]
        consensus_data = {}
        if len(successful_extractions) >= 2:
            consensus_data = calculate_consensus(successful_extractions)
            if consensus_data:
                # Save consensus data
                consensus_path = (
                    f"outputs/final_extraction/statistical_consensus_{timestamp}.json"
                )
                with open(consensus_path, "w") as f:
                    json.dump(consensus_data, f, indent=2)
                logger.info(f"Saved consensus data to {consensus_path}")

        return results, consensus_data

    except Exception as e:
        logger.error(f"Error in run_all_extractions: {str(e)}")
        return [], {}
