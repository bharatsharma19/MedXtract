from typing import Dict, Any, List, Tuple
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from datetime import datetime
import os
import json
import fitz
from utils.pdf_utils import convert_pdf_to_images, clean_json_string
from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    ANTHROPIC_API_KEY,
)
import concurrent.futures

logger = logging.getLogger(__name__)

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


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def extract_with_model(model, pdf_path: str, model_name: str) -> Dict[str, Any]:
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

        # Add source model information to each biomarker
        if "biomarkers" in result and isinstance(result["biomarkers"], list):
            for biomarker in result["biomarkers"]:
                if isinstance(biomarker, dict):
                    biomarker["source_model"] = model_name

        # Add extraction metadata
        result["_metadata"] = {
            "model_type": model_name,
            "timestamp": get_timestamp(),
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
                "model_type": model_name,
                "timestamp": get_timestamp(),
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

        # Clean and parse JSON
        json_str = clean_json_string(content)
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to parse model response: {str(e)}")
        raise


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

        # Function to process a single model extraction
        def process_model_extraction(agent):
            model_name = agent["name"]
            model = agent["model"]

            logger.info(f"Running extraction with {model_name}")
            try:
                output = extract_with_model(model, pdf_path, model_name)

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

                # Note: No longer saving files here, this will be handled by extraction_utils.py
                return result
            except Exception as e:
                logger.error(f"Error in {model_name} extraction: {str(e)}")
                return {
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

        # Run extractions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
            # Submit all extraction tasks
            future_to_agent = {
                executor.submit(process_model_extraction, agent): agent
                for agent in agents
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Extraction with {agent['name']} failed with error: {str(e)}"
                    )
                    results.append(
                        {
                            "model": agent["name"],
                            "output": {
                                "biomarkers": [],
                                "notes": [f"Exception during extraction: {str(e)}"],
                                "_metadata": {"status": "failed", "error": str(e)},
                            },
                            "timestamp": timestamp,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

        # Prepare initial consensus data (will be calculated fully in extraction_utils)
        initial_consensus = {
            "biomarkers": [],
            "notes": ["Initial consensus placeholder - to be calculated"],
            "_timestamp": timestamp,
        }

        return results, initial_consensus

    except Exception as e:
        logger.error(f"Error in run_all_extractions: {str(e)}")
        return [], {}
