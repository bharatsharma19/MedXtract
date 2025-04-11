from typing import List, Dict, Any
import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    ANTHROPIC_API_KEY,
    CONFIDENCE_THRESHOLD,
)
from datetime import datetime

logger = logging.getLogger(__name__)

CONSENSUS_PROMPT = """
You are a medical data validation expert. Your task is to analyze multiple extractions of lab report data and determine the most accurate values.

IMPORTANT: You must return ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

---

Step 1: Analyze Extractions
- Review all provided extractions
- Identify common patterns and values
- Note any discrepancies or outliers
- Consider the source model's reliability

Step 2: Determine Consensus
- For each biomarker:
  - Compare values across extractions
  - Consider units and reference ranges
  - Calculate statistical confidence
  - Identify the most reliable value

Step 3: Output Format
Return ONLY this exact JSON structure (no other text):

{
  "biomarkers": [
    {
      "test_name": "Hemoglobin",
      "value": 13.5,
      "unit": "g/dL",
      "reference_range": "13.0 - 17.0",
      "confidence": 0.95,
      "source_models": ["model1", "model2"]
    }
  ],
  "notes": [
    "High confidence in hemoglobin value (95%)",
    "Discrepancy in WBC count resolved using majority vote"
  ],
  "metadata": {
    "total_models": 3,
    "successful_models": 2,
    "confidence_threshold": 0.8
  }
}
"""


def create_consensus_agent(model_type: str):
    """Create the appropriate consensus agent based on model type"""
    try:
        if (
            model_type == "gpt-4"
            and OPENAI_API_KEY
            and OPENAI_API_KEY != "your_openai_api_key"
        ):
            return ChatOpenAI(
                model="gpt-4",
                api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=4096,
            )
        elif (
            model_type == "gemini"
            and GOOGLE_API_KEY
            and GOOGLE_API_KEY != "your_google_api_key"
        ):
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
                max_output_tokens=4096,
            )
        elif (
            model_type == "claude"
            and ANTHROPIC_API_KEY
            and ANTHROPIC_API_KEY != "your_anthropic_api_key"
        ):
            return ChatAnthropic(
                model="claude-3-opus",
                api_key=ANTHROPIC_API_KEY,
                temperature=0,
                max_tokens=4096,
            )
        else:
            raise ValueError(f"Unsupported model type or missing API key: {model_type}")
    except Exception as e:
        logger.error(f"Error creating consensus agent: {str(e)}")
        raise


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
            consensus_data["biomarkers"].append(biomarkers[0])
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

        # Only include if confidence meets threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            consensus_data["biomarkers"].append(
                {
                    "test_name": biomarkers[0]["test_name"],  # Use original casing
                    "value": avg_value,
                    "unit": unit_consensus,
                    "reference_range": range_consensus,
                    "confidence": confidence,
                    "source_models": [
                        b.get("_metadata", {}).get("model_type", "unknown")
                        for b in biomarkers
                    ],
                }
            )
            confidence_scores[test_name] = confidence

    consensus_data["_confidence_scores"] = confidence_scores
    consensus_data["_model_count"] = len(extractions)
    consensus_data["_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    return consensus_data


def run_consensus_analysis(
    extractions: List[Dict[str, Any]], model_type: str = "gpt-4"
) -> Dict[str, Any]:
    """Run consensus analysis on multiple extractions using LLM"""
    try:
        # First calculate statistical consensus
        statistical_consensus = calculate_statistical_consensus(extractions)

        # Create the consensus agent
        agent = create_consensus_agent(model_type)

        # Prepare the input data
        input_data = {
            "extractions": extractions,
            "statistical_consensus": statistical_consensus,
            "total_models": len(extractions),
            "model_names": [
                ext.get("_metadata", {}).get("model_type", "unknown")
                for ext in extractions
            ],
        }

        # Create messages for the agent
        messages = [
            SystemMessage(content=CONSENSUS_PROMPT),
            HumanMessage(content=json.dumps(input_data, indent=2)),
        ]

        # Get response from the agent
        response = agent.invoke(messages)

        # Parse and return the response
        try:
            # Try to find JSON in the response
            content = response.content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                llm_consensus = json.loads(json_str)

                # Merge statistical and LLM consensus
                final_consensus = {
                    "biomarkers": [],
                    "notes": [],
                    "metadata": {
                        "total_models": len(extractions),
                        "successful_models": len(
                            [
                                e
                                for e in extractions
                                if e.get("_metadata", {}).get("status") == "success"
                            ]
                        ),
                        "confidence_threshold": CONFIDENCE_THRESHOLD,
                        "consensus_method": "combined",
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    },
                }

                # Merge biomarkers from both consensus methods
                biomarker_map = {}
                for biomarker in statistical_consensus.get("biomarkers", []):
                    test_name = biomarker["test_name"].lower()
                    biomarker_map[test_name] = biomarker

                for biomarker in llm_consensus.get("biomarkers", []):
                    test_name = biomarker["test_name"].lower()
                    if test_name in biomarker_map:
                        # Use higher confidence value
                        if biomarker.get("confidence", 0) > biomarker_map[
                            test_name
                        ].get("confidence", 0):
                            biomarker_map[test_name] = biomarker
                    else:
                        biomarker_map[test_name] = biomarker

                final_consensus["biomarkers"] = list(biomarker_map.values())
                final_consensus["notes"] = llm_consensus.get("notes", [])

                return final_consensus
            else:
                raise ValueError("No valid JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing consensus response: {str(e)}")
            logger.error(f"Raw response: {response.content}")
            return statistical_consensus

    except Exception as e:
        logger.error(f"Error in consensus analysis: {str(e)}")
        return calculate_statistical_consensus(extractions)
