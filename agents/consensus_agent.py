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


def run_consensus_analysis(
    extractions: List[Dict[str, Any]], model_type: str = "gpt-4"
) -> Dict[str, Any]:
    """Run consensus analysis on multiple extractions using LLM"""
    try:
        # Create the consensus agent
        agent = create_consensus_agent(model_type)

        # Prepare the input data
        input_data = {
            "extractions": extractions,
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

                # Add metadata
                llm_consensus["metadata"] = {
                    "total_models": len(extractions),
                    "successful_models": len(
                        [
                            e
                            for e in extractions
                            if e.get("_metadata", {}).get("status") == "success"
                        ]
                    ),
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                }

                return llm_consensus
            else:
                raise ValueError("No valid JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"LLM consensus analysis failed: {str(e)}")
        raise
