import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


def get_api_key(env_var: str, default: str) -> Optional[str]:
    """Get API key from environment variable or return None if not set"""
    key = os.getenv(env_var, default)
    return key if key and key != default else None


# Constants for the application
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your_anthropic_api_key")

# Model Configuration
MODEL_TEMPERATURE = 0
MAX_TOKENS = 8192

# Consensus Configuration
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score for consensus (0.0 to 1.0)

# File paths
REPORT_FILE_PATH = os.getenv(
    "REPORT_FILE_PATH", os.path.join("uploads", "sample_report.pdf")
)
