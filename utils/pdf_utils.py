import os
import json
import base64
import re
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import io
import logging

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
        logger.error(f"Error cleaning JSON: {str(e)}")
        return json_str
