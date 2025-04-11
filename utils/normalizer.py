from datetime import datetime
from typing import Dict, Any, Optional
import re

# Standard units and their conversions
UNIT_CONVERSIONS = {
    "g/dL": {"g/L": lambda x: x * 10, "g/100mL": lambda x: x, "g%": lambda x: x},
    "million/uL": {
        "million/mm3": lambda x: x,
        "10^6/uL": lambda x: x,
        "10^6/mm3": lambda x: x,
        "M/uL": lambda x: x,
    },
}

# Standard biomarker names and their common variations
BIOMARKER_ALIASES = {
    "Hemoglobin": ["Hb", "HGB", "Haemoglobin", "HB", "Hgb"],
    "RBC Count": ["Red Blood Cells", "RBC", "Erythrocytes", "Red Cell Count", "RBCs"],
    "WBC Count": ["White Blood Cells", "WBC", "Leukocytes", "White Cell Count", "WBCs"],
    "Platelets": ["PLT", "Thrombocytes", "Platelet Count", "PLAT"],
    "Hematocrit": ["HCT", "PCV", "Packed Cell Volume", "Hct"],
    "MCV": ["Mean Corpuscular Volume", "Mean Cell Volume"],
    "MCH": ["Mean Corpuscular Hemoglobin", "Mean Cell Hemoglobin"],
    "MCHC": ["Mean Corpuscular Hemoglobin Concentration"],
}


def extract_numeric_value(value_str: str) -> Optional[float]:
    if not value_str:
        return None
    try:
        # Extract numeric value using regex
        matches = re.findall(r"[-+]?\d*\.?\d+", str(value_str))
        if matches:
            return float(matches[0])
        return None
    except (ValueError, TypeError):
        return None


def extract_unit(value_str: str) -> Optional[str]:
    if not value_str:
        return None
    # Common unit patterns
    unit_patterns = [
        r"(?:g/dL|g/L|g/100mL|g%)",
        r"(?:million/[uµ]L|million/mm3|10\^6/[uµ]L|M/[uµ]L)",
        r"(?:K/[uµ]L|thousand/[uµ]L|10\^3/[uµ]L)",
        r"(?:pg|fL|%|g/L)",
    ]
    pattern = "|".join(unit_patterns)
    match = re.search(pattern, value_str, re.IGNORECASE)
    return match.group() if match else None


def normalize_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None

    date_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%b %d, %Y",
        "%d %b %Y",
        "%Y %b %d",
        "%B %d, %Y",
        "%d %B %Y",
        "%Y %B %d",
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except:
            continue
    return None


def standardize_biomarker_name(name: str) -> str:
    name_lower = name.lower().strip()
    for standard_name, aliases in BIOMARKER_ALIASES.items():
        if name_lower == standard_name.lower() or name_lower in [
            alias.lower() for alias in aliases
        ]:
            return standard_name
    return name


def normalize_and_map(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes and standardizes lab report data with enhanced unit conversion
    and biomarker name standardization.
    """
    normalized = {}

    # Process each biomarker
    for key, value in data.items():
        if key == "_confidence_scores":
            normalized["_confidence_scores"] = value
            continue

        # Skip empty values
        if not value:
            continue

        # Standardize biomarker name
        std_name = standardize_biomarker_name(key)

        # Extract numeric value and unit
        numeric_value = extract_numeric_value(value)
        unit = extract_unit(value)

        if numeric_value is not None:
            normalized[std_name] = {
                "value": numeric_value,
                "unit": unit,
                "original_value": value,
            }

            # Add reference ranges if available
            if isinstance(value, dict) and "reference_range" in value:
                normalized[std_name]["reference_range"] = value["reference_range"]

    # Add test date if available
    if "Test Date" in data:
        normalized["Test Date"] = normalize_date(data["Test Date"])

    return normalized
