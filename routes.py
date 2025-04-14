import uuid
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from langgraph_workflow import run_langgraph_workflow
from utils.accuracy_checker import compare_csvs
from pathlib import Path
import json, csv, shutil
import os
from datetime import datetime

router = APIRouter()

# Use only outputs folder for extracted files and uploads for uploads
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs/final_extraction")

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_timestamp():
    """Return a formatted timestamp string for file naming and tracking"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@router.post("/upload-report/")
async def upload_report(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse({"error": "File must be a PDF."}, status_code=400)

    # Generate a unique file ID with timestamp
    timestamp = get_timestamp()
    file_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"

    # Save the uploaded PDF
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run LangGraph pipeline
    final_data = run_langgraph_workflow(pdf_path)

    # All extraction data is now saved by the workflow in the outputs folder
    # No need to save again in a separate location

    # Return the file ID and data for API response
    response = {
        "id": file_id,
        "data": final_data.get("data", {}),
        "statistics": final_data.get("statistics", {}),
        "files": {
            "pdf": str(pdf_path),
            "json_output": f"outputs/final_extraction/final_extraction_{final_data.get('metadata', {}).get('extraction_timestamp', timestamp)}.json",
            "consensus": f"outputs/consensus_data/{final_data.get('metadata', {}).get('consensus_file', '')}",
            "csv_output": f"outputs/final_extraction/{final_data.get('metadata', {}).get('final_csv', '')}",
        },
        "metadata": final_data.get("metadata", {}),
    }

    return response


@router.post("/verify-accuracy/")
async def verify_accuracy(
    actual_csv: UploadFile = File(...), extracted_csv: UploadFile = File(...)
):
    stats = compare_csvs(actual_csv.file, extracted_csv.file)
    return stats
