import uuid
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from langgraph_workflow import run_langgraph_workflow
from utils.accuracy_checker import compare_csvs
from pathlib import Path
import json, csv, shutil

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
EXTRACTED_DIR = Path("data/extracted")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload-report/")
async def upload_report(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse({"error": "File must be a PDF."}, status_code=400)

    file_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"
    
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run LangGraph pipeline
    final_data = run_langgraph_workflow(pdf_path)

    # Save results
    json_path = EXTRACTED_DIR / f"{file_id}.json"
    csv_path = EXTRACTED_DIR / f"{file_id}.csv"

    with open(json_path, "w") as jf:
        json.dump(final_data, jf, indent=2)

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=final_data.keys())
        writer.writeheader()
        writer.writerow(final_data)

    return {"id": file_id, "data": final_data}


@router.post("/verify-accuracy/")
async def verify_accuracy(actual_csv: UploadFile = File(...), extracted_csv: UploadFile = File(...)):
    stats = compare_csvs(actual_csv.file, extracted_csv.file)
    return stats
