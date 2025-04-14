import requests
import os
import sys

# Add parent directory to path to access config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from config, but provide fallback
try:
    from config import REPORT_FILE_PATH
except (ImportError, AttributeError):
    # Default path if import fails
    REPORT_FILE_PATH = os.path.join(os.path.dirname(__file__), "sample_report.pdf")

# Check if report path is provided as argument
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = REPORT_FILE_PATH

# Verify file exists
if not os.path.exists(file_path):
    print(f"Error: Report file not found at {file_path}")
    print("Usage: python upload.py [path_to_pdf_file]")
    sys.exit(1)

files = {
    "file": (os.path.basename(file_path), open(file_path, "rb"), "application/pdf")
}

try:
    res = requests.post("http://127.0.0.1:8000/upload-report/", files=files)
    print(f"Status Code: {res.status_code}")
    print("Response:")
    print(res.json())
except Exception as e:
    print(f"Error: {str(e)}")
