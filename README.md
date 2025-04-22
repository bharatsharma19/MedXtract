# 🧪 Lab Report Extraction System

A smart, fast, and accurate system to extract structured data from unstructured PDF lab reports using LangGraph, FastAPI, and multiple AI agents. Built to handle real-world lab report messiness and ensure consensus-driven data integrity.

## 🚀 Features

- 📦 **Upload + Extract**: Upload PDF lab reports and automatically extract data into structured JSON and CSV format.
- 🧠 **LangGraph Integration**: Uses LangGraph for orchestrating data extraction workflows with multi-agent coordination.
- ✅ **Accuracy Validation**: Compares actual vs extracted CSV data to ensure precision.
- 👥 **Consensus Agent**: Aggregates output from multiple models to agree on the best data.
- 🔍 **PDF Utils**: Converts PDFs to images and normalizes messy extracted data.
- 📊 **Data Validators**: Enforces consistency and quality using custom validators.
- 🐛 **Comprehensive Logging**: Built-in logs to trace errors and simplify debugging.
- ⚡ **Parallel Extraction**: Runs all extraction models simultaneously for faster processing.
- 🔄 **Optimized Data Flow**: Stores extraction results by agent for better traceability and consensus generation.
- 📈 **Statistical Analysis**: Provides detailed statistics on extraction quality and consensus reliability.

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastAPI** – lightning-fast API layer
- **LangGraph** – workflow + multi-agent coordination
- **Pandas** – for CSV handling and accuracy checks
- **OpenCV / PDF Libraries** – for PDF-to-image conversion
- **CORS Middleware** – for secure API usage
- **Concurrent Futures** – for parallel processing of extractions

## 📂 Project Structure

```
MedXtract/
├── uploads/                 # For storing uploaded PDFs
├── outputs/                 # Processed outputs
│   ├── consensus_data/      # Consensus results from multiple models
│   ├── final_extraction/    # Final processed extraction results
│   │   ├── csv/             # CSV outputs for easier viewing
│   ├── raw_extractions/     # Raw extraction outputs from each model
├── tests/                   # Test scripts
│   ├── upload.py            # Script to test the upload endpoint
├── utils/                   # Helper utilities
│   ├── accuracy_checker.py  # Validation of extraction accuracy
│   ├── consensus_utils.py   # Utilities for merging multiple outputs
│   ├── extraction_utils.py  # PDF data extraction logic
│   ├── file_utils.py        # File handling utilities
│   ├── normalizer.py        # Data normalization utilities
│   ├── pdf_utils.py         # PDF processing utilities
│   ├── response_utils.py    # API response formatting
│   ├── state_utils.py       # LangGraph state management
│   ├── statistical_utils.py # Statistical consensus calculation
│   ├── validation_utils.py  # Data validation utilities
│   ├── validators.py        # Data validators
│   ├── workflow_nodes.py    # LangGraph workflow nodes
├── agents/                  # Extraction agents
│   ├── extraction_agent.py  # Parallel extraction implementation
│   ├── consensus_agent.py   # Consensus generation between models
├── .env                     # Environment variables
├── .env.example             # Example env configuration
├── .gitignore               # Git ignore file
├── config.py                # Configuration settings
├── langgraph_workflow.py    # LangGraph workflow definition
├── main.py                  # Application entry point
├── requirements.txt         # Project dependencies
├── routes.py                # API endpoints
└── README.md                # This file
```

## ⚙️ Setup & Run

### Clone the repo

```
git clone https://github.com/bharatsharma19/MedXtract.git
cd MedXtract
```

### Setup environment variables

```
cp .env.example .env
# Edit .env with your configuration values
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the app

```
python main.py
```

The server will start at `http://localhost:8000`.

### Running tests

To test the upload functionality:

```
cd MedXtract
python tests/upload.py [path_to_pdf_file]
```

If you don't provide a path, it will look for the file specified in `config.py` as `REPORT_FILE_PATH`.

You can also set the path in your .env file:

```
REPORT_FILE_PATH=path/to/your/lab/report.pdf
```

### Using the API

Hit `http://localhost:8000/docs` for the interactive Swagger UI.

#### POST /upload-report/

Upload a lab report PDF and extract structured data:

```
POST /upload-report/
```

Response format:

```
{
  "id": "20231120_123456_abc12345",
  "data": {
    "biomarkers": [
      {
        "test_name": "Hemoglobin",
        "value": 13.5,
        "unit": "g/dL",
        "reference_range": "13.0 - 17.0"
      }
    ]
  },
  "statistics": {
    "extraction_success_rate": 0.95,
    "consensus_confidence": 0.87,
    "model_agreement": 0.91
  },
  "files": {
    "pdf": "uploads/20231120_123456_abc12345.pdf",
    "json_output": "outputs/final_extraction/final_extraction_20231120_123456.json",
    "consensus": "outputs/consensus_data/statistical_consensus_20231120_123456.json",
    "csv_output": "outputs/final_extraction/csv/final_result_20231120_123456.csv"
  },
  "metadata": {
    "extraction_timestamp": "20231120_123456",
    "total_models": 3,
    "successful_models": 3
  }
}
```

#### POST /verify-accuracy/

Compare actual vs. extracted CSV data to validate extraction:

```
POST /verify-accuracy/
```

## 🧪 Example Workflow

1.  Upload a PDF.
2.  LangGraph orchestrates parallel extraction using multiple model agents simultaneously.
3.  Extraction results are stored by agent for better traceability.
4.  Consensus agent merges outputs from all successful extractions.
5.  Validators clean and ensure quality.
6.  Output: a clean, verified JSON/CSV with detailed extraction metrics and statistics.

## 📈 API Response Statistics

The API now includes detailed statistics about the extraction process:

- **extraction_success_rate**: Percentage of successful extractions from all models
- **consensus_confidence**: Average confidence score of the consensus decisions
- **model_agreement**: Level of agreement between different extraction models
- **biomarker_confidence**: Confidence scores for individual biomarkers

## 📌 TODOs

- Add support for more lab report formats
- Integrate with EMR systems
- Add frontend for upload + results view
- Benchmark agent accuracy
- Add caching for repeated extractions

## 🤝 Contributing

Pull requests are welcome! Let's make this the go-to open source for lab data automation.

## 📝 License

MIT – do what you want, just don't sue. Attribution is appreciated.
