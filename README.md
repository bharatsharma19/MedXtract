# 🧪 Lab Report Extraction System

A smart, fast, and accurate system to extract structured data from unstructured PDF lab reports using LangGraph, FastAPI, and multiple AI agents. Built to handle real-world lab report messiness and ensure consensus-driven data integrity.

## 🚀 Features

*   📦 **Upload + Extract**: Upload PDF lab reports and automatically extract data into structured CSV format.
*   🧠 **LangGraph Integration**: Uses LangGraph for orchestrating data extraction workflows with multi-agent coordination.
*   ✅ **Accuracy Validation**: Compares actual vs extracted CSV data to ensure precision.
*   👥 **Consensus Agent**: Aggregates output from multiple models to agree on the best data.
*   🔍 **PDF Utils**: Converts PDFs to images and normalizes messy extracted data.
*   📊 **Data Validators**: Enforces consistency and quality using custom validators.
*   🐛 **Comprehensive Logging**: Built-in logs to trace errors and simplify debugging.

## 🛠️ Tech Stack

*   **Python 3.10+**
*   **FastAPI** – lightning-fast API layer
*   **LangGraph** – workflow + multi-agent coordination
*   **Pandas** – for CSV handling and accuracy checks
*   **OpenCV / PDF Libraries** – for PDF-to-image conversion
*   **CORS Middleware** – for secure API usage

## 📂 Project Structure

```
MedXtract/
├── data/                    # For storing sample data
├── outputs/                 # Processed outputs
├── tests/                   # Test scripts
├── utils/                   # Helper utilities
│   ├── accuracy_checker.py  # Validation of extraction accuracy
│   ├── consensus_agent.py   # Agent for merging multiple outputs
│   ├── extractors.py        # PDF data extraction logic
│   ├── normalizer.py        # Data normalization utilities
│   └── validators.py        # Data quality validation
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

### Try it out

Hit `http://localhost:8000/docs` for the interactive Swagger UI.  
Upload a lab report and see the magic happen ✨

## 🧪 Example Workflow

1.  Upload a PDF.
2.  LangGraph orchestrates model agents to extract fields.
3.  Consensus agent merges outputs.
4.  Validators clean and ensure quality.
5.  Output: a clean, verified CSV.

## 📌 TODOs

*   Move agents to dedicated modules (currently in utils/)
*   Add support for more lab report formats
*   Integrate with EMR systems
*   Add frontend for upload + results view
*   Benchmark agent accuracy

## 🤝 Contributing

Pull requests are welcome! Let's make this the go-to open source for lab data automation.

## 📝 License

MIT – do what you want, just don't sue. Attribution is appreciated.