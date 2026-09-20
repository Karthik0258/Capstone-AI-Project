# Capstone-Project-AI-Project
This repository contains three integrated modules that together demonstrate a full analyst‑to‑data‑scientist workflow and a GenAI support assistant service.

##  Modules Overview

### 1. `data_pipeline/`
- **Purpose**: End‑to‑end ETL pipeline for the Titanic dataset.
- **Contents**:
  - `pipeline.py`: Loads, cleans, and preprocesses the dataset.
  - `requirements.txt`: Dependencies (pandas, scikit‑learn, seaborn).
  - `README.txt`: Notes on install/run steps and design decisions.
- **Output**: Produces `processed_titanic.csv` for downstream analytics.

### 2. `analytics/`
- **Purpose**: Full analyst‑to‑data‑scientist workflow on the Titanic dataset.
- **Contents**:
  - `01_eda.ipynb`: Exploratory Data Analysis (profiling, cleaning, charts).
  - `02_modeling.ipynb`: Predictive modeling (classification + regression).
  - `README.md`: Workflow summary, charts, and final recommendation.
- **Highlights**:
  - Univariate, bivariate, and multivariate analysis.
  - Standardization checks.
  - Classification models: Logistic Regression, Decision Tree, Random Forest.
  - Regression side‑task: predicting `fare`.
  - Evaluation tables (accuracy, precision, recall, F1, ROC AUC, regression metrics).
  - Final recommendation with saved pipeline artifact.

### 3. `support_assistant/`
- **Purpose**: A small but complete GenAI RAG service for Zepto policy queries.
- **Contents**:
  - `main.py`: FastAPI app with LangGraph nodes and Pydantic schema.
  - `ingest.py`: One‑time ingestion script embedding corpus docs into ChromaDB.
  - `docs/`: 8 Zepto policy text files (delivery, returns, membership, etc.).
  - `Dockerfile`: Builds and runs the FastAPI app locally.
  - `README.md`: Example calls and architecture description.
- **Pipeline Architecture**:
  - **Ingestion**: Corpus docs loaded and chunked.
  - **Embedding**: Sentence‑Transformers (`all‑MiniLM‑L6‑v2`) → ChromaDB.
  - **Retrieval**: Top‑3 chunks via cosine similarity.
  - **Generation**: 
    - Mock mode (`MOCK_LLM=1`): deterministic canned answers.
    - Optional real LLM (`MOCK_LLM=0`): structured prompt with negative constraint + few‑shot.
  - **Routing**: LangGraph nodes (`classify_intent` → `retrieve_and_answer` or `direct_answer`).
  - **Output**: JSON schema enforced (answer, sources, confidence).

##  How to Run

### Data Pipeline
```bash
cd data_pipeline
python pipeline.py

cd support_assistant
docker build -t support-assistant .
docker run -p 7860:7860 support-assistant

Test:
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query":"What is Zepto delivery policy?"}'

