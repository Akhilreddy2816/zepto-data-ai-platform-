# ⚡ Zepto Data & AI Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, production-ready AI & Data Platform engineered for **Zepto Quick-Commerce Delivery Analytics**. This unified platform combines automated web scraping ETL data pipelines, predictive machine learning delivery delay models, and a Retrieval-Augmented Generation (RAG) GenAI support assistant into one cohesive repository.

---

## 📌 Table of Contents

- [System Architecture](#-system-architecture)
- [Module Breakdown](#-module-breakdown)
  - [Module 1: Data Engineering Pipeline](#module-1-data-engineering-pipeline)
  - [Module 2: Data Analytics & Machine Learning](#module-2-data-analytics--machine-learning)
  - [Module 3: GenAI Support Assistant (RAG)](#module-3-genai-support-assistant-rag)
- [Repository Folder Structure](#-repository-folder-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [API Endpoint Reference](#-api-endpoint-reference)
- [Running Test Suite](#-running-test-suite)
- [License & Credits](#-license--credits)

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │             USER INTERFACE LAYER             │
                               │        Streamlit Dashboard (Port 8501)       │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼ REST APIs
                               ┌──────────────────────────────────────────────┐
                               │              BACKEND API LAYER               │
                               │           FastAPI Server (Port 8000)         │
                               └──────┬───────────────────┬───────────────────┘
                                      │                   │
                ┌─────────────────────┘                   └─────────────────────┐
                ▼                                                               ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     MODULE 1: ETL PIPELINE    │                               │     MODULE 2: ANALYTICS & ML  │
│  - Requests / BeautifulSoup   │                               │  - Preprocessing & Scaling    │
│  - Pandas Cleaning & Transform│                               │  - 9 EDA Visualizations       │
│  - SQLAlchemy Engine          │                               │  - 5-Fold Stratified CV       │
│  - SQLite / PostgreSQL        │                               │  - Model Leaderboard & Pickles│
└───────────────┬───────────────┘                               └───────────────┬───────────────┘
                │                                                               │
                └───────────────────────────────┬───────────────────────────────┘
                                                │
                                                ▼
                                ┌───────────────────────────────┐
                                │   MODULE 3: RAG SUPPORT BOT   │
                                │  - Policy PDF & Text Loaders  │
                                │  - Recursive Text Splitter    │
                                │  - SentenceTransformers       │
                                │  - FAISS Vector Database      │
                                │  - Grounded LLM Generator     │
                                └───────────────────────────────┘
```

---

## 🧩 Module Breakdown

### Module 1: Data Engineering Pipeline (`data_pipeline/`)
- **Automated Web Scraper**: Extracts product catalog items (Product Name, Category, Brand, Price, Discount, Rating, Stock Status, Product URL, Image URL, Timestamp) with HTTP rate-limiting and mock HTML generator fallback.
- **Data Cleaning & Normalization**: Strips invalid prices, converts data types, normalizes string fields, calculates discounted price, and removes duplicate entries using Pandas.
- **SQL Database Storage**: SQLAlchemy ORM supporting SQLite and PostgreSQL with automated tables for products (`products`) and audit execution logs (`etl_logs`).
- **CSV Exporter**: Generates `raw_products.csv` and `cleaned_products.csv`.

### Module 2: Data Analytics & Machine Learning (`analytics/`)
- **Data Preprocessing**: Median/mode imputation, IQR outlier capping, label encoding, and standard scaling.
- **9 EDA Visualizations**: Histogram, Scatter plot, Box plot, Correlation Heatmap, Pair plot, Count plot, Pie chart, Bar chart, and Line chart.
- **Model Training & Comparison**: Benchmarks 5 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) using 5-Fold Stratified Cross Validation.
- **Evaluation & Artifact Export**: Computes Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix, and exports top model pickle (`model.pkl`).
- **Interactive Notebook**: `eda.ipynb` step-by-step notebook.

### Module 3: GenAI Support Assistant RAG (`support_assistant/`)
- **Policy Knowledge Base**: Built-in policies (`employee_handbook.txt`, `refund_policy.txt`, `hr_policy.txt`, `leave_policy.txt`, `safety_guidelines.txt`).
- **Document Chunking & Vector DB**: Recursive text splitter (500 chunk size, 80 overlap) with SentenceTransformers (`all-MiniLM-L6-v2`) and FAISS vector database.
- **Grounded Chatbot Engine**: Anti-hallucination prompt rules with support for Google Gemini API, OpenAI API, and local heuristic fallback.

---

## 📁 Repository Folder Structure

```
zepto-data-ai-platform/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data_pipeline/
│   ├── scraper.py
│   ├── transform.py
│   ├── database.py
│   ├── etl.py
│   ├── config.py
│   ├── utils.py
│   ├── requirements.txt
│   └── README.md
│
├── analytics/
│   ├── preprocessing.py
│   ├── visualize.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── eda.ipynb
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
│
├── support_assistant/
│   ├── config.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag.py
│   ├── chatbot.py
│   ├── requirements.txt
│   ├── documents/
│   │   ├── employee_handbook.txt
│   │   ├── refund_policy.txt
│   │   ├── hr_policy.txt
│   │   ├── leave_policy.txt
│   │   └── safety_guidelines.txt
│   └── README.md
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── app.py
│   └── style.css
│
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   └── installation_guide.md
│
└── tests/
    ├── test_data_pipeline.py
    ├── test_analytics.py
    ├── test_support_assistant.py
    └── test_backend_api.py
```

---

## 🚀 Installation & Quick Start

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/zepto/zepto-data-ai-platform.git
cd zepto-data-ai-platform
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Platform

**Start FastAPI REST API**:
```bash
python -m backend.main
```

**Start Streamlit UI Dashboard**:
```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Test Suite

```bash
pytest tests/ -v --cov=.
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
