# Zepto Data & AI Platform - System Architecture

## Overview Architecture

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

## Module Interactions & Data Flow

1. **Module 1 (Data Engineering)**: Scrapes e-commerce website HTML DOM, transforms data, cleans missing fields, calculates discounted prices, persists records to SQLite (`zepto_products.db`), and exports CSV artifacts.
2. **Module 2 (Analytics & Machine Learning)**: Loads delivery metrics, caps outliers via IQR, standardizes numerical features, trains 5 candidate models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost), selects the top model via F1/ROC-AUC, and pickles `model.pkl`.
3. **Module 3 (GenAI Support Assistant)**: Ingests company policy documents (`documents/`), creates 500-character chunks, computes dense vector embeddings via `all-MiniLM-L6-v2`, stores vectors in FAISS, and performs grounded RAG query generation.
4. **FastAPI & Streamlit**: Unifies all 3 modules into REST API endpoints and an interactive dark-purple themed dashboard.
