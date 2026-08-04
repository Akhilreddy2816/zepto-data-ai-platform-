"""
Zepto Data & AI Platform - Unified FastAPI REST Backend Server
Exposes RESTful endpoints for Data Engineering ETL Pipeline, Analytics ML Engine, and RAG Chatbot.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Imports from Platform Modules
from data_pipeline.database import DatabaseManager
from data_pipeline.etl import ETLPipeline
from analytics.evaluate import ModelEvaluator
from analytics.predict import DeliveryPredictor
from analytics.train import ModelTrainer
from support_assistant.chatbot import ZeptoSupportChatbot
from support_assistant.config import DOCUMENTS_DIR

app = FastAPI(
    title="Zepto Data & AI Platform API",
    description="Unified Enterprise REST Backend powering ETL Pipelines, Delivery ML Analytics, and RAG Chatbot.",
    version="1.0.0",
)

# Enable CORS for Frontend UI Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Instances
db_mgr = DatabaseManager()
etl_pipeline = ETLPipeline(db_manager=db_mgr)
support_chatbot = ZeptoSupportChatbot()
predictor = DeliveryPredictor()


# --- Pydantic Data Models ---
class ScrapeRequest(BaseModel):
    num_items: int = Field(default=50, ge=5, le=200, description="Target number of items to scrape")


class PredictionRequest(BaseModel):
    customer_tenure_months: int = Field(..., ge=0, le=120)
    order_distance_km: float = Field(..., ge=0.1, le=50.0)
    item_count: int = Field(..., ge=1, le=100)
    order_value_inr: float = Field(..., ge=1.0)
    traffic_density: str = Field(..., description="Options: Low, Medium, High")
    weather_condition: str = Field(..., description="Options: Clear, Rainy, Foggy")
    driver_experience_years: int = Field(..., ge=0, le=30)
    delivery_time_mins: float = Field(..., ge=1.0)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User question regarding Zepto policies")


# --- System Health Check ---
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Returns operational diagnostic metrics for all sub-systems."""
    return {
        "status": "ONLINE",
        "system": "Zepto Data & AI Platform",
        "modules": {
            "data_pipeline": "OPERATIONAL",
            "analytics_ml": "OPERATIONAL",
            "support_assistant_rag": "OPERATIONAL",
        },
        "database_connected": True,
    }


# --- Module 1: Data Pipeline Endpoints ---
@app.post("/api/v1/pipeline/run", tags=["Data Engineering Pipeline"])
def run_pipeline(request: ScrapeRequest):
    """Triggers the automated web scraper, data cleaning, SQL load, and CSV export pipeline."""
    try:
        _, summary = etl_pipeline.run(num_items=request.num_items)
        return {"status": "SUCCESS", "message": "ETL Pipeline completed successfully", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")


@app.get("/api/v1/pipeline/products", tags=["Data Engineering Pipeline"])
def get_products():
    """Fetches stored product records from database."""
    try:
        df = db_mgr.fetch_products_dataframe()
        if df.empty:
            # Trigger quick pipeline if DB is empty
            df, _ = etl_pipeline.run(num_items=30)
        return {"count": len(df), "products": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading database: {str(e)}")


# --- Module 2: Analytics & Machine Learning Endpoints ---
@app.post("/api/v1/analytics/train", tags=["Analytics & Machine Learning"])
def train_models():
    """Triggers model training across candidate algorithms and returns benchmark comparison."""
    try:
        trainer = ModelTrainer()
        leaderboard, best_name, _ = trainer.train_and_evaluate_all()
        # Refresh predictor reference
        global predictor
        predictor.load_model()
        return {
            "status": "SUCCESS",
            "best_model": best_name,
            "leaderboard": leaderboard,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@app.post("/api/v1/analytics/predict", tags=["Analytics & Machine Learning"])
def predict_delay(payload: PredictionRequest):
    """Real-time API predicting quick-commerce delivery delay probability."""
    try:
        input_dict = payload.model_dump()
        result = predictor.predict_sample(input_dict)
        return {"status": "SUCCESS", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/v1/analytics/metrics", tags=["Analytics & Machine Learning"])
def get_model_metrics():
    """Returns detailed metrics, confusion matrix, and feature importances for active model."""
    try:
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model()
        # Exclude matplotlib figures from JSON response
        clean_metrics = {k: v for k, v in metrics.items() if not k.startswith("fig_")}
        return {"status": "SUCCESS", "metrics": clean_metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model metrics: {str(e)}")


# --- Module 3: Support Assistant (RAG) Endpoints ---
@app.post("/api/v1/rag/chat", tags=["GenAI Support Assistant"])
def rag_chat(request: ChatRequest):
    """Queries the RAG Chatbot for grounded policy answers."""
    try:
        response = support_chatbot.ask(request.query)
        return {"status": "SUCCESS", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


@app.post("/api/v1/rag/upload", tags=["GenAI Support Assistant"])
def upload_policy_document(file: UploadFile = File(...)):
    """Uploads a PDF or text document to the support assistant knowledge base."""
    allowed_extensions = [".txt", ".pdf", ".md"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    target_path = DOCUMENTS_DIR / file.filename
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Refresh vector index
        num_chunks = support_chatbot.rag_pipeline.build_or_refresh_knowledge_base()
        return {
            "status": "SUCCESS",
            "filename": file.filename,
            "saved_path": str(target_path),
            "total_indexed_chunks": num_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload processing error: {str(e)}")


@app.get("/api/v1/rag/documents", tags=["GenAI Support Assistant"])
def list_indexed_documents():
    """Lists policy files currently stored in the knowledge base."""
    doc_files = [f.name for f in DOCUMENTS_DIR.glob("*.*") if f.suffix.lower() in [".txt", ".pdf", ".md"]]
    return {"indexed_documents_count": len(doc_files), "documents": doc_files}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
