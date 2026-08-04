# Zepto Data & AI Platform - REST API Reference

The FastAPI backend server runs on `http://localhost:8000`. Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Diagnostic status check for all modules |
| `POST` | `/api/v1/pipeline/run` | Triggers scraper, transform, DB insert, and CSV export |
| `GET` | `/api/v1/pipeline/products` | Retrieves stored product records from database |
| `POST` | `/api/v1/analytics/train` | Trains and benchmarks 5 ML models; exports best model |
| `POST` | `/api/v1/analytics/predict` | Predicts delivery delay probability for an order payload |
| `GET` | `/api/v1/analytics/metrics` | Retrieves accuracy, F1 score, confusion matrix & feature importances |
| `POST` | `/api/v1/rag/chat` | Queries grounded policy chatbot |
| `POST` | `/api/v1/rag/upload` | Uploads PDF/TXT policy file to knowledge base |
| `GET` | `/api/v1/rag/documents` | Lists indexed policy document filenames |

## Request & Response Examples

### 1. Delivery Delay Prediction (`POST /api/v1/analytics/predict`)

**Request Payload:**
```json
{
  "customer_tenure_months": 12,
  "order_distance_km": 6.5,
  "item_count": 8,
  "order_value_inr": 1250.0,
  "traffic_density": "High",
  "weather_condition": "Rainy",
  "driver_experience_years": 2,
  "delivery_time_mins": 22.0
}
```

**Response Payload:**
```json
{
  "status": "SUCCESS",
  "prediction": {
    "prediction_class": 1,
    "status_label": "Delayed (>15 mins)",
    "delay_probability": 0.842,
    "delay_probability_percent": "84.2%",
    "risk_level": "High",
    "model_used": "Random Forest"
  }
}
```

### 2. RAG Chatbot Query (`POST /api/v1/rag/chat`)

**Request Payload:**
```json
{
  "query": "What is the refund SLA for damaged perishable groceries?"
}
```

**Response Payload:**
```json
{
  "status": "SUCCESS",
  "response": {
    "query": "What is the refund SLA for damaged perishable groceries?",
    "answer": "If delivered products are damaged, expired, or missing, customers are eligible for a 100% instant refund or Zepto Cash credit. Refund requests must be logged within 48 hours.\n\n📌 **Sources Cited**: `refund_policy.txt`",
    "sources": ["refund_policy.txt"],
    "retrieved_chunks_count": 3,
    "llm_provider": "Heuristic Fallback Engine (Offline)"
  }
}
```
