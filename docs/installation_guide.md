# Zepto Data & AI Platform - Installation & Operations Guide

## Prerequisites

- Python 3.10 or higher
- Git
- Docker & Docker Compose (Optional for containerized deployment)

## Environment Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/zepto/zepto-data-ai-platform.git
cd zepto-data-ai-platform

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

Copy `.env.example` to `.env` and fill in your API keys (optional):

```bash
cp .env.example .env
```

```env
DATABASE_URL=sqlite:///./zepto_platform.db
GEMINI_API_KEY=your_gemini_key_optional
OPENAI_API_KEY=your_openai_key_optional
```

## Running the Application

### Option A: Local Development

1. **Start FastAPI REST Backend Server**:
```bash
python -m backend.main
# Server starts at http://localhost:8000 (OpenAPI docs at http://localhost:8000/docs)
```

2. **Start Streamlit Frontend Dashboard**:
```bash
streamlit run frontend/app.py
# UI Dashboard opens at http://localhost:8501
```

### Option B: Docker Deployment

```bash
docker-compose up --build
```

Access Services:
- Streamlit UI Dashboard: `http://localhost:8501`
- FastAPI REST Backend: `http://localhost:8000`

## Running Test Suite

```bash
pytest tests/ -v --cov=.
```
