#!/bin/bash
# Start FastAPI backend in the background on port 8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit UI on the port assigned by the cloud platform (default 8501)
PORT=${PORT:-8501}
echo "Starting Streamlit on port $PORT..."
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
