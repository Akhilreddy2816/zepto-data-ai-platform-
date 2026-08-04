#!/bin/bash
# Start Streamlit UI on the port assigned by the cloud platform (default 8000)
PORT=${PORT:-8000}
echo "Starting Zepto Platform on port $PORT..."
exec streamlit run frontend/app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --server.enableCORS false \
    --server.enableXsrfProtection false


