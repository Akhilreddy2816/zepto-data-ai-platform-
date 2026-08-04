#!/bin/bash
# Start Streamlit UI on the port assigned by the cloud platform (default 8501)
PORT=${PORT:-8501}
echo "Starting Zepto Platform on port $PORT..."
exec streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.fileWatcherType none

