#!/bin/bash

# RAG System Startup Script

set -e  # Exit on any error

echo "Starting RAG System..."

# Check if running inside the rag_system directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: This script must be run from the rag_system directory"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# For Windows: venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads vector_store logs

# Start the API server in the background
echo "Starting API server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &

API_PID=$!

echo "API server started with PID $API_PID"
echo "RAG System is now running on http://localhost:8000"
echo "Press Ctrl+C to stop the system"

# Function to handle shutdown
cleanup() {
    echo "Shutting down RAG System..."
    kill $API_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
    echo "RAG System stopped"
    exit 0
}

# Set up signal trapping for graceful shutdown
trap cleanup INT TERM

# Wait for the API process
wait $API_PID