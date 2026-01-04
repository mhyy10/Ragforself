"""Main FastAPI application for RAG system"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import uuid
from core.services import get_service_manager
from config.settings import Settings
from api.routes import documents, chat, auth


# Initialize settings
settings = Settings()

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Enterprise RAG Knowledge Base System API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts"""
    service_manager = get_service_manager()
    service_manager.initialize_vector_store()
    service_manager.scan_and_process_uploaded_files()

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

@app.get("/")
def read_root():
    return {"message": "RAG System API is running!", "base_url": Settings.BASE_URL}

@app.get("/health")
def health_check():
    return {"status": "healthy"}