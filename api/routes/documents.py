"""Document management routes for RAG system API"""
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import List
import os
import uuid
from pydantic import BaseModel
from core.services import get_service_manager
from config.settings import Settings


class DocumentUploadResponse(BaseModel):
    filename: str
    file_path: str
    chunks_count: int
    message: str


class QueryRequest(BaseModel):
    query: str
    k: int = 4


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: List[str]


# Initialize router
router = APIRouter()

# Get services from the global service manager
service_manager = get_service_manager()
document_processor = service_manager.get_document_processor()
embedding_engine = service_manager.get_embedding_engine()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.html', '.htm', '.md', '.rst'}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(Settings.UPLOAD_FOLDER, unique_filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document
        from pathlib import Path
        file_stat = Path(file_path).stat()
        file_id = f"{unique_filename}_{int(file_stat.st_mtime)}_{file_stat.st_size}"

        metadata = {
            "source": unique_filename,
            "original_filename": file.filename,
            "upload_date": str(file_stat.st_ctime),
            "file_id": file_id
        }

        documents = document_processor.process_document(file_path, metadata)

        # Add to vector store
        if embedding_engine.vector_store is None:
            embedding_engine.create_vector_store(documents)
        else:
            embedding_engine.add_documents(documents)

        # Track this file as processed
        try:
            processed_files = service_manager.load_processed_files()
            processed_files.add(file_id)
            service_manager.save_processed_files(processed_files)
        except Exception as track_error:
            service_manager.logger.error(f"Could not track processed file: {str(track_error)}")

        # Update the service manager's vector store status
        service_manager.get_embedding_engine().vector_store = embedding_engine.vector_store

        return DocumentUploadResponse(
            filename=unique_filename,
            file_path=file_path,
            chunks_count=len(documents),
            message=f"Successfully uploaded and processed {file.filename}"
        )
    except Exception as e:
        service_manager.logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/status")
async def get_system_status():
    """
    Get the status of the RAG system
    """
    try:
        # Get system status from the service manager
        status_info = service_manager.get_system_status()

        # Map the service manager status to the expected response format
        return {
            "status": status_info["status"],
            "message": status_info["message"],
            "documents_count": status_info["documents_count"],
            "vector_store_initialized": status_info["vector_store_ready"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking system status: {str(e)}")