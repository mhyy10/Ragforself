"""Global services manager for RAG system"""
import logging
import os
from typing import Optional
from core.embedding_engine import EmbeddingEngine
from core.retrieval_engine import RetrievalEngine
from core.document_processor import DocumentProcessor
from config.settings import Settings


class ServiceManager:
    """Manages global services for the RAG system"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the service manager"""
        if not ServiceManager._initialized:
            self.logger = logging.getLogger(__name__)
            self.settings = Settings()
            
            # Initialize services
            self.document_processor = DocumentProcessor(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP
            )
            
            self.embedding_engine = EmbeddingEngine()
            self.retrieval_engine = RetrievalEngine(self.embedding_engine)
            
            # Flag to track initialization status
            self._initialized = True
            self.logger.info("Service manager initialized")
    
    def initialize_vector_store(self):
        """Initialize the vector store by loading existing data or creating a new one"""
        try:
            # Try to load existing vector store
            self.embedding_engine.load_vector_store()
            self.logger.info("Successfully loaded existing vector store")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load existing vector store: {str(e)}")
            self.logger.info("Vector store will be created when first document is uploaded")
            return False

    def get_processed_files_tracker_path(self):
        """Get the path for the processed files tracker"""
        return os.path.join(self.settings.VECTOR_DB_PATH, "processed_files.json")

    def load_processed_files(self):
        """Load the list of already processed files"""
        import json
        tracker_path = self.get_processed_files_tracker_path()

        if os.path.exists(tracker_path):
            try:
                with open(tracker_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                self.logger.warning(f"Could not load processed files list: {str(e)}")
                return set()
        return set()

    def save_processed_files(self, processed_files):
        """Save the list of processed files"""
        import json
        tracker_path = self.get_processed_files_tracker_path()

        # Ensure the directory exists
        os.makedirs(os.path.dirname(tracker_path), exist_ok=True)

        try:
            with open(tracker_path, 'w', encoding='utf-8') as f:
                json.dump(list(processed_files), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save processed files list: {str(e)}")

    def scan_and_process_uploaded_files(self):
        """Scan the uploads directory and process any unprocessed files"""
        import os
        import hashlib
        from pathlib import Path

        uploaded_files = os.listdir(self.settings.UPLOAD_FOLDER)
        processed_files = self.load_processed_files()
        new_processed_count = 0

        for filename in uploaded_files:
            file_path = os.path.join(self.settings.UPLOAD_FOLDER, filename)

            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Create a unique identifier for the file based on name and modification time
                file_stat = Path(file_path).stat()
                file_id = f"{filename}_{int(file_stat.st_mtime)}_{file_stat.st_size}"

                # Check if this file has already been processed
                if file_id not in processed_files:
                    try:
                        # Process the document
                        metadata = {
                            "source": filename,
                            "original_filename": filename,
                            "upload_date": str(file_stat.st_ctime),
                            "file_id": file_id  # Add file ID to metadata
                        }

                        documents = self.document_processor.process_document(file_path, metadata)

                        # Add to vector store
                        if self.embedding_engine.vector_store is None:
                            self.embedding_engine.create_vector_store(documents)
                        else:
                            self.embedding_engine.add_documents(documents)

                        # Mark as processed
                        processed_files.add(file_id)
                        self.save_processed_files(processed_files)

                        self.logger.info(f"Processed and added {filename} to vector store")
                        new_processed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error processing file {filename}: {str(e)}")
                else:
                    self.logger.debug(f"File {filename} already processed, skipping")

        self.logger.info(f"Scanned upload directory, found {len(uploaded_files)} files, processed {new_processed_count} new files")
        return new_processed_count
    
    def get_embedding_engine(self):
        """Get the global embedding engine instance"""
        return self.embedding_engine
    
    def get_retrieval_engine(self):
        """Get the global retrieval engine instance"""
        return self.retrieval_engine
    
    def get_document_processor(self):
        """Get the global document processor instance"""
        return self.document_processor

    def get_system_status(self):
        """Get the overall system status"""
        import os
        from config.settings import Settings

        # Check vector store status
        vector_store_ready = self.embedding_engine.is_vector_store_ready()

        # Check if vector store files exist in directory
        vector_store_path = self.settings.VECTOR_DB_PATH
        vector_store_exists = os.path.exists(vector_store_path) and any(os.listdir(vector_store_path))

        # Count uploaded documents
        documents_count = len([f for f in os.listdir(self.settings.UPLOAD_FOLDER)
                              if os.path.isfile(os.path.join(self.settings.UPLOAD_FOLDER, f))])

        # Determine status
        if vector_store_ready and documents_count > 0:
            status = "ready"
            message = "Vector store is ready and accessible"
        elif vector_store_exists and not vector_store_ready:
            status = "needs_reload"
            message = "Vector store files exist but need to be loaded. Restart the application to load them."
        elif documents_count > 0 and not vector_store_exists:
            status = "processing"
            message = "Documents have been uploaded but vector store not yet created."
        elif documents_count == 0:
            status = "empty"
            message = "No documents have been uploaded yet"
        else:
            status = "initializing"
            message = "System is initializing"

        return {
            "status": status,
            "message": message,
            "documents_count": documents_count,
            "vector_store_ready": vector_store_ready,
            "vector_store_exists": vector_store_exists
        }


# Global service manager instance
service_manager = ServiceManager()


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance"""
    return service_manager