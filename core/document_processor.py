"""Document processor module for RAG system"""
import os
import logging
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading, preprocessing, and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self.supported_formats = {'.pdf', '.txt', '.docx', '.doc', '.html', '.htm', '.md', '.rst'}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on its file type

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects
        """
        file_extension = Path(file_path).suffix.lower()

        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                # Use UnstructuredFileLoader for docx/doc files as it's more reliable
                loader = UnstructuredFileLoader(file_path)
            elif file_extension in ['.html', '.htm', '.md', '.rst']:
                loader = UnstructuredFileLoader(file_path)
            else:
                # Fallback to unstructured loader
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents

        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def process_document(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load and process a document (load, split, add metadata)
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata to add to documents
            
        Returns:
            List of processed Document objects
        """
        # Load the document
        raw_documents = self.load_document(file_path)
        
        # Add custom metadata if provided
        if metadata:
            for doc in raw_documents:
                doc.metadata.update(metadata)
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(raw_documents)
        
        self.logger.info(f"Processed document {file_path} into {len(split_documents)} chunks")
        
        return split_documents
    
    def process_multiple_documents(self, file_paths: List[str], metadata_list: Optional[List[dict]] = None) -> List[Document]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of paths to document files
            metadata_list: Optional list of metadata dicts corresponding to each file
            
        Returns:
            List of processed Document objects from all files
        """
        all_documents = []
        
        for i, file_path in enumerate(file_paths):
            try:
                if metadata_list and i < len(metadata_list):
                    metadata = metadata_list[i]
                else:
                    metadata = None
                
                documents = self.process_document(file_path, metadata)
                all_documents.extend(documents)
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        self.logger.info(f"Processed {len(file_paths)} files into {len(all_documents)} total chunks")
        return all_documents