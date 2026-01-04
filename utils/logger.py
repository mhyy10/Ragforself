"""Logging utilities for RAG system"""
import logging
import logging.config
from datetime import datetime
import os
from config.settings import Settings


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with specified name and configuration
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional, logs to console if not provided)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding multiple handlers to the same logger
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class RAGLogger:
    """Centralized logging class for RAG system"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = setup_logger(
            "rag_system",
            log_file="logs/rag_system.log",
            level=logging.INFO
        )
        
        # Create separate loggers for different components
        self.doc_logger = setup_logger(
            "rag_documents",
            log_file="logs/documents.log",
            level=logging.INFO
        )
        
        self.query_logger = setup_logger(
            "rag_queries",
            log_file="logs/queries.log",
            level=logging.INFO
        )
        
        self.security_logger = setup_logger(
            "rag_security",
            log_file="logs/security.log",
            level=logging.INFO
        )
    
    def log_document_processed(self, filename: str, chunks_count: int, user: str = "anonymous"):
        """Log document processing event"""
        self.doc_logger.info(
            f"User '{user}' processed document '{filename}' into {chunks_count} chunks"
        )
    
    def log_query(self, query: str, response: str, user: str = "anonymous", response_time: float = None):
        """Log query and response"""
        if response_time:
            self.query_logger.info(
                f"User '{user}' asked: '{query[:50]}...' - Response time: {response_time:.2f}s"
            )
        else:
            self.query_logger.info(f"User '{user}' asked: '{query[:50]}...'")
    
    def log_security_event(self, event: str, user: str = "anonymous", ip_address: str = None):
        """Log security-related events"""
        if ip_address:
            self.security_logger.warning(f"Security event for user '{user}' from IP {ip_address}: {event}")
        else:
            self.security_logger.warning(f"Security event for user '{user}': {event}")
    
    def log_error(self, component: str, error: str, user: str = "anonymous"):
        """Log error events"""
        self.logger.error(f"Error in {component} for user '{user}': {error}")


# Global logger instance
rag_logger = RAGLogger()