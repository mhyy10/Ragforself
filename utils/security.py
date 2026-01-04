"""Security utilities for RAG system"""
import hashlib
import secrets
from typing import Optional
from config.settings import Settings


def generate_api_key() -> str:
    """
    Generate a secure API key
    
    Returns:
        A secure random API key
    """
    return secrets.token_urlsafe(32)


def hash_document_content(content: str) -> str:
    """
    Generate a hash of document content for deduplication
    
    Args:
        content: Document content to hash
        
    Returns:
        SHA-256 hash of the content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """
    Validate file type based on extension
    
    Args:
        filename: Name of the file to validate
        allowed_extensions: Set of allowed file extensions (e.g., {'.pdf', '.txt'})
        
    Returns:
        True if file type is allowed, False otherwise
    """
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import os
    import re
    
    # Remove any path components
    basename = os.path.basename(filename)
    
    # Remove any non-alphanumeric characters except common ones
    sanitized = re.sub(r'[^\w\s\.\-_]', '', basename)
    
    # Limit length to prevent potential issues
    return sanitized[:255]


def verify_access_token(token: str, expected_token: str) -> bool:
    """
    Verify an access token using constant-time comparison to prevent timing attacks
    
    Args:
        token: Token provided by client
        expected_token: Expected token
        
    Returns:
        True if tokens match, False otherwise
    """
    return secrets.compare_digest(token, expected_token)