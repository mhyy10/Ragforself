"""Configuration settings for RAG system"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""

    # Database settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "rag_db")
    DB_USER = os.getenv("DB_USER", "rag_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

    # Database URL
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Vector database settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store_new")

    # LLM settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ms-c4982687-e50c-4d7e-9c8b-1eb2514c61d7")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-32B")

    # Application settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./upload")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16777216))  # 16MB
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # Base URL
    BASE_URL = os.getenv("BASE_URL", "https://api-inference.modelscope.cn/v1")

    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "ms-c4982687-e50c-4d7e-9c8b-1eb2514c61d7")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)