"""Deployment scripts for RAG system"""
import os
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install Python dependencies from requirements.txt"""
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully!")


def setup_environment():
    """Set up environment variables and directories"""
    print("Setting up environment...")
    
    # Create necessary directories
    directories = ["uploads", "vector_store", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Directory '{directory}' created/verified")
    
    print("Environment setup completed!")


def run_migrations():
    """Run database migrations (placeholder - would use alembic in real app)"""
    print("Running database migrations...")
    # In a real application, you would use alembic here:
    # subprocess.check_call(["alembic", "upgrade", "head"])
    print("Database migrations completed!")


def start_api_server():
    """Start the FastAPI server"""
    print("Starting API server...")
    subprocess.check_call([
        "uvicorn", 
        "api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"  # Remove this in production
    ])


def run_tests():
    """Run the test suite"""
    print("Running tests...")
    subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
    print("All tests passed!")


def build_docker_image():
    """Build Docker image for the application"""
    print("Building Docker image...")
    subprocess.check_call(["docker", "build", "-t", "rag-system:latest", "."])
    print("Docker image built successfully!")


def start_with_docker_compose():
    """Start services using docker-compose"""
    print("Starting services with docker-compose...")
    subprocess.check_call(["docker-compose", "up", "-d"])
    print("Services started successfully!")


def setup_production():
    """Complete setup for production environment"""
    print("Setting up production environment...")
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Run migrations
    run_migrations()
    
    print("Production environment setup completed!")


def main():
    """Main deployment function with command line options"""
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [setup|test|run|build|compose]")
        print("  setup   - Set up the environment and install dependencies")
        print("  test    - Run the test suite")
        print("  run     - Start the API server")
        print("  build   - Build Docker image")
        print("  compose - Start services with docker-compose")
        print("  prod    - Complete production setup")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "setup":
        install_dependencies()
        setup_environment()
    elif command == "test":
        run_tests()
    elif command == "run":
        start_api_server()
    elif command == "build":
        build_docker_image()
    elif command == "compose":
        start_with_docker_compose()
    elif command == "prod":
        setup_production()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()