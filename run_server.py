import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """Install Python dependencies from requirements.txt"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def setup_environment():
    """Set up environment variables and directories"""
    print("Setting up environment...")
    
    # Create necessary directories
    directories = ["uploads", "vector_store", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Directory '{directory}' created/verified")
    
    print("Environment setup completed!")


def start_api_server():
    """Start the FastAPI server"""
    print("Starting API server...")
    try:
        # Use the direct uvicorn command to start the server
        import uvicorn
        from api.main import app
        
        print("RAG System is now running on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError as e:
        print(f"Error importing server modules: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def main():
    """Main function to run the RAG system"""
    print("Setting up RAG System...")
    
    # Change to the script's directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"Changed to directory: {script_dir}")
    
    # Setup environment and install dependencies
    setup_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Start the API server
    start_api_server()


if __name__ == "__main__":
    main()