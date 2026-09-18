import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Configuration
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
USE_LOCAL_MODEL: bool = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta/llama-3.1-8b-instruct")

# Chunking Configuration
CHUNK_SIZE: int = 512
OVERLAP: int = 50
TOP_K: int = 5

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if not OPENAI_API_KEY and not USE_LOCAL_MODEL:
    logger.warning("No API key found and local model disabled.")
elif USE_LOCAL_MODEL:
    logger.info("Using local Ollama model.")
else:
    logger.info(f"Using API: {API_BASE_URL} with model {MODEL_NAME}")
