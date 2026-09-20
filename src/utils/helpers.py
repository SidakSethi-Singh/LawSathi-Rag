import time
import logging
import json
import requests
from pathlib import Path
from typing import Any, Callable, List, Dict, Union
from openai import OpenAI
from .config import OPENAI_API_KEY, USE_LOCAL_MODEL, API_BASE_URL

logger = logging.getLogger(__name__)

def call_openai_api(prompt: str, system_prompt: str) -> str:
    """Call the OpenAI API for chat completion."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        timeout=30.0
    )
    return response.choices[0].message.content.strip()

def call_ollama_api(prompt: str, system_prompt: str) -> str:
    """Call the local Ollama API generate endpoint."""
    url = f"{API_BASE_URL}/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": f"{system_prompt}\n\nUser Question:\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.2}
    }
    response = requests.post(url, json=payload, timeout=45.0)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()

def attempt_llm_call(prompt: str, system_prompt: str) -> str:
    """Route to the appropriate LLM provider based on settings."""
    if USE_LOCAL_MODEL:
        return call_ollama_api(prompt, system_prompt)
    return call_openai_api(prompt, system_prompt)

def call_llm(prompt: str, system_prompt: str = "You are a helpful legal assistant.") -> str:
    """Execute LLM call with 3 retries max and exponential backoff."""
    return retry_with_backoff(attempt_llm_call, prompt, system_prompt)

def retry_with_backoff(func: Callable, *args: Any, max_retries: int = 3, initial_delay: float = 2.0, **kwargs: Any) -> Any:
    """Execute a function with exponential backoff retries on failure."""
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}. No retries left.")
    raise RuntimeError(f"All {max_retries} attempts failed for function '{func.__name__}'.")

def save_jsonl(file_path: Path, data: list[dict[str, Any]]) -> None:
    """Save a list of dictionaries to a JSONL file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(data)} records to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSONL file {file_path}: {e}")

def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Helper function to load line-delimited JSON rows into a list.

    Returns an empty list if the file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"File not found: {path_obj}. Returning empty list.")
        return []

    with open(path_obj, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]