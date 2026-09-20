import os
import sys
import json
import random
import logging
import zipfile
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from src.utils import config, helpers
from src.preprocessing.cleaners import clean_text
from src.preprocessing.chunker import chunk_text

logger = logging.getLogger(__name__)

def find_data_file(directory: Path) -> Path:
    """Recursively search for json, jsonl, or csv, extracting zips if found."""
    json_files = []
    csv_files = []
    zip_files = []
    for f in directory.rglob("*"):
        if f.is_file():
            suffix = f.suffix.lower()
            if suffix in (".json", ".jsonl"):
                json_files.append(f)
            elif suffix == ".csv":
                csv_files.append(f)
            elif suffix == ".zip":
                zip_files.append(f)
    if json_files:
        return json_files[0]
    if csv_files:
        return csv_files[0]
    if zip_files:
        zip_path = zip_files[0]
        extract_dir = zip_path.parent / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            safe_extract_zip(zip_ref, extract_dir)
        return find_data_file(extract_dir)
    return None

def safe_extract_zip(zip_ref: zipfile.ZipFile, extract_dir: Path) -> None:
    """Extract a zip archive without allowing members to escape extract_dir."""
    extract_root = extract_dir.resolve()
    for member in zip_ref.infolist():
        target_path = (extract_root / member.filename).resolve()
        if target_path != extract_root and extract_root not in target_path.parents:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    zip_ref.extractall(extract_root)

def load_raw_data(file_path: Path) -> list[dict]:
    """Load raw records from JSON, JSONL, or CSV format."""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                return content
            if isinstance(content, dict):
                for val in content.values():
                    if isinstance(val, list):
                        return val
                return [content]
        elif suffix == ".csv":
            df = pd.read_csv(file_path)
            return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error loading raw data from {file_path}: {e}")
    return []

def extract_qa_fields(record: dict) -> tuple[str, str, str]:
    """Extract question, answer, and context using candidate keys in priority order."""
    keys_sets = [
        ("question", "answer", "context"),
        ("query", "response", "passage"),
        ("Question", "Answer", "Context"),
        ("text", "summary", "content"),
        # Fallbacks for datasets missing an explicit context field
        ("question", "answer", "case_name"),
        ("question", "answer", "answer")
    ]
    for q_key, a_key, c_key in keys_sets:
        if q_key in record and a_key in record and c_key in record:
            return record[q_key], record[a_key], record[c_key]
    logger.error(f"Keys mismatch. Available keys: {list(record.keys())}")
    raise ValueError(f"Could not map record keys to any known QA schema.")

def process_record(record: dict, idx: int) -> dict:
    """Clean and chunk a single record, returning structured dict or None if invalid."""
    try:
        q_raw, a_raw, c_raw = extract_qa_fields(record)
        q = clean_text(q_raw)
        a = clean_text(a_raw)
        c = clean_text(c_raw)
        chunks = chunk_text(c)
        if not q or not a or not chunks:
            logger.warning(f"Record {idx} filtered out: empty fields after cleaning.")
            return None
        return {
            "id": f"q_{idx:04d}",
            "question": q,
            "answer": a,
            "context_chunks": chunks
        }
    except Exception as e:
        logger.warning(f"Failed to process record at index {idx}: {e}")
        return None

def split_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into 70% train, 15% val, 15% test with test capped at 200."""
    total = len(records)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    train = records[:train_end]
    val = records[train_end:val_end]
    test = records[val_end:]
    if len(test) > 200:
        excess = test[200:]
        test = test[:200]
        train.extend(excess)
    return train, val, test

def save_splits(data_dir: Path, processed: list, train: list, val: list, test: list) -> None:
    """Save all jsonl splits using helpers.save_jsonl."""
    helpers.save_jsonl(data_dir / "benchmark.jsonl", processed)
    helpers.save_jsonl(data_dir / "train.jsonl", train)
    helpers.save_jsonl(data_dir / "val.jsonl", val)
    helpers.save_jsonl(data_dir / "test.jsonl", test)

def log_metrics(processed: list, train: list, val: list, test: list) -> None:
    """Compute and log corpus statistics."""
    total = len(processed)
    avg_chunks = sum(len(r["context_chunks"]) for r in processed) / total if total > 0 else 0
    logger.info(f"Total processed records: {total}")
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)} (capped at 200)")
    logger.info(f"Avg chunks per question: {avg_chunks:.1f}")

def main() -> None:
    """Main orchestrator for dataset curation and splitting."""
    raw_dir = project_root / "data" / "raw"
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logger.error("No dataset files found in data/raw/")
        sys.exit(1)
    data_file = find_data_file(raw_dir)
    if not data_file:
        logger.error("No valid data file discovered.")
        sys.exit(1)
    raw_records = load_raw_data(data_file)
    processed = []
    for record in raw_records:
        res = process_record(record, len(processed))
        if res:
            processed.append(res)
    random.seed(42)
    random.shuffle(processed)
    train, val, test = split_records(processed)
    save_splits(project_root / "data", processed, train, val, test)
    log_metrics(processed, train, val, test)

if __name__ == "__main__":
    main()
