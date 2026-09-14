import os
import sys
import json
import random
import logging
import zipfile
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path to resolve src.* imports cross-platform
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import tiktoken
from src.utils import config, helpers
from src.preprocessing.cleaners import clean_text
from src.preprocessing.chunker import chunk_text

logger = logging.getLogger(__name__)

# Candidate key groupings for QA field schema mapping
QUESTION_KEYS = ["question", "query", "Question", "prompt"]
ANSWER_KEYS = ["answer", "response", "Answer", "ground_truth"]
CASE_NAME_KEYS = ["case_name", "case_title", "title", "case", "judgment_name"]
CONTEXT_KEYS = [
    "context", "judgment_text", "judgment", "document", "doc_text",
    "full_text", "content", "passage", "text", "ratio", "facts", "summary"
]

def find_data_file(directory: Path) -> Optional[Path]:
    """Recursively search for json, jsonl, or csv, extracting zips if found."""
    json_files = []
    csv_files = []
    zip_files = []
    for f in directory.rglob("*"):
        if f.is_file():
            if f.suffix in (".json", ".jsonl"):
                json_files.append(f)
            elif f.suffix == ".csv":
                csv_files.append(f)
            elif f.suffix == ".zip":
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
            zip_ref.extractall(extract_dir)
        return find_data_file(extract_dir)
    return None

def load_raw_data(file_path: Path) -> List[Dict]:
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

def load_judgment_corpus(judgments_dir: Optional[Path]) -> Dict[str, str]:
    """
    Load external judgment corpus (e.g. ILDC text files) indexed by case name / ID.
    Returns a dictionary mapping case name (normalized lowercase) to judgment text.
    """
    corpus = {}
    if not judgments_dir or not judgments_dir.exists():
        return corpus
    logger.info(f"Loading external judgment corpus from: {judgments_dir}")
    for file_path in judgments_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            stem = file_path.stem.lower().strip()
            if file_path.suffix in (".txt", ".md"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if text:
                    corpus[stem] = text
            elif file_path.suffix in (".json", ".jsonl"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f) if file_path.suffix == ".json" else [json.loads(l) for l in f if l.strip()]
                records = data if isinstance(data, list) else [data]
                for rec in records:
                    case_key = rec.get("case_name") or rec.get("title") or rec.get("id") or stem
                    doc_text = rec.get("text") or rec.get("judgment") or rec.get("content") or ""
                    if case_key and doc_text:
                        corpus[str(case_key).lower().strip()] = doc_text.strip()
        except Exception as e:
            logger.warning(f"Failed to read judgment file {file_path}: {e}")
    logger.info(f"Loaded {len(corpus)} judgments into corpus lookup.")
    return corpus

def extract_qa_fields(
    record: Dict, judgments_lookup: Optional[Dict[str, str]] = None
) -> Tuple[str, str, str, str]:
    """
    Extract question, answer, context, and case metadata from a raw record.
    Eliminates unsafe fallback that mapped case_name to context.
    
    Returns:
        tuple: (question, answer, context, case_name)
    """
    q_val, a_val, c_val, case_name = "", "", "", ""

    # 1. Extract question
    for qk in QUESTION_KEYS:
        if qk in record and record[qk]:
            q_val = str(record[qk])
            break

    # 2. Extract answer
    for ak in ANSWER_KEYS:
        if ak in record and record[ak]:
            a_val = str(record[ak])
            break

    # 3. Extract case metadata (case_name / title)
    for mk in CASE_NAME_KEYS:
        if mk in record and record[mk]:
            case_name = str(record[mk])
            break

    # 4. Extract context from explicit context keys
    for ck in CONTEXT_KEYS:
        if ck in record and record[ck]:
            candidate = str(record[ck])
            if candidate.strip():
                c_val = candidate
                break

    # 5. Check external judgment corpus if context missing in record
    if not c_val and judgments_lookup and case_name:
        norm_case = case_name.lower().strip()
        if norm_case in judgments_lookup:
            c_val = judgments_lookup[norm_case]
        else:
            rec_id = str(record.get("id", "")).lower().strip()
            if rec_id and rec_id in judgments_lookup:
                c_val = judgments_lookup[rec_id]

    if not q_val or not a_val:
        available_keys = list(record.keys())
        raise ValueError(f"Could not map record keys to question/answer schema. Available keys: {available_keys}")

    return q_val, a_val, c_val, case_name

def validate_context(
    context: str, case_name: str = "", min_tokens: int = 30, min_chars: int = 150
) -> Tuple[bool, str]:
    """
    Validate that the extracted context is substantive legal text rather than
    a case title, header, docket number, or empty placeholder.
    
    Returns:
        tuple: (is_valid: bool, rejection_reason: str)
    """
    if not context or not context.strip():
        return False, "Context is empty or missing."

    clean_c = context.strip()
    # Check if context is merely the case title
    if case_name and clean_c.lower() == case_name.strip().lower():
        return False, f"Context is identical to case title metadata ('{case_name}')."

    # Token count check using cl100k_base or whitespace fallback
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(clean_c))
    except Exception:
        token_count = len(clean_c.split())

    if token_count < min_tokens:
        return False, f"Context too short ({token_count} tokens < threshold {min_tokens}); likely metadata or header."

    if len(clean_c) < min_chars:
        return False, f"Context character length too short ({len(clean_c)} < {min_chars})."

    return True, "Valid substantive context."

def process_record(
    record: Dict,
    idx: int,
    judgments_lookup: Optional[Dict[str, str]] = None,
    min_tokens: int = 30,
    strict_context: bool = True
) -> Optional[Dict]:
    """
    Clean, validate, and chunk a single record.
    Returns a structured dictionary or None if invalid or context fails validation.
    """
    try:
        q_raw, a_raw, c_raw, case_raw = extract_qa_fields(record, judgments_lookup)
        q = clean_text(q_raw)
        a = clean_text(a_raw)
        c = clean_text(c_raw)
        case_name = clean_text(case_raw)

        if not q or not a:
            logger.warning(f"Record {idx} filtered out: empty question or answer after cleaning.")
            return None

        # Validate context quality
        is_valid_ctx, reason = validate_context(c, case_name=case_name, min_tokens=min_tokens)
        if not is_valid_ctx:
            if strict_context:
                logger.warning(f"Record {idx} filtered out: {reason}")
                return None
            else:
                logger.warning(f"Record {idx} warning: {reason} (strict mode disabled)")

        chunks = chunk_text(c)
        if not chunks:
            logger.warning(f"Record {idx} filtered out: chunking produced 0 chunks.")
            return None

        result = {
            "id": f"q_{idx:04d}",
            "question": q,
            "answer": a,
            "context_chunks": chunks
        }
        if case_name:
            result["case_name"] = case_name
        return result
    except Exception as e:
        logger.warning(f"Failed to process record at index {idx}: {e}")
        return None

def split_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
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

def save_splits(data_dir: Path, processed: List[Dict], train: List[Dict], val: List[Dict], test: List[Dict]) -> None:
    """Save all jsonl splits using helpers.save_jsonl."""
    helpers.save_jsonl(data_dir / "benchmark.jsonl", processed)
    helpers.save_jsonl(data_dir / "train.jsonl", train)
    helpers.save_jsonl(data_dir / "val.jsonl", val)
    helpers.save_jsonl(data_dir / "test.jsonl", test)

def log_metrics(processed: List[Dict], train: List[Dict], val: List[Dict], test: List[Dict]) -> None:
    """Compute and log corpus statistics."""
    total = len(processed)
    avg_chunks = sum(len(r["context_chunks"]) for r in processed) / total if total > 0 else 0
    logger.info(f"Total processed records: {total}")
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)} (capped at 200)")
    logger.info(f"Avg chunks per question: {avg_chunks:.1f}")

def main() -> None:
    """Main orchestrator for dataset curation, context validation, and splitting."""
    parser = argparse.ArgumentParser(description="Curate LawSaathi-RAG benchmark dataset with context validation.")
    parser.add_argument("--raw-dir", type=Path, default=project_root / "data" / "raw", help="Directory containing raw QA files.")
    parser.add_argument("--judgments-dir", type=Path, default=None, help="Directory containing external judgment texts.")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data", help="Output directory for curated splits.")
    parser.add_argument("--min-tokens", type=int, default=30, help="Minimum token threshold for valid context chunks.")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict context validation filtering.")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logger.error(f"No dataset files found in {raw_dir}")
        sys.exit(1)

    data_file = find_data_file(raw_dir)
    if not data_file:
        logger.error("No valid data file discovered.")
        sys.exit(1)

    judgments_dir = args.judgments_dir or (raw_dir / "judgments" if (raw_dir / "judgments").exists() else None)
    judgments_lookup = load_judgment_corpus(judgments_dir) if judgments_dir else {}

    raw_records = load_raw_data(data_file)
    logger.info(f"Loaded {len(raw_records)} raw records from {data_file}")

    processed = []
    skipped_count = 0
    strict_mode = not args.no_strict

    for idx, record in enumerate(raw_records):
        res = process_record(
            record,
            len(processed),
            judgments_lookup=judgments_lookup,
            min_tokens=args.min_tokens,
            strict_context=strict_mode
        )
        if res:
            processed.append(res)
        else:
            skipped_count += 1

    logger.info(f"Curated {len(processed)} valid QA records with substantive context. Filtered {skipped_count} invalid records.")

    if not processed:
        logger.error("No valid records curated after applying context validation. Aborting split generation.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(processed)
    train, val, test = split_records(processed)
    save_splits(args.output_dir, processed, train, val, test)
    log_metrics(processed, train, val, test)

if __name__ == "__main__":
    main()
