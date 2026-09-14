import pytest

from src.preprocessing.curate_benchmark import extract_qa_fields, process_record


def test_prefers_substantive_judgment_text_over_case_title():
    record = {
        "question": "What did the court decide?",
        "answer": "The appeal was dismissed.",
        "case_name": "Example v. State",
        "judgment_text": "The court considered the evidence in detail and dismissed the appeal.",
    }

    question, answer, context = extract_qa_fields(record)

    assert question == record["question"]
    assert answer == record["answer"]
    assert context == record["judgment_text"]
    assert context != record["case_name"]


def test_rejects_case_name_as_context_fallback():
    record = {
        "question": "What did the court decide?",
        "answer": "The appeal was dismissed.",
        "case_name": "Example v. State",
    }

    with pytest.raises(ValueError, match="substantive QA context"):
        extract_qa_fields(record)


def test_rejects_answer_as_context_fallback():
    record = {
        "question": "What did the court decide?",
        "answer": "The appeal was dismissed.",
    }

    with pytest.raises(ValueError, match="substantive QA context"):
        extract_qa_fields(record)


def test_process_record_filters_title_only_records_instead_of_indexing_them():
    record = {
        "question": "What did the court decide?",
        "answer": "The appeal was dismissed.",
        "case_name": "Example v. State",
    }

    assert process_record(record, 0) is None
