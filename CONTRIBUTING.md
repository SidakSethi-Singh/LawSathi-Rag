# Contributing to LawSaathi-RAG

Thank you for your interest in contributing to **LawSaathi-RAG**! We welcome contributions from researchers, developers, legal tech enthusiasts, and students.

---

## Code of Conduct

Please maintain a welcoming, inclusive, and respectful environment for all contributors. Constructive feedback and collaborative problem solving are strongly encouraged.

---

## Ways to Contribute

1. **Bug Fixes**: Fix issues in preprocessing, retrieval pipelines, or evaluation routines.
2. **Architecture Enhancements**:
   - Integrate cross-encoder rerankers (e.g., `BAAI/bge-reranker-base`).
   - Implement advanced legal chunking strategies (section-aware, order-aware).
   - Add query expansion techniques (e.g., HyDE, legal terminology expansion).
3. **Evaluation & Benchmarking**:
   - Add RAGAS metrics (faithfulness, answer relevance, context recall).
   - Add support for new Indian legal datasets and bilingual/multilingual queries.
4. **Tooling & UI**:
   - Build interactive CLI tools or web demos (Streamlit, Gradio, FastAPI).
   - Enhance visualization charts and leaderboard dashboards.
5. **Documentation**:
   - Improve documentation, docstrings, typing, and tutorials.

---

## Development Setup

### 1. Fork and Clone
```bash
git clone https://github.com/<your-username>/LawSathi-Rag.git
cd LawSathi-Rag
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

---

## Development Guidelines

### Branch Naming Conventions
- `feature/<feature-name>`: New feature or architecture implementation.
- `fix/<issue-name>`: Bug fix.
- `docs/<doc-name>`: Documentation improvements.
- `refactor/<module-name>`: Code refactoring without changing functionality.

### Coding Standards
- Follow **PEP 8** guidelines.
- Include descriptive **docstrings** and **type hints** for all new functions and classes.
- Ensure cross-platform compatibility (use `pathlib.Path` for file paths).

### Testing Your Changes
Before opening a PR, ensure your code runs cleanly without errors:
```bash
# Verify imports and benchmark orchestration
python src/run_full_benchmark.py
```

---

## Submitting a Pull Request (PR)

1. Push your changes to your feature branch:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request against the `main` branch of `codeXsidd/LawSathi-Rag`.
3. Provide a clear PR description detailing:
   - What changed and why.
   - Any benchmarks or test runs performed.
   - Relevant issue numbers (e.g., `Closes #12`).
4. Wait for review and participate constructively in discussion feedback!
