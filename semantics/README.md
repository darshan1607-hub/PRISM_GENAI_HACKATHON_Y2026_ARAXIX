# Agentic Code Intelligence

Modular agentic code-understanding system for JavaScript and TypeScript repositories.

## Overview

The repository consists of modular intelligence layers designed to analyze, parse, index, and query source code:

```
Repository Files (.js, .jsx, .ts, .tsx)
               │
               ▼
┌───────────────────────────────┐
│ Member 1: Repository Scanner  │  ──> Ingests files, excludes build artifacts/node_modules
└───────────────────────────────┘
               │ SourceFile
               ▼
┌───────────────────────────────┐
│ Member 3: Tree-Sitter Parser  │  ──> Builds multi-language ASTs, identifies grammar constructs
└───────────────────────────────┘
               │ AST & Structural Nodes
               ▼
┌───────────────────────────────┐
│ Member 2: Semantic Chunker    │  ──> Extracts functions, classes, methods, docstrings
└───────────────────────────────┘
               │ Code Chunks
               ▼
┌───────────────────────────────┐
│ Member 2: Gemini Embedding 2  │  ──> Generates normalized vector representations
└───────────────────────────────┘
               │ Vector Embeddings
               ▼
┌───────────────────────────────┐
│ Member 2: FAISS Vector Index  │  ──> Persisted index with 1-to-1 source code metadata mapping
└───────────────────────────────┘
               │ Semantic Search Query
               ▼
     Top-K Relevant Code Chunks
```

## Directory Structure

```
agentic_code_intelligence/
├── backend/
│   ├── __init__.py
│   ├── structural/              # Member 1 & 3: Ingestion and Tree-Sitter Parser
│   │   ├── __init__.py
│   │   ├── models.py            # SourceFile, NodeInfo, ParseResult dataclasses
│   │   ├── repository.py        # RepositoryScanner and directory filters
│   │   └── parser.py            # TreeSitterParser and AST traversal helpers
│   └── semantic/                # Member 2: Semantic Code Search
│       ├── __init__.py          # Module public exports
│       ├── chunker.py           # AST and line-based CodeChunker
│       ├── embeddings.py        # Gemini Embedding 2 & Mock test providers
│       └── faiss_search.py      # FAISSIndex & SemanticCodeSearch coordinator
├── demo_repo/                   # Sample test JS/TS codebase
│   └── src/
│       ├── app.js               # Tax calculation
│       ├── auth.ts              # Authentication & token verification
│       ├── broken.js            # Intentional syntax error test file
│       ├── db.ts                # Database connection & connection pool
│       ├── index.ts             # User service class & methods
│       ├── login.ts             # Login route handler & auth caller
│       └── components/
│           └── Button.tsx       # React component
├── tests/
│   ├── __init__.py
│   ├── test_structural.py       # Structural unit tests (12 tests)
│   └── test_semantic.py         # Semantic unit & integration tests (13 tests)
├── demo_structural.py           # Demonstration for Member 1 & 3
├── demo_semantic.py             # Demonstration for Member 2 (Semantic Search)
├── requirements.txt             # Project dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `tree-sitter>=0.22.0`
- `tree-sitter-javascript>=0.21.0`
- `tree-sitter-typescript>=0.21.0`
- `google-genai>=2.0.0`
- `faiss-cpu>=1.9.0`
- `numpy>=1.25.0`
- `pytest>=7.0.0`

## Configuration

### Configuring `GEMINI_API_KEY`

To generate vector embeddings using **Gemini Embedding 2** (`gemini-embedding-2`):

**PowerShell (Windows):**
```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
```

**Bash / macOS / Linux:**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

> **Note:** If `GEMINI_API_KEY` is not set, `demo_semantic.py` will clearly display a notice and utilize `MockEmbeddingProvider` (deterministic subword token hashing) strictly for offline pipeline verification without network calls.

## Running the Semantic Search Demo

Run the end-to-end semantic search demonstration:

```bash
python demo_semantic.py
```

### Run Custom Queries:

```bash
python demo_semantic.py --query "Where is authentication implemented?"
python demo_semantic.py --query "Which code calls authenticateUser?"
python demo_semantic.py --query "Where is the database connection used?"
python demo_semantic.py --query "How does the login process work?"
```

## Running the Structural Demo

```bash
python demo_structural.py
```

## Running Tests

Run all 25 automated unit and integration tests:

```bash
pytest -v
```

## Core API Usage (Member 2 - Semantic Code Search)

### 1. Code Chunking

```python
from backend.structural import RepositoryScanner, TreeSitterParser
from backend.semantic import CodeChunker

scanner = RepositoryScanner("demo_repo")
source_files = scanner.scan()

chunker = CodeChunker(parser=TreeSitterParser())
chunks = chunker.chunk_repository(source_files)

for c in chunks:
    print(f"[{c.chunk_type}] {c.file}:{c.start_line}-{c.end_line} ({c.name})")
```

### 2. Embeddings Generation

```python
from backend.semantic import GeminiEmbeddingProvider

# Reads GEMINI_API_KEY securely from environment
embedder = GeminiEmbeddingProvider()
vectors = embedder.embed_chunks(chunks)
```

### 3. FAISS Indexing & Semantic Search

```python
from backend.semantic import SemanticCodeSearch, GeminiEmbeddingProvider

search_engine = SemanticCodeSearch(embedding_provider=GeminiEmbeddingProvider())
search_engine.index_repository(source_files)

# Natural language query
results = search_engine.search("Where is authentication implemented?", top_k=3)

for r in results:
    print(f"File: {r['file']} (Lines {r['start_line']}-{r['end_line']})")
    print(f"Similarity Score: {r['score']:.4f}")
    print(f"Code:\n{r['code']}\n")
```
