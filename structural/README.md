# Agentic Code Intelligence — Structural Code Intelligence (Stage 1)

Modular agentic code-understanding system for JavaScript and TypeScript repositories.

## Overview

This package implements **Stage 1 (Repository Ingestion & Tree-Sitter Parser)** of **Member 3 (Structural Code Intelligence)**.

It provides a clean, modular foundation for discovering, reading, and parsing JS/TS source code into structured Abstract Syntax Trees (ASTs) using Tree-Sitter, while strictly adhering to separation of concerns.

```
Repository Files (.js, .jsx, .ts, .tsx)
               │
               ▼
┌───────────────────────────────┐
│       RepositoryScanner       │  ──> Skips node_modules, .git, dist, build, etc.
└───────────────────────────────┘  ──> Skips binary and non-UTF8 files
               │
               ▼
          SourceFile               ──> Relative path, language, content, line count
               │
               ▼
┌───────────────────────────────┐
│       TreeSitterParser        │  ──> JavaScript, JSX, TypeScript, TSX grammars
└───────────────────────────────┘  ──> Detects syntax errors gracefully
               │
               ▼
          ParseResult              ──> Live Tree-Sitter AST root node + NodeInfo
               │
      (Future Stages: 2 & 3)
  Symbol Extraction & Call Graphs
```

## Directory Structure

```
agentic_code_intelligence/
├── backend/
│   ├── __init__.py
│   └── structural/
│       ├── __init__.py          # Public package exports
│       ├── models.py            # SourceFile, NodeInfo, ParseResult dataclasses
│       ├── repository.py        # RepositoryScanner and directory filters
│       └── parser.py            # TreeSitterParser and AST traversal helpers
├── demo_repo/                   # Sample test JS/TS codebase
├── tests/
│   ├── __init__.py
│   └── test_structural.py       # 12 automated unit tests
├── demo_structural.py           # End-to-end demonstration script
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
- `pytest>=7.0.0`

## Running the Demo

```bash
python demo_structural.py
```

## Running the Tests

```bash
pytest tests/test_structural.py -v
```

## Core API Usage

### 1. Repository Discovery

```python
from backend.structural import RepositoryScanner

scanner = RepositoryScanner("path/to/repo")
source_files = scanner.scan()

for file in source_files:
    print(f"{file.path} ({file.language}): {file.line_count} lines")
```

### 2. Tree-Sitter Parsing

```python
from backend.structural import TreeSitterParser

parser = TreeSitterParser()

for file in source_files:
    result = parser.parse(file)
    if result.success:
        print(f"Successfully parsed {result.file_path}: root={result.root_node_type}")
    else:
        print(f"Failed parsing {result.file_path}: error={result.error}")
```

### 3. AST Traversal (Preparation for Stage 2 & 3)

```python
from backend.structural import traverse_ast, get_node_info

# Depth-first pre-order traversal across all AST nodes
for node in traverse_ast(result.root_node):
    info = get_node_info(node)
    print(f"[{info.node_type}] lines {info.start_line}..{info.end_line}, bytes {info.start_byte}..{info.end_byte}")
```
