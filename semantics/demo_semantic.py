#!/usr/bin/env python3
"""
Demonstration script for Semantic Code Search (Member 2).

Demonstrates:
1. Repository/source files are obtained (Member 1 Ingestion).
2. Code is chunked via AST and line-based fallback (Member 2 Chunker + Member 3 Parser).
3. Embeddings are generated using Gemini Embedding 2 (or Mock test mode if offline).
4. FAISS index is created with 1-to-1 code metadata mapping.
5. Natural-language queries are evaluated across four key use cases:
   A. LOCATE FUNCTIONALITY ("Where is authentication implemented?")
   B. TRACE FUNCTIONALITY ("How does the login process work?")
   C. FIND CALLERS ("Which code calls authenticateUser?")
   D. CROSS-MODULE QUESTIONS ("Where is the database connection used?")
6. Saving and loading the FAISS index from disk.
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.structural import RepositoryScanner, TreeSitterParser
from backend.semantic import (
    CodeChunker,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    SemanticCodeSearch,
    create_embedding_provider,
)


def format_search_results(query: str, results: list, max_lines_preview: int = 12) -> None:
    """Print results in the standardized expected format."""
    print("=" * 60)
    print("SEMANTIC CODE SEARCH")
    print("=" * 60)
    print(f"\nQuery:\n{query}\n")
    print("Top Results:\n")

    if not results:
        print("  No relevant results found.")
        print("=" * 60)
        return

    for idx, r in enumerate(results, 1):
        file_path = r.get("file", "unknown")
        start_line = r.get("start_line", 0)
        end_line = r.get("end_line", 0)
        score = r.get("score", 0.0)
        code = r.get("code", "").strip()

        # Truncate code preview if too long for clean presentation
        code_lines = code.splitlines()
        if len(code_lines) > max_lines_preview:
            preview = "\n   ".join(code_lines[:max_lines_preview]) + "\n   ... [truncated]"
        else:
            preview = "\n   ".join(code_lines)

        print(f"{idx}. File: {file_path}")
        print(f"   Lines: {start_line}-{end_line}")
        print(f"   Score: {score:.4f}")
        print(f"   Code:\n   {preview}\n")

    print("=" * 60)


def run_demo(target_repo: Path, custom_query: str = None, force_mock: bool = False) -> None:
    """Run the complete Semantic Code Search demo."""
    print("*" * 70)
    print("AGENTIC CODE INTELLIGENCE — MEMBER 2: SEMANTIC CODE SEARCH")
    print("*" * 70)
    print(f"Target Repository: {target_repo}\n")

    # Step 1: Member 1 Repository Ingestion
    print("[Step 1] Ingesting repository source files (Member 1)...")
    scanner = RepositoryScanner(target_repo)
    source_files = scanner.scan()
    print(f"  Discovered {len(source_files)} source file(s):")
    for sf in source_files:
        print(f"    - {sf.path} ({sf.language}, {sf.line_count} lines)")

    # Step 2: Member 2 Code Chunking with Member 3 TreeSitterParser
    print("\n[Step 2] Chunking code into semantic boundaries (Member 2 + Member 3)...")
    parser = TreeSitterParser()
    chunker = CodeChunker(parser=parser)
    chunks = chunker.chunk_repository(source_files, parser=parser)
    print(f"  Created {len(chunks)} code chunk(s) preserving line numbers & AST metadata:")
    for ch in chunks[:6]:
        sym = f" ({ch.name})" if ch.name else ""
        print(f"    - [{ch.chunk_type}{sym}] {ch.file}:{ch.start_line}-{ch.end_line}")
    if len(chunks) > 6:
        print(f"    ... and {len(chunks) - 6} more chunks")

    # Step 3: Embedding Provider Initialization
    print("\n[Step 3] Initializing vector embeddings provider...")
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    if force_mock or not has_gemini_key:
        if not force_mock:
            print("  NOTICE: GEMINI_API_KEY not found in environment.")
            print("  To use Gemini Embedding 2 in production, configure:")
            print("    PowerShell:  $env:GEMINI_API_KEY = 'your_api_key'")
            print("    Linux/macOS: export GEMINI_API_KEY='your_api_key'")
            print("  Running with MockEmbeddingProvider for offline demonstration...\n")
        embedding_provider = MockEmbeddingProvider()
    else:
        print("  GEMINI_API_KEY detected. Using real Gemini Embedding 2 ('gemini-embedding-2').")
        embedding_provider = GeminiEmbeddingProvider()

    # Step 4: FAISS Vector Index Creation
    print("\n[Step 4] Generating embeddings and building FAISS vector index...")
    search_engine = SemanticCodeSearch(
        embedding_provider=embedding_provider,
        chunker=chunker,
        metric="cosine",
    )
    faiss_index = search_engine.index_chunks(chunks)
    print(f"  FAISS index successfully built with {faiss_index.total} vectors (dimension={faiss_index.dimension}).")

    # Step 5: Test Saving and Loading Index
    index_cache_dir = PROJECT_ROOT / ".cache" / "semantic_index"
    search_engine.save(index_cache_dir, name="demo_index")
    print(f"  Saved FAISS index + metadata to '{index_cache_dir}'.")
    # Verify index reloads cleanly
    reloaded_engine = SemanticCodeSearch(embedding_provider=embedding_provider, chunker=chunker)
    reloaded_engine.load(index_cache_dir, name="demo_index")
    print(f"  Reloaded FAISS index: {reloaded_engine.index.total} vectors verified.")

    # Step 6: Execute Showcase Queries (The 4 Required Use Cases)
    showcase_queries = [
        ("A. LOCATE FUNCTIONALITY", "Where is authentication implemented?"),
        ("B. TRACE FUNCTIONALITY", "How does the login process work?"),
        ("C. FIND CALLERS", "Which code calls authenticateUser?"),
        ("D. CROSS-MODULE QUESTIONS", "Where is the database connection used?"),
    ]

    print("\n[Step 6] Executing Semantic Code Search Use Cases...")

    if custom_query:
        print(f"\nExecuting Custom Query: {custom_query}")
        results = reloaded_engine.search(custom_query, top_k=3)
        format_search_results(custom_query, results)
    else:
        for category, query in showcase_queries:
            print(f"\n>>> USE CASE: {category}")
            results = reloaded_engine.search(query, top_k=2)
            format_search_results(query, results)

    print("\n" + "=" * 70)
    print("MEMBER 2: SEMANTIC CODE SEARCH DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Member 2 Semantic Code Search Demo")
    parser.add_argument(
        "--repo",
        type=str,
        default="demo_repo",
        help="Path to repository to index and search (default: demo_repo)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom natural-language search query",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline test mode using MockEmbeddingProvider",
    )

    args = parser.parse_args()
    target_repo = Path(args.repo)
    if not target_repo.is_absolute():
        target_repo = PROJECT_ROOT / target_repo

    run_demo(target_repo=target_repo, custom_query=args.query, force_mock=args.mock)


if __name__ == "__main__":
    main()
