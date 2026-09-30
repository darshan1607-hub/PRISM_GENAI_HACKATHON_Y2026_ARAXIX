#!/usr/bin/env python3
"""
Demonstration script for Structural Code Intelligence (Stage 1).

Demonstrates:
1. Repository scanning with file filtering and directory exclusion.
2. Safe UTF-8 source file ingestion.
3. Tree-Sitter parsing across JavaScript, TypeScript, and TSX.
4. Graceful handling of syntax errors and broken files.
5. Handling and rejection of unsupported file types.
6. AST traversal (root -> children -> nested nodes) for downstream analysis.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.structural import (
    RepositoryScanner,
    TreeSitterParser,
    traverse_ast,
    get_node_info,
)


def run_demo() -> None:
    demo_repo_dir = PROJECT_ROOT / "demo_repo"
    print("=" * 70)
    print("AGENTIC CODE INTELLIGENCE — STRUCTURAL CODE INTELLIGENCE (STAGE 1)")
    print("Repository Ingestion & Tree-Sitter Parser Demonstration")
    print("=" * 70)
    print(f"Target Repository: {demo_repo_dir}\n")

    # 1. Initialize RepositoryScanner
    print("[1] Initializing RepositoryScanner...")
    scanner = RepositoryScanner(demo_repo_dir)

    # 2. Discover and ingest source files
    print("[2] Scanning repository for JS/TS source files...")
    files = scanner.scan()

    print(f"\nDiscovered {len(files)} supported source file(s):")
    for f in files:
        print(f"  - {f.path} ({f.language}, {f.line_count} lines)")

    # 3. Parse files using TreeSitterParser
    print("\n[3] Parsing discovered source files using Tree-Sitter...")
    parser = TreeSitterParser()

    for sf in files:
        print("\n" + "-" * 50)
        result = parser.parse(sf)
        print(f"File: {result.file_path}")
        print(f"Language: {result.language}")
        print(f"Lines: {sf.line_count}")
        print(f"Parse successful: {result.success}")
        print(f"Root node type: {result.root_node_type}")

        if not result.success:
            print(f"Error details: {result.error}")
        else:
            root = result.root_node
            child_count = len(root.children) if root else 0
            print(f"Top-level child count: {child_count}")

    # 4. Demonstrate handling of unsupported file
    print("\n" + "=" * 70)
    print("[4] Demonstrating Handling of Unsupported File Types")
    print("=" * 70)
    readme_path = demo_repo_dir / "README.md"
    detected_lang = scanner.get_language_for_file(readme_path)
    print(f"File: README.md")
    print(f"Scanner language check: {detected_lang} (Skipped by repository scanner)")

    unsupported_parse = parser.parse_code(
        code="# Sample Markdown",
        language="markdown",
        file_path="README.md",
    )
    print(f"Attempt direct parse: success={unsupported_parse.success}")
    print(f"Parser error message: {unsupported_parse.error}")

    # 5. Demonstrate AST Traversal for Downstream Stages
    print("\n" + "=" * 70)
    print("[5] Demonstrating AST Traversal (Root -> Children -> Nested Nodes)")
    print("=" * 70)
    app_file = next((f for f in files if f.path == "src/app.js"), None)
    if app_file:
        app_result = parser.parse(app_file)
        if app_result.success and app_result.root_node:
            print(f"Traversing AST nodes for '{app_result.file_path}':")
            node_count = 0
            for node in traverse_ast(app_result.root_node):
                info = get_node_info(node)
                if node_count < 10:  # Print first 10 nodes as sample
                    print(
                        f"  [{info.node_type}] "
                        f"Lines {info.start_line}:{info.start_point[1]} -> "
                        f"{info.end_line}:{info.end_point[1]} "
                        f"(bytes {info.start_byte}..{info.end_byte})"
                    )
                node_count += 1
            print(f"  ... total AST nodes traversed: {node_count}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
