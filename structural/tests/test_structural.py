"""
Unit test suite for Structural Code Intelligence (Stage 1).

Tests:
1. JavaScript file discovery (.js, .jsx)
2. TypeScript file discovery (.ts, .tsx)
3. Exclusion of ignored directories (node_modules, .git, dist, build, coverage, .next, out)
4. Rejection of unsupported files (.md, .json, binary files)
5. JavaScript parsing and AST node generation
6. TypeScript parsing and AST node generation
7. TSX / JSX parsing and AST node generation
8. Graceful handling of invalid / malformed source (does not crash, marks success=False)
9. Preservation of repository-relative file paths and source line numbers
10. AST traversal and node detail verification (type, start/end points, byte offsets)
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.structural import (
    RepositoryScanner,
    TreeSitterParser,
    SourceFile,
    ParseResult,
    NodeInfo,
    traverse_ast,
    get_node_info,
    SUPPORTED_EXTENSIONS,
    IGNORED_DIRECTORIES,
)


class TestRepositoryScanner(unittest.TestCase):
    """Test suite for RepositoryScanner."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)

        # Create directory hierarchy
        (self.repo_root / "src").mkdir(parents=True)
        (self.repo_root / "src" / "components").mkdir(parents=True)
        (self.repo_root / "node_modules" / "pkg").mkdir(parents=True)
        (self.repo_root / ".git" / "hooks").mkdir(parents=True)
        (self.repo_root / "dist").mkdir(parents=True)
        (self.repo_root / "build").mkdir(parents=True)
        (self.repo_root / "coverage").mkdir(parents=True)
        (self.repo_root / ".next").mkdir(parents=True)
        (self.repo_root / "out").mkdir(parents=True)

        # Create valid source files
        (self.repo_root / "src" / "index.js").write_text(
            "console.log('hello');\nconst x = 10;", encoding="utf-8"
        )
        (self.repo_root / "src" / "App.jsx").write_text(
            "export const App = () => <div>Hello</div>;", encoding="utf-8"
        )
        (self.repo_root / "src" / "types.ts").write_text(
            "export interface User { id: string; }", encoding="utf-8"
        )
        (self.repo_root / "src" / "components" / "Card.tsx").write_text(
            "export const Card = () => <section>Card</section>;", encoding="utf-8"
        )

        # Create ignored directory files
        (self.repo_root / "node_modules" / "pkg" / "index.js").write_text(
            "module.exports = {};", encoding="utf-8"
        )
        (self.repo_root / ".git" / "hooks" / "pre-commit.js").write_text(
            "console.log('git');", encoding="utf-8"
        )
        (self.repo_root / "dist" / "bundle.js").write_text(
            "var bundled = true;", encoding="utf-8"
        )
        (self.repo_root / "build" / "app.js").write_text(
            "var built = true;", encoding="utf-8"
        )
        (self.repo_root / "coverage" / "lcov.js").write_text(
            "var cov = 1;", encoding="utf-8"
        )
        (self.repo_root / ".next" / "server.js").write_text(
            "var next = true;", encoding="utf-8"
        )
        (self.repo_root / "out" / "static.js").write_text(
            "var out = true;", encoding="utf-8"
        )

        # Create unsupported files
        (self.repo_root / "README.md").write_text("# Readme", encoding="utf-8")
        (self.repo_root / "package.json").write_text('{"name": "test"}', encoding="utf-8")
        (self.repo_root / "src" / "binary.ts").write_bytes(
            b"\x00\x01\x02\x03BINARY_DATA\x00"
        )

        self.scanner = RepositoryScanner(self.repo_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_javascript_discovery(self) -> None:
        """Verify discovery of .js and .jsx files."""
        files = self.scanner.scan()
        paths = [f.path for f in files]
        self.assertIn("src/index.js", paths)
        self.assertIn("src/App.jsx", paths)

        js_file = next(f for f in files if f.path == "src/index.js")
        self.assertEqual(js_file.language, "javascript")
        self.assertEqual(js_file.line_count, 2)

        jsx_file = next(f for f in files if f.path == "src/App.jsx")
        self.assertEqual(jsx_file.language, "jsx")

    def test_typescript_discovery(self) -> None:
        """Verify discovery of .ts and .tsx files."""
        files = self.scanner.scan()
        paths = [f.path for f in files]
        self.assertIn("src/types.ts", paths)
        self.assertIn("src/components/Card.tsx", paths)

        ts_file = next(f for f in files if f.path == "src/types.ts")
        self.assertEqual(ts_file.language, "typescript")

        tsx_file = next(f for f in files if f.path == "src/components/Card.tsx")
        self.assertEqual(tsx_file.language, "tsx")

    def test_ignored_directories(self) -> None:
        """Verify all dependency and build directories are ignored."""
        files = self.scanner.scan()
        paths = [f.path for f in files]
        for path in paths:
            self.assertFalse(path.startswith("node_modules/"), f"Leaked: {path}")
            self.assertFalse(path.startswith(".git/"), f"Leaked: {path}")
            self.assertFalse(path.startswith("dist/"), f"Leaked: {path}")
            self.assertFalse(path.startswith("build/"), f"Leaked: {path}")
            self.assertFalse(path.startswith("coverage/"), f"Leaked: {path}")
            self.assertFalse(path.startswith(".next/"), f"Leaked: {path}")
            self.assertFalse(path.startswith("out/"), f"Leaked: {path}")

    def test_unsupported_files_ignored(self) -> None:
        """Verify unsupported file extensions (.md, .json) are ignored."""
        files = self.scanner.scan()
        paths = [f.path for f in files]
        self.assertNotIn("README.md", paths)
        self.assertNotIn("package.json", paths)

    def test_binary_file_ignored(self) -> None:
        """Verify binary files with valid extensions are safely skipped."""
        files = self.scanner.scan()
        paths = [f.path for f in files]
        self.assertNotIn("src/binary.ts", paths)

    def test_relative_paths_preserved(self) -> None:
        """Verify relative paths from repo root are preserved and normalized."""
        files = self.scanner.scan()
        for f in files:
            self.assertFalse(Path(f.path).is_absolute())
            self.assertNotIn("\\", f.path)  # Normalized with forward slash


class TestTreeSitterParser(unittest.TestCase):
    """Test suite for TreeSitterParser."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = TreeSitterParser()

    def test_javascript_parsing(self) -> None:
        """Verify parsing valid JavaScript code."""
        code = "function add(a, b) {\n  return a + b;\n}\nconst result = add(2, 3);"
        res = self.parser.parse_code(code, "javascript", "add.js")
        self.assertTrue(res.success)
        self.assertIsNone(res.error)
        self.assertEqual(res.root_node_type, "program")
        self.assertIsNotNone(res.root_node)

        # Check NodeInfo extraction
        summary = res.get_root_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary.node_type, "program")
        self.assertEqual(summary.start_point, (0, 0))

    def test_typescript_parsing(self) -> None:
        """Verify parsing valid TypeScript code."""
        code = (
            "interface Point {\n"
            "  x: number;\n"
            "  y: number;\n"
            "}\n"
            "function getOrigin(): Point {\n"
            "  return { x: 0, y: 0 };\n"
            "}"
        )
        res = self.parser.parse_code(code, "typescript", "point.ts")
        self.assertTrue(res.success)
        self.assertIsNone(res.error)
        self.assertEqual(res.root_node_type, "program")
        self.assertIsNotNone(res.root_node)

    def test_tsx_parsing(self) -> None:
        """Verify parsing valid TSX code."""
        code = (
            "import React from 'react';\n"
            "export const Header: React.FC<{ title: string }> = ({ title }) => {\n"
            "  return <h1>{title}</h1>;\n"
            "};"
        )
        res = self.parser.parse_code(code, "tsx", "Header.tsx")
        self.assertTrue(res.success)
        self.assertIsNone(res.error)
        self.assertEqual(res.root_node_type, "program")

    def test_invalid_source_does_not_crash(self) -> None:
        """Verify malformed JavaScript does not crash the parser and reports failure."""
        broken_code = "function broken( { let x = ; if ("
        res = self.parser.parse_code(broken_code, "javascript", "broken.js")
        self.assertFalse(res.success)
        self.assertIsNotNone(res.error)
        self.assertIn("Syntax error", res.error)
        # Root node is still accessible for partial inspection if needed
        self.assertIsNotNone(res.root_node)
        self.assertGreaterEqual(len(res.syntax_errors), 1)

    def test_unsupported_language_fails_gracefully(self) -> None:
        """Verify requesting an unsupported language returns failure without crashing."""
        res = self.parser.parse_code("print('hello')", "python", "script.py")
        self.assertFalse(res.success)
        self.assertIn("Unsupported language", res.error)

    def test_ast_traversal(self) -> None:
        """Verify AST traversal (root -> children -> nested nodes) preserves positional data."""
        code = "const x = 42;"
        res = self.parser.parse_code(code, "javascript", "const.js")
        self.assertTrue(res.success)

        nodes = list(traverse_ast(res.root_node))
        self.assertGreater(len(nodes), 1)

        types = [n.type for n in nodes]
        self.assertIn("program", types)
        self.assertIn("lexical_declaration", types)

        # Test get_node_info
        for node in nodes:
            info = get_node_info(node)
            self.assertIsInstance(info, NodeInfo)
            self.assertIsInstance(info.start_point, tuple)
            self.assertIsInstance(info.end_point, tuple)
            self.assertGreaterEqual(info.end_byte, info.start_byte)


if __name__ == "__main__":
    unittest.main()
