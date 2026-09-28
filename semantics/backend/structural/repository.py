"""
Repository scanner for JavaScript and TypeScript projects.

Discovers relevant source files while safely ignoring build artifacts,
package dependencies, and binary files.
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Union, Optional
import logging

from .models import SourceFile

logger = logging.getLogger(__name__)

# Centralized extension-to-language mapping
SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
}

# Directories to exclude from repository traversal
IGNORED_DIRECTORIES: Set[str] = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "coverage",
    ".next",
    "out",
    ".turbo",
    ".cache",
    "__pycache__",
    ".svn",
    ".hg",
}

# Initial buffer size used to detect binary files
BINARY_DETECTION_CHUNK_SIZE = 1024


def is_binary_content(buffer: bytes) -> bool:
    """Determine if a byte buffer represents binary content.

    Checks for null bytes, which typically indicate compiled or binary data.
    """
    return b"\x00" in buffer


class RepositoryScanner:
    """Recursively scans a repository directory to locate and load JavaScript/TypeScript files.

    Preserves repository-relative paths, computes line counts, detects languages,
    and skips dependency and build output folders.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        supported_extensions: Optional[Dict[str, str]] = None,
        ignored_directories: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the repository scanner.

        Args:
            root_path: Filesystem path to the root of the repository.
            supported_extensions: Optional custom extension mapping (defaults to JS/TS extensions).
            ignored_directories: Optional custom set of directory names to ignore.

        Raises:
            ValueError: If root_path does not exist or is not a directory.
        """
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise ValueError(f"Repository path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {self.root_path}")

        self.supported_extensions = (
            supported_extensions if supported_extensions is not None else SUPPORTED_EXTENSIONS
        )
        self.ignored_directories = (
            ignored_directories if ignored_directories is not None else IGNORED_DIRECTORIES
        )

    def is_ignored_directory(self, dir_name: str) -> bool:
        """Check if a directory name should be excluded from traversal."""
        return dir_name in self.ignored_directories or dir_name.startswith(".git")

    def get_language_for_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Determine the programming language from file extension.

        Args:
            file_path: File name or path.

        Returns:
            Language string if supported, else None.
        """
        suffix = Path(file_path).suffix.lower()
        return self.supported_extensions.get(suffix)

    def read_source_file(self, file_path: Path) -> Optional[SourceFile]:
        """Safely read and create a SourceFile instance.

        Performs UTF-8 decoding and skips binary or unreadable files.

        Args:
            file_path: Absolute path to the file.

        Returns:
            SourceFile instance if read successfully, None otherwise.
        """
        try:
            rel_path = file_path.relative_to(self.root_path).as_posix()
        except ValueError:
            rel_path = file_path.as_posix()

        language = self.get_language_for_file(file_path)
        if not language:
            return None

        try:
            # Check for binary content in the first chunk before full read
            with open(file_path, "rb") as f:
                header = f.read(BINARY_DETECTION_CHUNK_SIZE)
                if is_binary_content(header):
                    logger.warning("Skipping binary file: %s", rel_path)
                    return None

            # Read full content with UTF-8
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            byte_size = file_path.stat().st_size
            line_count = len(content.splitlines()) if content else 0

            return SourceFile(
                path=rel_path,
                language=language,
                content=content,
                line_count=line_count,
                abs_path=str(file_path),
                byte_size=byte_size,
            )
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF8 / unreadable file: %s", rel_path)
            return None
        except (PermissionError, OSError) as err:
            logger.warning("Error reading file %s: %s", rel_path, err)
            return None

    def scan(self) -> List[SourceFile]:
        """Recursively scan repository and return all discovered source files.

        Returns:
            List of SourceFile objects sorted deterministically by relative path.
        """
        discovered: List[SourceFile] = []

        for current_root, dirs, files in os.walk(self.root_path, topdown=True):
            # Prune ignored directories in-place so os.walk does not recurse into them
            dirs[:] = [d for d in dirs if not self.is_ignored_directory(d)]

            current_path = Path(current_root)

            for filename in files:
                file_path = current_path / filename
                suffix = file_path.suffix.lower()

                # Filter by supported extensions only
                if suffix not in self.supported_extensions:
                    continue

                source_file = self.read_source_file(file_path)
                if source_file is not None:
                    discovered.append(source_file)

        # Sort files deterministically by relative path
        discovered.sort(key=lambda sf: sf.path)
        return discovered
