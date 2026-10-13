"""Vault discovery and Markdown parsing into structured chunks."""

from pathlib import Path

from app.chunking import chunk_markdown
from app.models import Chunk

IGNORED_DIRECTORY_NAMES = {".git", ".obsidian", ".trash"}


def find_markdown_files(vault_path: Path) -> list[Path]:
    """Return Markdown files, excluding Obsidian and VCS implementation directories."""
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")
    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")

    return sorted(
        path
        for path in vault_path.rglob("*.md")
        if not any(parent.name in IGNORED_DIRECTORY_NAMES for parent in path.parents)
    )


def parse_vault_chunks(vault_path: Path) -> list[Chunk]:
    """Scan vault and parse all Markdown notes into structured chunks."""
    files = find_markdown_files(vault_path)
    all_chunks: list[Chunk] = []

    for file_path in files:
        relative_path = file_path.relative_to(vault_path).as_posix()
        raw_text = file_path.read_text(encoding="utf-8")
        file_chunks = chunk_markdown(source_path=relative_path, raw_text=raw_text)
        all_chunks.extend(file_chunks)

    return all_chunks

