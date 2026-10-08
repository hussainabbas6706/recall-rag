"""Phase-1 ingestion discovery; embedding and Qdrant upserts follow in Phase 2."""

from pathlib import Path


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
