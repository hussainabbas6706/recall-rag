"""Vault discovery and Markdown parsing into structured chunks."""

import logging
import os
from pathlib import Path
from time import perf_counter

from app.chunking import chunk_markdown
from app.models import Chunk, VaultEntry

logger = logging.getLogger(__name__)

IGNORED_DIRECTORY_NAMES = {".git", ".obsidian", ".trash"}


def _raise_walk_error(error: OSError) -> None:
    raise error


def list_vault_entries(vault_path: Path) -> list[VaultEntry]:
    """List safe, vault-relative folders and Markdown files without reading notes."""
    try:
        vault_root = vault_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError("Configured vault path does not exist.") from error

    if not vault_root.is_dir():
        raise NotADirectoryError("Configured vault path is not a directory.")

    entries: list[VaultEntry] = []
    for current, directory_names, file_names in os.walk(
        vault_root,
        topdown=True,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        current_path = Path(current)
        safe_directory_names = []
        for name in sorted(directory_names):
            directory = current_path / name
            if (
                name.startswith(".")
                or name in IGNORED_DIRECTORY_NAMES
                or directory.is_symlink()
            ):
                continue
            try:
                if not directory.resolve(strict=True).is_relative_to(vault_root):
                    continue
            except (OSError, RuntimeError):
                continue
            safe_directory_names.append(name)
            entries.append(
                VaultEntry(
                    path=directory.relative_to(vault_root).as_posix(),
                    kind="folder",
                )
            )
        directory_names[:] = safe_directory_names

        for name in sorted(file_names):
            file_path = current_path / name
            if name.startswith(".") or file_path.suffix.lower() != ".md":
                continue
            if file_path.is_symlink():
                continue
            try:
                resolved_file = file_path.resolve(strict=True)
                if not resolved_file.is_relative_to(vault_root) or not resolved_file.is_file():
                    continue
            except (OSError, RuntimeError):
                continue
            entries.append(
                VaultEntry(
                    path=file_path.relative_to(vault_root).as_posix(),
                    kind="file",
                )
            )

    return sorted(entries, key=lambda entry: (entry.path.casefold(), entry.path))


def find_markdown_files(vault_path: Path) -> list[Path]:
    """Return safe Markdown files inside the vault, excluding internal paths."""
    if not vault_path.exists():
        raise FileNotFoundError("Configured vault path does not exist.")
    if not vault_path.is_dir():
        raise NotADirectoryError("Configured vault path is not a directory.")

    started = perf_counter()
    files = [
        vault_path / entry.path
        for entry in list_vault_entries(vault_path)
        if entry.kind == "file"
    ]
    logger.info(
        "timing stage=ingestion.discovery files=%d duration_ms=%.2f",
        len(files),
        (perf_counter() - started) * 1000,
    )
    return files


def parse_vault_chunks(vault_path: Path) -> list[Chunk]:
    """Scan vault and parse all Markdown notes into structured chunks."""
    files = find_markdown_files(vault_path)
    all_chunks: list[Chunk] = []
    file_read_seconds = 0.0
    chunking_seconds = 0.0

    for file_path in files:
        relative_path = file_path.relative_to(vault_path).as_posix()
        read_started = perf_counter()
        raw_text = file_path.read_text(encoding="utf-8")
        file_read_seconds += perf_counter() - read_started

        chunking_started = perf_counter()
        file_chunks = chunk_markdown(source_path=relative_path, raw_text=raw_text)
        chunking_seconds += perf_counter() - chunking_started
        all_chunks.extend(file_chunks)

    logger.info(
        "timing stage=ingestion.parse files=%d chunks=%d file_read_ms=%.2f chunking_ms=%.2f",
        len(files),
        len(all_chunks),
        file_read_seconds * 1000,
        chunking_seconds * 1000,
    )
    return all_chunks
