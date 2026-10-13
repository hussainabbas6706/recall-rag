"""Developer CLI for inspecting a vault before indexing it."""

from pathlib import Path

import typer

from app.config import get_settings
from app.ingest import find_markdown_files, parse_vault_chunks

cli = typer.Typer(help="Local development commands for Recall.")


@cli.callback()
def main() -> None:
    """Run a Recall development command."""


@cli.command()
def inspect(vault_path: Path | None = None) -> None:
    """List Markdown notes that Recall will ingest."""
    path = vault_path or get_settings().vault_path
    files = find_markdown_files(path)
    typer.echo(f"Found {len(files)} Markdown file(s) in {path}.")
    for file in files:
        typer.echo(f"- {file.relative_to(path)}")


@cli.command()
def chunks(vault_path: Path | None = None) -> None:
    """List parsed chunks with their heading breadcrumbs."""
    path = vault_path or get_settings().vault_path
    parsed_chunks = parse_vault_chunks(path)
    typer.echo(f"Parsed {len(parsed_chunks)} chunk(s) from {path}.")
    for chunk in parsed_chunks:
        crumbs = " > ".join(chunk.heading_path) if chunk.heading_path else "(root)"
        snippet = chunk.content.replace("\n", " ")[:60]
        typer.echo(f"- [{chunk.source_path}#{chunk.chunk_index}] [{crumbs}]: {snippet}...")



if __name__ == "__main__":
    cli()
