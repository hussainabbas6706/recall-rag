"""Developer CLI for inspecting a vault before indexing it."""

from pathlib import Path

import typer

from app.config import get_settings
from app.ingest import find_markdown_files

cli = typer.Typer(help="Local development commands for Recall.")


@cli.command()
def inspect(vault_path: Path | None = None) -> None:
    """List Markdown notes that Recall will ingest."""
    path = vault_path or get_settings().vault_path
    files = find_markdown_files(path)
    typer.echo(f"Found {len(files)} Markdown file(s) in {path}.")
    for file in files:
        typer.echo(f"- {file.relative_to(path)}")


if __name__ == "__main__":
    cli()
