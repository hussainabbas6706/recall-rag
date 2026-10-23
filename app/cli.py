"""Developer CLI for inspecting a vault before indexing it."""

from pathlib import Path

import typer

from app.config import get_settings
from app.ingest import find_markdown_files, parse_vault_chunks
from app.vectorstore import get_qdrant_client, index_chunks, search_chunks

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


@cli.command()
def index(vault_path: Path | None = None) -> None:
    """Parse and index vault notes into Qdrant."""
    settings = get_settings()
    path = vault_path or settings.vault_path
    typer.echo(f"Indexing vault from: {path}")
    parsed_chunks = parse_vault_chunks(path)
    typer.echo(
        f"Parsed {len(parsed_chunks)} chunk(s). Connecting to Qdrant at {settings.qdrant_url}..."
    )
    client = get_qdrant_client()
    count = index_chunks(client, collection_name=settings.qdrant_collection, chunks=parsed_chunks)
    typer.echo(f"Successfully indexed {count} chunk(s) into '{settings.qdrant_collection}'.")


@cli.command()
def stats() -> None:
    """Show status and point count of the Qdrant collection."""
    settings = get_settings()
    client = get_qdrant_client()
    if not client.collection_exists(settings.qdrant_collection):
        typer.echo(f"Collection '{settings.qdrant_collection}' does not exist in Qdrant yet.")
        return

    info = client.get_collection(settings.qdrant_collection)
    typer.echo(f"Collection: {settings.qdrant_collection}")
    typer.echo(f"Status:     {info.status}")
    typer.echo(f"Points:     {info.points_count}")
    vectors_config = info.config.params.vectors
    typer.echo(f"Vectors:    {vectors_config.size}d ({vectors_config.distance})")


@cli.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    top_k: int = typer.Option(
        3, "--top-k", "-k", min=1, max=20, help="Number of results to return"
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", "-t", min=0.0, max=1.0, help="Minimum cosine score cutoff"
    ),
) -> None:
    """Search indexed vault chunks by semantic similarity."""
    settings = get_settings()
    client = get_qdrant_client()
    typer.echo(f"Searching '{settings.qdrant_collection}' for: \"{query}\"")
    results = search_chunks(
        client=client,
        collection_name=settings.qdrant_collection,
        query=query,
        top_k=top_k,
        score_threshold=threshold,
    )

    if not results:
        typer.echo("No matching chunks found.")
        return

    typer.echo(f"Found {len(results)} matching chunk(s):\n")
    for i, result in enumerate(results, 1):
        crumbs = " > ".join(result.chunk.heading_path) if result.chunk.heading_path else "(root)"
        snippet = result.chunk.content.replace("\n", " ").strip()
        typer.echo(f"[{i}] Score: {result.score:.4f} | {result.chunk.source_path} > {crumbs}")
        typer.echo(f'    "{snippet}"\n')


if __name__ == "__main__":
    cli()
