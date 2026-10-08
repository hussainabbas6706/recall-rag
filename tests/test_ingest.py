from pathlib import Path

from app.ingest import find_markdown_files


FIXTURE_VAULT = Path("tests/fixtures/vault")


def test_find_markdown_files_returns_fixture_notes() -> None:
    files = find_markdown_files(FIXTURE_VAULT)

    assert [file.relative_to(FIXTURE_VAULT).as_posix() for file in files] == [
        "career/job-applications.md",
        "projects/atlas.md",
        "projects/recall.md",
    ]
