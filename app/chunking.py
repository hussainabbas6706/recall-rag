"""Heading-aware parsing and chunking for Markdown documents."""

import re

import frontmatter
import yaml

from app.models import Chunk


def parse_heading_line(line: str) -> tuple[int, str] | None:
    """Parse a Markdown heading line, returning (level, title) or None.

    Matches 1 to 6 '#' characters followed by at least one space and title text.
    Lines without spaces after '#' (e.g. '#tag') or non-heading lines return None.
    """
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not match:
        return None
    level = len(match.group(1))
    title = match.group(2).strip()
    return level, title


def update_heading_stack(
    stack: list[tuple[int, str]], level: int, title: str
) -> list[tuple[int, str]]:
    """Return a new heading stack after adding a heading at the given level.

    Any existing headings at the same or deeper level are discarded.
    """
    new_stack = [item for item in stack if item[0] < level]
    new_stack.append((level, title))
    return new_stack


def strip_frontmatter(raw_text: str) -> str:
    """Strip YAML frontmatter from Markdown text, returning only the body."""
    try:
        post = frontmatter.loads(raw_text)
        return post.content
    except (ValueError, yaml.YAMLError):
        # Fallback if frontmatter parsing fails on unusual text
        return raw_text


def extract_markdown_sections(raw_text: str) -> list[tuple[list[str], str]]:
    """Parse Markdown text into sections based on heading hierarchy.

    Returns a list of tuples: (heading_path, section_content).
    - Ignores headings inside fenced code blocks.
    - Handles text before the first heading (with heading_path=[]).
    - Discards empty or whitespace-only sections.
    """
    clean_text = strip_frontmatter(raw_text)
    lines = clean_text.splitlines()

    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_code_block = False

    def flush_section() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            path = [title for _, title in heading_stack]
            sections.append((path, content))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block:
            heading = parse_heading_line(line)
            if heading is not None:
                flush_section()
                current_lines = []
                level, title = heading
                heading_stack = update_heading_stack(heading_stack, level, title)
                continue

        current_lines.append(line)

    flush_section()
    return sections


def split_oversized_text(text: str, max_chars: int) -> list[str]:
    """Split a single oversized block (e.g. long paragraph) into bounded pieces."""
    # First try splitting by sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return [text]

    pieces: list[str] = []
    current_piece = ""

    for sentence in sentences:
        # If an individual sentence exceeds max_chars, hard-slice it
        if len(sentence) > max_chars:
            if current_piece:
                pieces.append(current_piece)
                current_piece = ""
            for i in range(0, len(sentence), max_chars):
                pieces.append(sentence[i : i + max_chars])
            continue

        if not current_piece:
            current_piece = sentence
        elif len(current_piece) + 1 + len(sentence) <= max_chars:
            current_piece = f"{current_piece} {sentence}"
        else:
            pieces.append(current_piece)
            current_piece = sentence

    if current_piece:
        pieces.append(current_piece)

    return pieces


def split_section_content(content: str, max_chars: int = 1000) -> list[str]:
    """Split section content into chunks bounded by max_chars.

    Preserves paragraph boundaries (\\n\\n) whenever possible. Groups short
    paragraphs together until adding another would exceed max_chars.
    """
    clean_content = content.strip()
    if not clean_content:
        return []

    if len(clean_content) <= max_chars:
        return [clean_content]

    paragraphs = [p.strip() for p in clean_content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_group: list[str] = []
    current_len = 0

    for para in paragraphs:
        # If an individual paragraph exceeds max_chars, split it specifically
        if len(para) > max_chars:
            if current_group:
                chunks.append("\n\n".join(current_group))
                current_group = []
                current_len = 0
            chunks.extend(split_oversized_text(para, max_chars))
            continue

        para_len = len(para)
        # Accounting for 2 characters of "\n\n" between paragraphs
        additional_len = para_len if not current_group else 2 + para_len

        if current_len + additional_len <= max_chars:
            current_group.append(para)
            current_len += additional_len
        else:
            chunks.append("\n\n".join(current_group))
            current_group = [para]
            current_len = para_len

    if current_group:
        chunks.append("\n\n".join(current_group))

    return chunks


def chunk_markdown(
    source_path: str,
    raw_text: str,
    max_chars: int = 1000,
) -> list[Chunk]:
    """Parse and chunk raw Markdown content into structured Chunk models.

    Preserves source-relative path, hierarchical heading path, and assigns
    a deterministic 0-based chunk_index for each chunk in the file.
    """
    sections = extract_markdown_sections(raw_text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for heading_path, section_content in sections:
        text_pieces = split_section_content(section_content, max_chars=max_chars)
        for piece in text_pieces:
            chunks.append(
                Chunk(
                    source_path=source_path,
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                    content=piece,
                )
            )
            chunk_index += 1

    return chunks


