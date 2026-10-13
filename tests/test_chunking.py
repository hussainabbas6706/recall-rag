"""Unit tests for heading-aware parsing and chunking."""

from app.chunking import (
    chunk_markdown,
    extract_markdown_sections,
    parse_heading_line,
    split_section_content,
    strip_frontmatter,
    update_heading_stack,
)


def test_parse_heading_line_valid_levels() -> None:
    assert parse_heading_line("# Title") == (1, "Title")
    assert parse_heading_line("## Subheading") == (2, "Subheading")
    assert parse_heading_line("### Deep Heading") == (3, "Deep Heading")
    assert parse_heading_line("###### Level 6") == (6, "Level 6")


def test_parse_heading_line_invalid_or_non_headings() -> None:
    assert parse_heading_line("#NoSpace") is None
    assert parse_heading_line("Regular prose text") is None
    assert parse_heading_line("Text with # hashtag") is None
    assert parse_heading_line("####### Too many hashes") is None
    assert parse_heading_line("") is None


def test_update_heading_stack_transitions() -> None:
    stack: list[tuple[int, str]] = []

    # 1. Add Level 1
    stack = update_heading_stack(stack, 1, "Guide")
    assert [title for _, title in stack] == ["Guide"]

    # 2. Add Level 2
    stack = update_heading_stack(stack, 2, "Setup")
    assert [title for _, title in stack] == ["Guide", "Setup"]

    # 3. Add Level 3
    stack = update_heading_stack(stack, 3, "Windows")
    assert [title for _, title in stack] == ["Guide", "Setup", "Windows"]

    # 4. Sibling Level 2 replaces Setup & Windows
    stack = update_heading_stack(stack, 2, "Usage")
    assert [title for _, title in stack] == ["Guide", "Usage"]

    # 5. New Level 1 replaces everything
    stack = update_heading_stack(stack, 1, "Summary")
    assert [title for _, title in stack] == ["Summary"]


def test_update_heading_stack_skipped_level() -> None:
    stack = update_heading_stack([], 1, "Top")
    stack = update_heading_stack(stack, 3, "Skipped L2")
    assert [title for _, title in stack] == ["Top", "Skipped L2"]


def test_strip_frontmatter() -> None:
    markdown = """---
tags: [career, search]
status: active
---

# Title
Body text.
"""
    cleaned = strip_frontmatter(markdown)
    assert "tags:" not in cleaned
    assert "status:" not in cleaned
    assert "# Title" in cleaned
    assert "Body text." in cleaned


def test_extract_markdown_sections_with_pre_heading_content() -> None:
    markdown = """---
tags: [notes]
---
Introduction before any heading.

# Overview
This is the overview.

## Details
These are the details.
"""
    sections = extract_markdown_sections(markdown)
    assert len(sections) == 3

    # Section 0: before first heading
    assert sections[0][0] == []
    assert sections[0][1] == "Introduction before any heading."

    # Section 1: under Overview
    assert sections[1][0] == ["Overview"]
    assert sections[1][1] == "This is the overview."

    # Section 2: under Overview > Details
    assert sections[2][0] == ["Overview", "Details"]
    assert sections[2][1] == "These are the details."


def test_extract_markdown_sections_ignores_code_block_hashes() -> None:
    markdown = """# Code Examples

Here is some Python:
```python
# This is a comment, not a heading!
def add(a, b):
    return a + b
```

After the code block.
"""
    sections = extract_markdown_sections(markdown)
    assert len(sections) == 1
    assert sections[0][0] == ["Code Examples"]
    assert "# This is a comment, not a heading!" in sections[0][1]


def test_extract_markdown_sections_no_headings() -> None:
    markdown = "Just a quick note without any headings at all."
    sections = extract_markdown_sections(markdown)
    assert len(sections) == 1
    assert sections[0][0] == []
    assert sections[0][1] == "Just a quick note without any headings at all."


def test_extract_markdown_sections_empty_content() -> None:
    assert extract_markdown_sections("") == []
    assert extract_markdown_sections("   \n\n  \t ") == []


def test_split_section_content_under_limit() -> None:
    text = "Short section content that is well under the default limit."
    chunks = split_section_content(text, max_chars=1000)
    assert chunks == [text]


def test_split_section_content_packs_paragraphs() -> None:
    para1 = "First short paragraph."
    para2 = "Second short paragraph."
    combined = f"{para1}\n\n{para2}"

    # Both fit inside 100 chars
    chunks = split_section_content(combined, max_chars=100)
    assert chunks == [combined]


def test_split_section_content_splits_at_paragraph_boundaries() -> None:
    # 3 paragraphs of ~40 chars each
    para1 = "Paragraph 1: Introduction to the topic."
    para2 = "Paragraph 2: Detailed explanation text."
    para3 = "Paragraph 3: Concluding final summary."
    full_text = f"{para1}\n\n{para2}\n\n{para3}"

    # max_chars=90: para1 + para2 = 39 + 2 + 39 = 80 chars (fits!).
    # adding para3 would be 80 + 2 + 38 = 120 chars (exceeds 90!).
    chunks = split_section_content(full_text, max_chars=90)
    assert len(chunks) == 2
    assert chunks[0] == f"{para1}\n\n{para2}"
    assert chunks[1] == para3


def test_split_section_content_oversized_paragraph() -> None:
    # Single paragraph with 3 sentences, total ~130 chars
    sentence1 = "This is sentence one of the paragraph."
    sentence2 = "This is sentence two of the paragraph."
    sentence3 = "This is sentence three of the paragraph."
    para = f"{sentence1} {sentence2} {sentence3}"

    # max_chars=80 forces splitting the single paragraph by sentences
    chunks = split_section_content(para, max_chars=80)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 80
    # verify no content was lost
    assert sentence1 in chunks[0]
    assert sentence3 in chunks[-1]


def test_split_section_content_empty() -> None:
    assert split_section_content("") == []
    assert split_section_content("   \n\n   ") == []


def test_chunk_markdown_structure_and_metadata() -> None:
    markdown = """---
tags: [test]
---
Note lead intro.

# Main Title

First section body under main title.

## Nested Topic

Nested subsection body.
"""
    chunks = chunk_markdown(source_path="notes/sample.md", raw_text=markdown)

    assert len(chunks) == 3

    assert chunks[0].source_path == "notes/sample.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].heading_path == []
    assert chunks[0].content == "Note lead intro."

    assert chunks[1].source_path == "notes/sample.md"
    assert chunks[1].chunk_index == 1
    assert chunks[1].heading_path == ["Main Title"]
    assert chunks[1].content == "First section body under main title."

    assert chunks[2].source_path == "notes/sample.md"
    assert chunks[2].chunk_index == 2
    assert chunks[2].heading_path == ["Main Title", "Nested Topic"]
    assert chunks[2].content == "Nested subsection body."


def test_chunk_markdown_sequential_indices_across_splits() -> None:
    # A single section that will be split into multiple chunks due to max_chars
    para1 = "Paragraph 1 with enough text."
    para2 = "Paragraph 2 with more text."
    markdown = f"# Title\n\n{para1}\n\n{para2}"

    chunks = chunk_markdown(source_path="doc.md", raw_text=markdown, max_chars=40)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].heading_path == ["Title"]
    assert chunks[1].chunk_index == 1
    assert chunks[1].heading_path == ["Title"]


def test_chunk_markdown_empty_file() -> None:
    assert chunk_markdown(source_path="empty.md", raw_text="") == []
    assert chunk_markdown(source_path="blank.md", raw_text="   \n\n  ") == []


