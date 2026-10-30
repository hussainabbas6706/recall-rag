"""Synthetic vault builder for offline retrieval evaluation."""

from pathlib import Path

SYNTHETIC_VAULT_FILES: dict[str, str] = {
    "notes/🗓️ WEEKLY GOALS.md": """# Weekly Goals — Check off as you go

## Launch Checklist

- Confirm the launch checklist is complete and verified
- Review the final sign-off with stakeholders
- Publish the release notes for the engineering team

## Weekly Rhythm

- Track the sprint work in the delivery log
- Summarize progress before Friday standup
- Review open pull requests every Tuesday morning

## Week 9 — Agent from Scratch

- ReAct agent built with pure Python and OpenAI, zero frameworks
- Manual loop implementation: observe, think, act
- Function calling with 3 tools working
- Memory: in-context dict and Redis both implemented

## Week 16 — Project Ship and Freeze

- Final QA pass on the full multi-user pipeline
- Note: The hard code freeze takes effect at midnight before the final sprint
- Deploy production containers to staging cluster
- Record 5-minute Loom demo and write technical README
""",
    "notes/daily-routine.md": """# Daily Routine

## Morning Review

- Check the next delivery window on the dashboard
- Note blockers before the 10:00 AM standup
- Review high priority bug reports

## Evening Recap

- Update the task list for the next day
- Log follow-ups from the daily review
- Clear urgent Slack notifications
""",
    "notes/master-timeline.md": """# Master Timeline

## Milestones

- Kickoff starts after the planning review
- Release goes live after final QA and stakeholder sign-off
- Follow-up review closes the project retrospective

## Sprint Boundaries

- Sprint 1: Foundation and local embeddings setup
- Sprint 2: Scoped retrieval and metadata routing
- Sprint 3: Hybrid search and cross-encoder reranking
""",
    "notes/project-plan.md": """# Project Plan

## Decision Log

- Keep the launch plan local to the repository notes
- Confirm the final review before publishing
- Use local CPU embeddings to avoid external rate limits

## Risk Register

- Status depends on the launch checklist and review timing
- Dense vector search may miss exact keyword mentions in long notes
- Memory constraints on local machines during batch embedding
""",
    "notes/architecture-decisions.md": """# Architecture Decisions

## Framework Selection

- FastAPI provides asynchronous request handling, OpenAPI documentation, and typed validation with Pydantic
- Pure Python scripts are preferred for standalone agents and algorithmic prototypes to avoid framework overhead

## State and Memory Storage

- Redis is chosen for persistent session memory across agent restarts
- In-memory dict is used for short-lived scratchpad operations during a single chat turn
""",
    "notes/archive/old-plan.md": """# Old Plan

## Archive

- This note is historical and not part of the current launch plan
- Deprecated legacy milestones from the Q1 exploratory phase
""",
}


def build_synthetic_vault(target_dir: Path) -> Path:
    """Materialize the synthetic evaluation vault at the specified directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in SYNTHETIC_VAULT_FILES.items():
        file_path = target_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content.strip() + "\n", encoding="utf-8")
    return target_dir
