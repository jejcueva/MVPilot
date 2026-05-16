from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

TimelineStatus = Literal["pending", "running", "completed", "failed"]

# Product-facing phases shown in the Mission Control UI.
BUILD_TIMELINE_PHASES: tuple[dict[str, str], ...] = (
    {"id": "understand_idea", "title": "Understanding user idea", "category": "planning"},
    {"id": "extract_requirements", "title": "Extracting MVP requirements", "category": "planning"},
    {"id": "plan_architecture", "title": "Planning architecture", "category": "planning"},
    {"id": "decompose_tasks", "title": "Decomposing tasks", "category": "planning"},
    {"id": "create_repo_structure", "title": "Creating repo structure", "category": "github"},
    {"id": "generate_frontend", "title": "Generating frontend", "category": "code"},
    {"id": "generate_backend", "title": "Generating backend/API layer", "category": "code"},
    {"id": "add_mock_integrations", "title": "Adding labeled mock integrations", "category": "code"},
    {"id": "create_documentation", "title": "Creating documentation", "category": "code"},
    {"id": "validate_output", "title": "Validating idea-specific output", "category": "validation"},
    {"id": "finalize_repo", "title": "Finalizing repo", "category": "delivery"},
)


def default_build_timeline() -> list[dict[str, Any]]:
    return [
        {
            "id": phase["id"],
            "title": phase["title"],
            "category": phase["category"],
            "status": "pending",
            "detail": "",
            "artifacts": [],
            "updated_at": None,
        }
        for phase in BUILD_TIMELINE_PHASES
    ]


def timeline_event(
    *,
    phase_id: str,
    status: TimelineStatus,
    detail: str,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    title = next(
        (phase["title"] for phase in BUILD_TIMELINE_PHASES if phase["id"] == phase_id),
        phase_id.replace("_", " ").title(),
    )
    category = next(
        (phase["category"] for phase in BUILD_TIMELINE_PHASES if phase["id"] == phase_id),
        "planning",
    )
    return {
        "id": phase_id,
        "title": title,
        "category": category,
        "status": status,
        "detail": detail,
        "artifacts": list(artifacts or []),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def apply_timeline_events(
    current: list[dict[str, Any]] | None,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {item["id"]: dict(item) for item in (current or default_build_timeline())}
    for event in events:
        phase_id = str(event.get("id") or "")
        if not phase_id:
            continue
        existing = indexed.get(phase_id, timeline_event(phase_id=phase_id, status="pending", detail=""))
        indexed[phase_id] = {**existing, **event, "id": phase_id}
    ordered_ids = [phase["id"] for phase in BUILD_TIMELINE_PHASES]
    return [indexed[phase_id] for phase_id in ordered_ids if phase_id in indexed]
