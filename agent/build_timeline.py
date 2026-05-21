from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

TimelineStatus = Literal["pending", "running", "completed", "failed"]


BUILD_TIMELINE_PHASES: tuple[dict[str, str], ...] = (
    {"id": "idea_intake", "title": "Idea intake", "category": "planning"},
    {"id": "requirement_expansion", "title": "Requirement expansion", "category": "planning"},
    {"id": "domain_research", "title": "Domain research", "category": "research"},
    {"id": "reference_url_analysis", "title": "Reference URL analysis", "category": "research"},
    {"id": "user_goal_interpretation", "title": "User goal interpretation", "category": "planning"},
    {"id": "feature_system_design", "title": "Feature system design", "category": "architecture"},
    {
        "id": "tech_stack_recommendation",
        "title": "Tech stack recommendation",
        "category": "planning",
    },
    {"id": "data_model_design", "title": "Data model design", "category": "architecture"},
    {"id": "api_design", "title": "API design", "category": "architecture"},
    {"id": "frontend_architecture", "title": "Frontend architecture", "category": "architecture"},
    {"id": "backend_architecture", "title": "Backend architecture", "category": "architecture"},
    {"id": "auth_authorization_design", "title": "Auth and authorization design", "category": "architecture"},
    {"id": "database_schema_planning", "title": "Database schema planning", "category": "architecture"},
    {"id": "file_tree_generation", "title": "File tree generation", "category": "code"},
    {"id": "code_implementation", "title": "Code implementation", "category": "code"},
    {"id": "testing_strategy", "title": "Testing strategy", "category": "validation"},
    {"id": "documentation_generation", "title": "Documentation generation", "category": "code"},
    {"id": "deployment_instructions", "title": "Deployment instructions", "category": "delivery"},
    {"id": "github_repo_export", "title": "GitHub repo export", "category": "github"},
    {"id": "build_test_validation", "title": "Build and test validation", "category": "validation"},
    {"id": "final_project_report", "title": "Final project report", "category": "delivery"},
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
