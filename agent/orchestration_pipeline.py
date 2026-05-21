from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from agent.build_timeline import BUILD_TIMELINE_PHASES, apply_timeline_events, timeline_event
from agent.project_agents import PROJECT_AGENTS, agent_name, project_agent_manifest

TimelineStatus = Literal["pending", "running", "completed", "failed"]

# Canonical pipeline phase ids (must match agent/build_timeline.py).
IDEA_INTAKE = "idea_intake"
REQUIREMENT_EXPANSION = "requirement_expansion"
DOMAIN_RESEARCH = "domain_research"
REFERENCE_URL_ANALYSIS = "reference_url_analysis"
USER_GOAL_INTERPRETATION = "user_goal_interpretation"
FEATURE_SYSTEM_DESIGN = "feature_system_design"
TECH_STACK_RECOMMENDATION = "tech_stack_recommendation"
DATA_MODEL_DESIGN = "data_model_design"
API_DESIGN = "api_design"
FRONTEND_ARCHITECTURE = "frontend_architecture"
BACKEND_ARCHITECTURE = "backend_architecture"
AUTH_AUTHORIZATION_DESIGN = "auth_authorization_design"
DATABASE_SCHEMA_PLANNING = "database_schema_planning"
FILE_TREE_GENERATION = "file_tree_generation"
CODE_IMPLEMENTATION = "code_implementation"
TESTING_STRATEGY = "testing_strategy"
DOCUMENTATION_GENERATION = "documentation_generation"
DEPLOYMENT_INSTRUCTIONS = "deployment_instructions"
GITHUB_REPO_EXPORT = "github_repo_export"
BUILD_TEST_VALIDATION = "build_test_validation"
FINAL_PROJECT_REPORT = "final_project_report"

PIPELINE_STAGE_ORDER: tuple[str, ...] = tuple(phase["id"] for phase in BUILD_TIMELINE_PHASES)

# Maps LangGraph node names to primary agent keys.
NODE_AGENT_KEYS: dict[str, str] = {
    "receive_idea": "product_strategist",
    "exchange_github_code": "github",
    "retrieve_context": "research_rag",
    "scope_mvp": "product_strategist",
    "recommend_stack": "stack_selector",
    "plan_repo": "system_architect",
    "create_repo": "github",
    "generate_files": "backend",
    "validate_mvp": "qa",
    "commit_progress": "github",
    "verify_build": "qa",
    "handle_blocker": "qa",
    "generate_final_package": "documentation",
    "remember_outcome": "logger",
    "report_result": "logger",
    "failed": "logger",
}


def pipeline_manifest() -> dict[str, Any]:
    return {
        "stages": [dict(phase) for phase in BUILD_TIMELINE_PHASES],
        "agents": project_agent_manifest(),
    }


def agent_log_entry(
    *,
    agent_key: str,
    stage_id: str,
    message: str,
    status: str = "completed",
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "agent_key": agent_key,
        "agent_name": agent_name(agent_key),
        "stage_id": stage_id,
        "status": status,
        "message": message,
        "detail": detail or message,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def append_agent_logs(
    current: list[dict[str, Any]] | None,
    *entries: dict[str, Any],
) -> list[dict[str, Any]]:
    logs = list(current or [])
    logs.extend(entries)
    return logs[-500:]


def record_phases(
    orchestrator: Any,
    state: dict[str, Any],
    *events: tuple[str, TimelineStatus, str, list[str] | None],
) -> dict[str, Any]:
    """Record multiple timeline phases; returns merged build_timeline update."""
    timeline = list(state.get("build_timeline") or [])
    for phase_id, status, detail, artifacts in events:
        event = timeline_event(
            phase_id=phase_id,
            status=status,
            detail=detail,
            artifacts=artifacts,
        )
        timeline = apply_timeline_events(timeline, [event])
    return {"build_timeline": timeline}


def log_node_activity(
    *,
    state: dict[str, Any],
    node_name: str,
    stage_id: str,
    message: str,
    status: str = "completed",
    detail: str | None = None,
) -> dict[str, Any]:
    agent_key = NODE_AGENT_KEYS.get(node_name, "logger")
    entry = agent_log_entry(
        agent_key=agent_key,
        stage_id=stage_id,
        message=message,
        status=status,
        detail=detail,
    )
    return {"agent_logs": append_agent_logs(state.get("agent_logs"), entry)}


def agent_for_node(node_name: str) -> str:
    return agent_name(NODE_AGENT_KEYS.get(node_name, "logger"))


def list_project_agents() -> tuple[Any, ...]:
    return PROJECT_AGENTS
