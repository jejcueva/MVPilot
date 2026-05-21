from __future__ import annotations

from typing import Any

from agent.build_timeline import (
    BUILD_TIMELINE_PHASES,
    apply_timeline_events,
    default_build_timeline,
    timeline_event,
)
from agent.config import Settings
from agent.openclaw_runtime import openclaw_runtime_ready, registered_openclaw_tools
from agent.orchestration_pipeline import pipeline_manifest


def _planned_file_path(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("path") or item.get("name") or item)
    return str(item)


def _stack_label(stack: dict[str, Any]) -> str | None:
    if not stack:
        return None
    parts = [
        stack.get("frontend"),
        stack.get("backend"),
        stack.get("database"),
    ]
    labels = [str(part).strip() for part in parts if isinstance(part, str) and part.strip()]
    return " · ".join(labels) if labels else None


class OpenClawOrchestrator:
    """OpenClaw-facing orchestration layer for complex project generation.

    LangGraph still executes nodes; this class owns phase definitions, project plan
    snapshots, agent manifests, and the user-visible build timeline so tools
    (GitHub, RAG, deploy) can plug in without rewriting the graph.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runtime = "openclaw" if openclaw_runtime_ready(settings) else "langgraph"

    @property
    def runtime(self) -> str:
        return self._runtime

    @property
    def registered_tools(self) -> list[str]:
        return registered_openclaw_tools() if self._runtime == "openclaw" else []

    def pipeline_phases(self) -> list[dict[str, str]]:
        return [dict(phase) for phase in BUILD_TIMELINE_PHASES]

    def initial_state_extras(self) -> dict[str, Any]:
        manifest = pipeline_manifest()
        return {
            "runtime": self._runtime,
            "registered_tools": self.registered_tools,
            "build_timeline": default_build_timeline(),
            "mvp_plan": {},
            "project_plan": {},
            "agent_logs": [],
            "project_agents": manifest["agents"],
            "openclaw_pipeline": {
                "runtime": self._runtime,
                "phases": manifest["stages"],
                "agents": manifest["agents"],
                "tools": self.registered_tools,
            },
        }

    def record_phase(
        self,
        state: dict[str, Any],
        *,
        phase_id: str,
        status: str,
        detail: str,
        artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        event = timeline_event(
            phase_id=phase_id,
            status=status,  # type: ignore[arg-type]
            detail=detail,
            artifacts=artifacts,
        )
        return {
            "build_timeline": apply_timeline_events(
                state.get("build_timeline"),
                [event],
            ),
        }

    def update_mvp_plan(self, state: dict[str, Any], **fields: Any) -> dict[str, Any]:
        current = dict(state.get("mvp_plan") or state.get("project_plan") or {})
        current.update({key: value for key, value in fields.items() if value is not None})
        return {"mvp_plan": current, "project_plan": current}

    def compose_project_plan(
        self,
        *,
        idea: str,
        intake: dict[str, Any] | None,
        mvp_scope: dict[str, Any] | None,
        repo_plan: dict[str, Any] | None,
        build_context: dict[str, Any] | None,
        recommended_stack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        intake = intake or {}
        scope = mvp_scope or {}
        plan = repo_plan or {}
        context = build_context or {}
        stack_rec = recommended_stack or context.get("recommendedStack") or {}
        features = list(scope.get("core_features") or scope.get("must_have") or [])
        advanced = list(scope.get("advanced_features") or [])
        required = intake.get("requiredFeatures") or intake.get("required_features") or []
        if isinstance(required, list):
            for item in required:
                label = str(item).strip()
                if label and label not in features:
                    features.append(label)

        return {
            "title": intake.get("title") or scope.get("title"),
            "idea": idea,
            "project_depth": scope.get("project_depth")
            or intake.get("projectDepth")
            or intake.get("project_depth")
            or "Advanced Project",
            "target_platform": scope.get("target_platform")
            or intake.get("targetPlatform")
            or intake.get("target_platform")
            or "web app",
            "vertical_pack": scope.get("vertical_pack") or scope.get("project_archetype"),
            "user_flows": scope.get("user_flows") or scope.get("demo_path") or [],
            "demo_path": scope.get("demo_path") or scope.get("user_flows") or [],
            "primary_entity": scope.get("primary_entity"),
            "api_routes": scope.get("api_routes") or [],
            "core_features": features,
            "advanced_features": advanced,
            "features": features + [item for item in advanced if item not in features],
            "target_users": intake.get("targetUsers") or intake.get("target_users"),
            "tech_stack_preference": (
                intake.get("techStackPreference")
                or intake.get("tech_stack_preference")
                or plan.get("selected_stack")
            ),
            "recommended_stack": stack_rec,
            "recommendedStack": stack_rec,
            "reference_url": intake.get("primaryRulesUrl") or intake.get("reference_url"),
            "project_boundary": scope.get("project_boundary") or scope.get("demo_boundary"),
            "demo_boundary": scope.get("demo_boundary") or scope.get("project_boundary"),
            "architecture_overview": plan.get("architecture_overview")
            or plan.get("architecture_notes")
            or plan.get("architectureNotes"),
            "architecture_notes": plan.get("architecture_notes") or plan.get("architectureNotes"),
            "data_model": plan.get("data_model") or [],
            "api_design": plan.get("api_design") or [],
            "database_schema": plan.get("database_schema") or [],
            "file_tree": plan.get("file_tree") or plan.get("files") or [],
            "implementation_steps": list(
                plan.get("implementation_steps") or plan.get("implementationSteps") or []
            ),
            "selected_stack": plan.get("selected_stack")
            or plan.get("selectedStack")
            or _stack_label(stack_rec),
            "files_planned": [
                _planned_file_path(item) for item in (plan.get("files") or plan.get("file_tree") or []) if item
            ],
            "deployment_plan": plan.get("deployment_plan") or [],
            "test_plan": plan.get("test_plan") or [],
            "rag_evidence_count": len(context.get("evidence") or []),
            "runtime": self._runtime,
            "orchestration_mode": self._runtime,
        }

    def compose_mvp_plan(
        self,
        *,
        idea: str,
        intake: dict[str, Any] | None,
        mvp_scope: dict[str, Any] | None,
        repo_plan: dict[str, Any] | None,
        build_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self.compose_project_plan(
            idea=idea,
            intake=intake,
            mvp_scope=mvp_scope,
            repo_plan=repo_plan,
            build_context=build_context,
        )
