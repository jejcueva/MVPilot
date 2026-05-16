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


def _planned_file_path(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("path") or item.get("name") or item)
    return str(item)


class OpenClawOrchestrator:
    """OpenClaw-facing orchestration layer for the MVP build pipeline.

    LangGraph still executes nodes; this class owns phase definitions, MVP plan
    snapshots, and the user-visible build timeline so tools (GitHub, RAG, deploy)
    can plug in without rewriting the graph.
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
        return {
            "runtime": self._runtime,
            "registered_tools": self.registered_tools,
            "build_timeline": default_build_timeline(),
            "mvp_plan": {},
            "openclaw_pipeline": {
                "runtime": self._runtime,
                "phases": self.pipeline_phases(),
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
        current = dict(state.get("mvp_plan") or {})
        current.update({key: value for key, value in fields.items() if value is not None})
        return {"mvp_plan": current}

    def compose_mvp_plan(
        self,
        *,
        idea: str,
        intake: dict[str, Any] | None,
        mvp_scope: dict[str, Any] | None,
        repo_plan: dict[str, Any] | None,
        build_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        intake = intake or {}
        scope = mvp_scope or {}
        plan = repo_plan or {}
        context = build_context or {}
        features = list(scope.get("must_have") or [])
        required = intake.get("requiredFeatures") or intake.get("required_features") or []
        if isinstance(required, list):
            for item in required:
                label = str(item).strip()
                if label and label not in features:
                    features.append(label)

        return {
            "title": intake.get("title") or scope.get("title"),
            "idea": idea,
            "target_users": intake.get("targetUsers") or intake.get("target_users"),
            "tech_stack_preference": (
                intake.get("techStackPreference")
                or intake.get("tech_stack_preference")
                or plan.get("selected_stack")
            ),
            "reference_url": intake.get("primaryRulesUrl") or intake.get("reference_url"),
            "features": features,
            "demo_boundary": scope.get("demo_boundary"),
            "architecture_notes": plan.get("architecture_notes") or plan.get("architectureNotes"),
            "implementation_steps": list(
                plan.get("implementation_steps") or plan.get("implementationSteps") or []
            ),
            "selected_stack": plan.get("selected_stack") or plan.get("selectedStack"),
            "files_planned": [
                _planned_file_path(item) for item in (plan.get("files") or []) if item
            ],
            "rag_evidence_count": len(context.get("evidence") or []),
            "runtime": self._runtime,
        }
