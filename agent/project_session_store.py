from __future__ import annotations

import logging
from typing import Any, Mapping

from agent.config import Settings
from agent.schemas import (
    ApprovalDecisionRequest,
    ApprovalRecord,
    RunAgentRequest,
    TaskDetailResponse,
    TaskRecord,
    UploadedSourceFileContent,
)
from agent.task_store import ApprovalNotFoundError, InMemoryTaskStore, TaskNotFoundError

logger = logging.getLogger(__name__)


def _supabase_client(settings: Settings):
    from supabase import create_client

    secret = settings.supabase_service_role_key
    return create_client(
        settings.supabase_url or "",
        secret.get_secret_value() if secret else "",
    )


def _raise_for_error(response: object, action: str) -> None:
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"Supabase failed to {action}: {error}")


class SupabasePersistingTaskStore:
    """In-memory task API with optional Supabase persistence for project generation."""

    def __init__(
        self,
        settings: Settings,
        memory: InMemoryTaskStore | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory or InMemoryTaskStore()
        self._client = None
        self._persisted_log_count: dict[str, int] = {}
        if settings.supabase_configured:
            try:
                self._client = _supabase_client(settings)
            except Exception as exc:
                logger.warning("Supabase task persistence disabled: %s", exc)

    @property
    def persistence_enabled(self) -> bool:
        return self._client is not None

    async def create_task(self, request: RunAgentRequest) -> TaskRecord:
        task = await self._memory.create_task(request)
        if self._client is None:
            return task
        try:
            intake = request.model_dump()
            self._client.table("project_sessions").upsert(
                {
                    "task_id": task.id,
                    "idea": task.idea,
                    "project_depth": request.project_depth,
                    "target_platform": request.target_platform,
                    "orchestration_mode": "openclaw"
                    if request.use_openclaw_orchestration
                    else "langgraph",
                    "status": task.status,
                    "project_plan": {},
                    "recommended_stack": {},
                    "build_timeline": [],
                },
                on_conflict="task_id",
            ).execute()
        except Exception as exc:
            logger.warning("Failed to persist project session %s: %s", task.id, exc)
        return task

    async def get_task(self, task_id: str) -> TaskDetailResponse:
        return await self._memory.get_task(task_id)

    async def get_uploaded_file_contents(
        self,
        task_id: str,
    ) -> list[UploadedSourceFileContent]:
        return await self._memory.get_uploaded_file_contents(task_id)

    async def snapshot_task(
        self,
        task_id: str,
        state: Mapping[str, Any],
    ) -> TaskDetailResponse:
        detail = await self._memory.snapshot_task(task_id, state)
        if self._client is None:
            return detail
        try:
            self._persist_snapshot(task_id, state, detail)
        except Exception as exc:
            logger.warning("Failed to persist snapshot for %s: %s", task_id, exc)
        return detail

    def _persist_snapshot(
        self,
        task_id: str,
        state: Mapping[str, Any],
        detail: TaskDetailResponse,
    ) -> None:
        recommended = state.get("recommended_stack")
        if not isinstance(recommended, dict):
            recommended = (detail.mvp_plan or {}).get("recommended_stack") or (
                detail.mvp_plan or {}
            ).get("recommendedStack")
        if not isinstance(recommended, dict):
            recommended = {}

        project_plan = state.get("project_plan") or state.get("mvp_plan") or detail.mvp_plan or {}
        if not isinstance(project_plan, dict):
            project_plan = {}

        session_row = {
            "task_id": task_id,
            "idea": detail.task.idea,
            "status": str(state.get("status", detail.task.status)),
            "project_plan": project_plan,
            "recommended_stack": recommended if isinstance(recommended, dict) else {},
            "build_timeline": list(
                state.get("build_timeline", detail.build_timeline) or []
            ),
            "updated_at": detail.task.updated_at.isoformat(),
        }
        response = (
            self._client.table("project_sessions")
            .upsert(session_row, on_conflict="task_id")
            .execute()
        )
        _raise_for_error(response, "upsert project session")

        requirements = state.get("mvp_scope")
        if isinstance(requirements, dict) and requirements:
            self._upsert_json_row("project_requirements", task_id, "requirements", requirements)

        repo_plan = state.get("repo_plan")
        if isinstance(repo_plan, dict) and repo_plan:
            self._upsert_json_row("project_architectures", task_id, "architecture", repo_plan)

        validation = state.get("mvp_validation")
        if isinstance(validation, dict) and validation.get("checks"):
            self._client.table("validation_results").delete().eq("task_id", task_id).execute()
            response = (
                self._client.table("validation_results")
                .insert(
                    {
                        "task_id": task_id,
                        "passed": bool(validation.get("passed")),
                        "report": validation,
                    }
                )
                .execute()
            )
            _raise_for_error(response, "insert validation_results")

        self._persist_agent_logs(task_id, state)

    def _upsert_json_row(
        self,
        table: str,
        task_id: str,
        column: str,
        payload: dict[str, Any],
    ) -> None:
        self._client.table(table).delete().eq("task_id", task_id).execute()
        response = (
            self._client.table(table)
            .insert({"task_id": task_id, column: payload})
            .execute()
        )
        _raise_for_error(response, f"insert {table}")

    def _persist_agent_logs(self, task_id: str, state: Mapping[str, Any]) -> None:
        logs = state.get("agent_logs")
        if not isinstance(logs, list):
            return
        previous = self._persisted_log_count.get(task_id, 0)
        if len(logs) <= previous:
            return
        new_entries = logs[previous:]
        rows = [
            {
                "task_id": task_id,
                "agent_key": str(entry.get("agent_key") or "logger"),
                "agent_name": str(entry.get("agent_name") or "Logger Agent"),
                "stage_id": str(entry.get("stage_id") or "unknown"),
                "status": str(entry.get("status") or "completed"),
                "message": str(entry.get("message") or "")[:2000],
                "detail": str(entry.get("detail") or entry.get("message") or "")[:4000],
                "logged_at": entry.get("timestamp"),
            }
            for entry in new_entries
            if isinstance(entry, dict)
        ]
        if not rows:
            return
        response = self._client.table("agent_logs").insert(rows).execute()
        _raise_for_error(response, "insert agent logs")
        self._persisted_log_count[task_id] = len(logs)

    async def seed_pending_approval(
        self,
        *,
        task_id: str,
        proposed_action: str,
        risk_level: str,
    ) -> ApprovalRecord:
        return await self._memory.seed_pending_approval(
            task_id=task_id,
            proposed_action=proposed_action,
            risk_level=risk_level,
        )

    async def resolve_approval(
        self,
        request: ApprovalDecisionRequest,
    ) -> ApprovalRecord:
        return await self._memory.resolve_approval(request)


def build_task_store(settings: Settings) -> InMemoryTaskStore | SupabasePersistingTaskStore:
    if settings.adapter_mode == "live" and settings.supabase_configured:
        return SupabasePersistingTaskStore(settings)
    return InMemoryTaskStore()
