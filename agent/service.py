from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import BackgroundTasks

from agent.config import Settings
from agent.schemas import (
    AgentStep,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    RunAgentRequest,
    RunAgentResponse,
    TaskStatus,
    TaskDetailResponse,
)
from agent.project_session_store import SupabasePersistingTaskStore
from agent.task_store import InMemoryTaskStore
from agent.frontend_intake import build_frontend_intake_from_task
from agent.github_oauth import GitHubConnectionService
from agent.workflow import build_initial_state, build_workflow


class AgentService:
    def __init__(
        self,
        task_store: InMemoryTaskStore | SupabasePersistingTaskStore,
        settings: Settings,
        *,
        github_connections: GitHubConnectionService | None = None,
    ) -> None:
        self._task_store = task_store
        self._settings = settings
        self._github_connections = github_connections

    async def start_task(
        self,
        request: RunAgentRequest,
        background_tasks: BackgroundTasks | None = None,
    ) -> RunAgentResponse:
        task = await self._task_store.create_task(request)
        if background_tasks is not None:
            background_tasks.add_task(self.run_task_workflow, task.id)
        return RunAgentResponse(task_id=task.id, status="started")

    async def run_task_workflow(self, task_id: str) -> None:
        detail = await self._task_store.get_task(task_id)
        uploaded_files = await self._task_store.get_uploaded_file_contents(task_id)
        frontend_intake = build_frontend_intake_from_task(detail.task).model_dump()
        state = build_initial_state(
            task_id=task_id,
            idea=detail.task.idea,
            repo_visibility=detail.task.repo_visibility,
            demo_mode=detail.task.demo_mode,
            source_urls=detail.task.source_urls,
            settings=self._settings,
            frontend_intake=frontend_intake,
            uploaded_file_contents=[
                {
                    "name": file.name,
                    "content_type": file.content_type,
                    "size_bytes": file.size_bytes,
                    "content": file.content,
                }
                for file in uploaded_files
            ],
        )
        
        from agent.live_adapters import LiveRagMemoryAdapter

        rag = LiveRagMemoryAdapter()
        if self._settings.mock_mode:
            from agent.adapters import InMemoryAuditAdapter, InMemoryToolAdapter

            tools = InMemoryToolAdapter()
            audit = InMemoryAuditAdapter(model_name=self._settings.nemotron_fast_model)
        else:
            from agent.live_adapters import LiveAuditAdapter, LiveToolAdapter

            tools = LiveToolAdapter()
            audit = LiveAuditAdapter(model_name=self._settings.nemotron_fast_model)

        if self._settings.openclaw_configured:
            from agent.openclaw_runtime import OpenClawToolAdapter

            tools = OpenClawToolAdapter(
                tools,
                environment=self._settings.openclaw_env,
            )

        workflow = build_workflow(
            self._settings,
            tools=tools,
            retrieval=rag,
            audit=audit,
            github_connections=self._github_connections,
        )

        try:
            async for snapshot in workflow.astream(state, stream_mode="values"):
                await self._task_store.snapshot_task(task_id, snapshot)
        except Exception as exc:
            current_detail = await self._task_store.get_task(task_id)
            failed_state = self._build_unexpected_failure_state(current_detail, exc)
            await self._task_store.snapshot_task(task_id, failed_state)

    async def get_task_detail(self, task_id: str) -> TaskDetailResponse:
        return await self._task_store.get_task(task_id)

    async def approve_action(
        self,
        request: ApprovalDecisionRequest,
    ) -> ApprovalDecisionResponse:
        approval = await self._task_store.resolve_approval(request)
        return ApprovalDecisionResponse(
            task_id=approval.task_id,
            approval_id=approval.approval_id,
            status=approval.status,
            approved_by=approval.approved_by or request.approved_by,
        )

    def _build_unexpected_failure_state(
        self,
        detail: TaskDetailResponse,
        exc: Exception,
    ) -> dict[str, Any]:
        detail_message = str(exc).strip() or exc.__class__.__name__
        message = f"Workflow failed before the graph could finish: {detail_message}"
        step = AgentStep(
            project_id=detail.task.id,
            flight_stage="failed",
            agent="orchestrator",
            node_name="failed",
            status="failed",
            message=message,
            model=self._settings.nemotron_fast_model,
            decision_trace=[
                "Unhandled exception stopped the LangGraph workflow.",
                f"{exc.__class__.__name__}: {detail_message}",
            ],
            timestamp=datetime.now(UTC),
        )
        return {
            "status": TaskStatus.FAILED,
            "agent_steps": [*detail.agent_steps, step],
            "retrieved_docs": detail.retrieved_docs,
            "build_context": detail.build_context or {},
            "memory_matches": detail.memory_matches,
            "tool_calls": detail.tool_calls,
            "runtime": detail.runtime,
            "registered_tools": detail.registered_tools,
            "openclaw_trace": detail.openclaw_trace,
            "generated_artifacts": detail.generated_artifacts,
            "graph_trace": [*detail.graph_trace, step],
            "mvp_plan": detail.mvp_plan,
            "build_timeline": detail.build_timeline,
            "final_report": {
                "status": "failed",
                "mode": "mock" if self._settings.mock_mode else "live",
                "model": self._settings.nemotron_fast_model,
                "summary": message,
            },
        }
