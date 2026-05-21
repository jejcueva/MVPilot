from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.config import Settings
from agent.project_session_store import SupabasePersistingTaskStore
from agent.schemas import RunAgentRequest, TaskStatus


@pytest.mark.asyncio
async def test_persisting_store_writes_session_on_snapshot(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        adapter_mode="live",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="test-key",
    )
    store = SupabasePersistingTaskStore(settings)
    mock_client = MagicMock()
    table = MagicMock()
    mock_client.table.return_value = table
    table.upsert.return_value.execute.return_value = MagicMock(error=None)
    table.delete.return_value.execute.return_value = MagicMock(error=None)
    table.insert.return_value.execute.return_value = MagicMock(error=None)
    store._client = mock_client

    request = RunAgentRequest(
        idea="StudyPilot helps students learn.",
        repo_visibility="public",
        project_depth="Hackathon-Winning Project",
        target_platform="web app",
    )
    task = await store.create_task(request)
    recommended = {"frontend": "Next.js", "backend": "FastAPI", "aiModels": ["Nemotron"]}

    await store.snapshot_task(
        task.id,
        {
            "status": TaskStatus.STARTED,
            "mvp_scope": {"core_features": ["quizzes"]},
            "recommended_stack": recommended,
            "mvp_plan": {"recommended_stack": recommended},
            "build_timeline": [],
            "agent_logs": [
                {
                    "agent_key": "stack_selector",
                    "agent_name": "Stack Selector Agent",
                    "stage_id": "tech_stack_recommendation",
                    "status": "completed",
                    "message": "Recommended stack",
                    "detail": "Recommended stack",
                    "timestamp": "2026-05-17T00:00:00Z",
                }
            ],
        },
    )

    assert mock_client.table.called
    session_rows = [call[0][0] for call in table.upsert.call_args_list]
    snapshot_row = next(row for row in session_rows if row.get("recommended_stack"))
    assert snapshot_row["task_id"] == task.id
    assert snapshot_row["recommended_stack"] == recommended
