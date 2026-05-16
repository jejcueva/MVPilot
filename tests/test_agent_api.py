import asyncio
from uuid import UUID, uuid4

from agent.config import Settings
from agent.main import create_app
from fastapi.testclient import TestClient


VALID_IDEA = (
    "Build a healthcare referral coordination agent that helps clinics "
    "prevent failed referrals."
)
FRONTEND_IDEA = "Build a study planner that turns messy class notes into a weekly review plan."


def start_task(client) -> str:
    response = client.post(
        "/agent/run",
        json={
            "idea": VALID_IDEA,
            "repo_visibility": "public",
            "demo_mode": True,
        },
    )
    assert response.status_code == 202
    return response.json()["task_id"]


def test_run_agent_accepts_valid_healthcare_referral_idea(client):
    response = client.post(
        "/agent/run",
        json={
            "idea": f"  {VALID_IDEA}  ",
            "repo_visibility": "public",
            "demo_mode": True,
        },
    )

    assert response.status_code == 202
    data = response.json()
    UUID(data["task_id"])
    assert data["status"] == "started"


def test_run_agent_accepts_frontend_form_payload_without_healthcare_mock(
    client,
    mock_live_rag_search,
):
    response = client.post(
        "/agent/run",
        data={
            "title": "Study planner",
            "idea": f"  {FRONTEND_IDEA}  ",
            "rule_source_type": "url",
            "rules_url": "https://example.com/hackathon-rules",
            "source": "mvpilot_frontend",
        },
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    detail_response = client.get(f"/agent/tasks/{task_id}")
    assert detail_response.status_code == 200
    data = detail_response.json()
    assert data["task"]["idea"] == FRONTEND_IDEA
    assert data["task"]["repo_visibility"] == "public"
    assert data["task"]["title"] == "Study planner"
    assert data["task"]["primary_rules_url"] == "https://example.com/hackathon-rules"
    assert data["task"]["source"] == "mvpilot_frontend"
    assert data["final_report"]["readme"]["content"].count(FRONTEND_IDEA) >= 1
    assert "healthcare" not in data["final_report"]["readme"]["content"].lower()


def test_run_agent_accepts_current_frontend_multipart_payload(
    client,
    mock_live_rag_search,
):
    response = client.post(
        "/agent/run",
        data={
            "title": "Clinic Navigator",
            "idea": FRONTEND_IDEA,
            "primary_rules_url": "https://example.com/rules",
            "additional_urls": [
                "https://example.com/judging",
                "https://developer.nvidia.com/nemotron",
            ],
            "github_connection_id": "conn_ready_123",
            "github_connected": "true",
            "source": "mvpilot_frontend",
        },
        files=[
            (
                "additional_files",
                ("notes.md", b"# Notes\nUse the hackathon rubric.", "text/markdown"),
            )
        ],
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    detail_response = client.get(f"/agent/tasks/{task_id}")
    assert detail_response.status_code == 200
    data = detail_response.json()
    task = data["task"]
    assert task["title"] == "Clinic Navigator"
    assert task["primary_rules_url"] == "https://example.com/rules"
    assert task["additional_urls"] == [
        "https://example.com/judging",
        "https://developer.nvidia.com/nemotron",
    ]
    assert task["additional_files"] == [
        {
            "name": "notes.md",
            "content_type": "text/markdown",
            "size_bytes": 33,
        }
    ]
    assert task["github_connected"] is True
    assert task["github_connection_id"] == "conn_ready_123"
    assert task["source"] == "mvpilot_frontend"
    assert "temporary-oauth-code" not in detail_response.text


def test_run_agent_accepts_frontend_json_payload(client, mock_live_rag_search):
    response = client.post(
        "/agent/run",
        json={
            "title": "Study Planner",
            "idea": f"  {FRONTEND_IDEA}  ",
            "primary_rules_url": " https://example.com/rules ",
            "additional_urls": [" ", "https://example.com/judging"],
            "additional_files": [
                {
                    "name": "notes.md",
                    "content_type": "text/markdown",
                    "size_bytes": 33,
                }
            ],
            "source": "mvpilot_frontend",
            "github_connected": True,
            "github_connection_id": " gh_conn_123 ",
            "repo_visibility": "public",
        },
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    detail_response = client.get(f"/agent/tasks/{task_id}")
    assert detail_response.status_code == 200
    task = detail_response.json()["task"]
    assert task["title"] == "Study Planner"
    assert task["idea"] == FRONTEND_IDEA
    assert task["primary_rules_url"] == "https://example.com/rules"
    assert task["additional_urls"] == ["https://example.com/judging"]
    assert task["additional_files"] == [
        {
            "name": "notes.md",
            "content_type": "text/markdown",
            "size_bytes": 33,
        }
    ]
    assert task["github_connected"] is True
    assert task["github_connection_id"] == "gh_conn_123"
    assert "uploaded_file_contents" not in detail_response.text
    assert "temporary-oauth-code" not in detail_response.text


def test_run_agent_defaults_demo_mode_to_false(client):
    response = client.post(
        "/agent/run",
        json={"idea": VALID_IDEA, "repo_visibility": "private"},
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    detail_response = client.get(f"/agent/tasks/{task_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["task"]["demo_mode"] is False


def test_run_agent_rejects_blank_ideas(client):
    response = client.post(
        "/agent/run",
        json={"idea": "   ", "repo_visibility": "public"},
    )

    assert response.status_code == 422


def test_run_agent_rejects_invalid_repo_visibility(client):
    response = client.post(
        "/agent/run",
        json={"idea": VALID_IDEA, "repo_visibility": "internal"},
    )

    assert response.status_code == 422


def test_task_detail_returns_populated_workflow_dashboard(client, mock_live_rag_search):
    task_id = start_task(client)

    response = client.get(f"/agent/tasks/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "task",
        "runtime",
        "registered_tools",
        "openclaw_trace",
        "agent_steps",
        "retrieved_docs",
        "build_context",
        "memory_matches",
        "tool_calls",
        "approvals",
            "generated_artifacts",
            "graph_trace",
            "mvp_plan",
            "build_timeline",
            "mvp_validation",
            "mvp_delivery",
            "final_report",
        }
    assert data["build_context"]["requiredDeliverables"]
    assert data["runtime"] == "langgraph"
    assert data["registered_tools"] == []
    assert data["openclaw_trace"] == []
    assert data["task"]["id"] == task_id
    assert data["task"]["idea"] == VALID_IDEA
    assert data["task"]["repo_visibility"] == "public"
    assert data["task"]["demo_mode"] is True
    assert data["task"]["status"] == "completed"
    assert data["retrieved_docs"]
    assert data["memory_matches"]
    assert [tool_call["tool"] for tool_call in data["tool_calls"]] == [
        "github.create_repo",
        "github.commit_files",
        "build.verify",
        "build.apply_recovery_patch",
        "build.verify",
    ]
    assert data["approvals"] == []
    assert {artifact["name"] for artifact in data["generated_artifacts"]} >= {
        "README.md",
        "package.json",
        "src/App.jsx",
        "backend/main.py",
        "backend/mvp_engine.py",
        "tests/test_backend.py",
        "docs/ARCHITECTURE.md",
        "docs/BUILD_LOG.md",
        "docs/DATABASE_SCHEMA.sql",
        "docs/IMPLEMENTATION_PLAN.md",
        "demo/demo_script.md",
        ".env.example",
        "final_report.json",
    }
    assert data["final_report"]["status"] == "completed"
    assert data["final_report"]["readme"]["content"]
    assert data["final_report"]["demo_script"]["content"]
    assert data["final_report"]["pitch"]["content"]
    assert data["final_report"]["blocker_analysis"]["blocker_type"]
    assert [step["node_name"] for step in data["graph_trace"]] == [
        "receive_idea",
        "exchange_github_code",
        "retrieve_context",
        "scope_mvp",
            "plan_repo",
            "create_repo",
            "generate_files",
            "validate_mvp",
            "commit_progress",
        "verify_build",
        "handle_blocker",
        "verify_build",
        "generate_final_package",
        "remember_outcome",
        "report_result",
    ]
    assert data["agent_steps"] == data["graph_trace"]
    assert all(step["model"] for step in data["agent_steps"])
    assert all(step["decision_trace"] for step in data["agent_steps"])
    model_steps = [step for step in data["agent_steps"] if step["prompt_purpose"]]
    assert {step["prompt_purpose"] for step in model_steps} == {
        "scope_mvp",
        "plan_repo",
        "file_manifest",
        "blocker_analysis",
        "final_package",
    }
    assert all(step["model_mode"] == "partial" for step in model_steps)
    assert "fake-nvidia-key" not in response.text


def test_task_detail_returns_404_for_missing_tasks(client):
    response = client.get(f"/agent/tasks/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


def test_task_detail_surfaces_openclaw_runtime_trace(mock_live_rag_search):
    app = create_app(
        settings=Settings(
            _env_file=None,
            adapter_mode="mock",
            openclaw_api_key="fake-openclaw-key",
            openclaw_env="development",
        )
    )

    with TestClient(app) as client:
        task_id = start_task(client)
        response = client.get(f"/agent/tasks/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["runtime"] == "openclaw"
    assert "github.create_repo" in data["registered_tools"]
    assert [entry["tool"] for entry in data["openclaw_trace"]] == [
        "github.create_repo",
        "github.commit_files",
        "build.verify",
        "build.apply_recovery_patch",
        "build.verify",
    ]
    assert all(tool_call["runtime"] == "openclaw" for tool_call in data["tool_calls"])


def test_approve_rejects_invalid_decisions(client):
    response = client.post(
        "/agent/approve",
        json={
            "task_id": str(uuid4()),
            "approval_id": str(uuid4()),
            "decision": "maybe",
            "approved_by": "demo_judge",
        },
    )

    assert response.status_code == 422


def test_approve_returns_404_for_unknown_approvals(client):
    task_id = start_task(client)

    response = client.post(
        "/agent/approve",
        json={
            "task_id": task_id,
            "approval_id": str(uuid4()),
            "decision": "approved",
            "approved_by": "demo_judge",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Approval not found"


def test_approve_updates_seeded_pending_approval(app, client):
    task_id = start_task(client)
    approval = asyncio.run(
        app.state.task_store.seed_pending_approval(
            task_id=task_id,
            proposed_action="Post a referral status update",
            risk_level="medium",
        )
    )

    response = client.post(
        "/agent/approve",
        json={
            "task_id": task_id,
            "approval_id": approval.approval_id,
            "decision": "approved",
            "approved_by": "demo_judge",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "task_id": task_id,
        "approval_id": approval.approval_id,
        "status": "approved",
        "approved_by": "demo_judge",
    }

    detail_response = client.get(f"/agent/tasks/{task_id}")
    stored_approval = detail_response.json()["approvals"][0]
    assert stored_approval["status"] == "approved"
    assert stored_approval["approved_by"] == "demo_judge"
    assert stored_approval["resolved_at"] is not None
