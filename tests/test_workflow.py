import pytest

from agent.config import Settings
from agent.model_client import DeterministicModelClient
from agent.live_adapters import LiveRagMemoryAdapter
from agent.schemas import RunAgentRequest, TaskStatus
from agent.service import AgentService
from agent.task_store import InMemoryTaskStore
from agent.adapters import InMemoryToolAdapter
from agent.github_oauth import GitHubWorkflowAuth
from agent.stack_recommendation import stack_items_from_recommended
from agent.model_client import ModelCallResult
from agent.workflow import (
    _require_live_nemotron_result,
    build_initial_state,
    build_workflow,
    route_after_tool_result,
)
from tools.github_tool import GitHubConfig


VALID_IDEA = (
    "Build a healthcare referral coordination agent that helps clinics "
    "prevent failed referrals."
)
FRONTEND_IDEA = "Build a study planner that turns messy class notes into a weekly review plan."


class CapturingModelClient(DeterministicModelClient):
    def __init__(self) -> None:
        super().__init__(mode="mock")
        self.prompts: dict[str, str] = {}

    async def complete_structured(self, **kwargs):
        self.prompts[kwargs["purpose"]] = kwargs["prompt"]
        return await super().complete_structured(**kwargs)


class EmptyRagAdapter:
    async def retrieve_build_context(
        self,
        project_id: str,
        idea: str,
        *,
        optional_params: dict | None = None,
        rules_url: str | None = None,
        reference_urls: list[str] | None = None,
        context_needed: list[str] | None = None,
        top_k: int = 8,
    ) -> dict:
        del (
            project_id,
            idea,
            optional_params,
            rules_url,
            reference_urls,
            context_needed,
            top_k,
        )
        return {
            "mode": "mock",
            "requiredDeliverables": [],
            "allowedToolsAndAPIs": [],
            "requiredRepositoryFormat": [],
            "requiredDemoFormat": [],
            "requiredTechStackPieces": [],
            "agentBoundaries": [],
            "resolvedTechStack": {
                "source": "default",
                "requiredItems": [],
                "defaultItems": [],
                "items": [],
            },
            "scopeWarnings": [],
            "evidence": [],
        }

    async def retrieve_hackathon_context(self, idea: str) -> list[dict]:
        return []

    async def retrieve_nvidia_context(self, idea: str) -> list[dict]:
        return []

    async def find_similar_builds(self, issue: str) -> list[dict]:
        return []

    async def write_memory(self, memory: dict) -> None:
        return None


class FakeGitHubConnections:
    def __init__(self) -> None:
        self.exchanged: list[tuple[str, str]] = []

    async def exchange_for_workflow(
        self,
        connection_id: str,
        *,
        task_id: str | None,
    ) -> GitHubWorkflowAuth:
        assert task_id is not None
        self.exchanged.append((connection_id, task_id))
        return GitHubWorkflowAuth(
            connection_id=connection_id,
            login="octocat",
            scopes=["repo", "read:user", "user:email"],
            config=GitHubConfig(
                token="gho-task-token",
                owner="octocat",
                repo_prefix="mvpilot-generated-",
                mock_tools=False,
            ),
        )


class RecordingToolAdapter(InMemoryToolAdapter):
    def __init__(self) -> None:
        self.github_config: GitHubConfig | None = None
        self.create_repo_owner: str | None = None

    def set_github_config(self, config: GitHubConfig) -> None:
        self.github_config = config

    def create_repo(
        self,
        task_id: str,
        visibility: str,
        *,
        repo_preference: str = "create_new_repo",
        repo_name: str | None = None,
        repo_description: str | None = None,
        repo_url: str | None = None,
    ) -> dict:
        self.create_repo_owner = self.github_config.owner if self.github_config else None
        return super().create_repo(
            task_id,
            visibility,
            repo_preference=repo_preference,
            repo_name=repo_name,
            repo_description=repo_description,
            repo_url=repo_url,
        )


@pytest.mark.asyncio
async def test_full_workflow_completes_with_expected_timeline(mock_live_rag_search):
    settings = Settings(_env_file=None, adapter_mode="mock")
    task_store = InMemoryTaskStore()
    service = AgentService(task_store, settings)
    response = await service.start_task(
        RunAgentRequest(
            idea=VALID_IDEA,
            repo_visibility="public",
            demo_mode=True,
        )
    )

    await service.run_task_workflow(response.task_id)

    detail = await task_store.get_task(response.task_id)
    node_names = [step.node_name for step in detail.graph_trace]

    assert detail.task.status == TaskStatus.COMPLETED
    assert node_names == [
        "receive_idea",
        "exchange_github_code",
        "retrieve_context",
        "scope_mvp",
        "recommend_stack",
        "plan_repo",
        "create_repo",
        "generate_files",
        "validate_mvp",
        "debug_generated_files",
        "commit_progress",
        "verify_build",
        "handle_blocker",
        "verify_build",
        "generate_final_package",
        "remember_outcome",
        "report_result",
    ]
    assert detail.retrieved_docs
    assert detail.build_context
    assert detail.build_context.get("requiredDeliverables")
    assert detail.build_context.get("allowedToolsAndAPIs")
    assert detail.build_context.get("requiredRepositoryFormat")
    assert detail.build_context.get("requiredDemoFormat")
    assert detail.build_context.get("requiredTechStackPieces")
    assert detail.build_context.get("mode") == "live"
    assert detail.build_context.get("evidence")
    assert detail.memory_matches
    assert detail.final_report is not None
    assert detail.final_report["mode"] == "mock"
    assert {artifact["name"] for artifact in detail.generated_artifacts} >= {
        "README.md",
        "package.json",
        "src/App.jsx",
        "backend/main.py",
        "backend/db.py",
        "tests/test_backend.py",
        "docs/ARCHITECTURE.md",
        "docs/BUILD_LOG.md",
        "docs/DATABASE_SCHEMA.sql",
        "docs/WALKTHROUGH.md",
        "docs/DEPLOY.md",
        "docs/PROJECT_PLAN.md",
        "final_report.json",
    }
    assert detail.mvp_plan.get("demo_path")
    assert detail.mvp_plan.get("vertical_pack")
    model_backed_steps = {
        step.node_name: step
        for step in detail.agent_steps
        if step.prompt_purpose is not None
    }
    assert set(model_backed_steps) == {
        "scope_mvp",
        "recommend_stack",
        "plan_repo",
        "generate_files",
        "handle_blocker",
        "generate_final_package",
    }
    assert model_backed_steps["scope_mvp"].model == settings.nemotron_model
    assert model_backed_steps["plan_repo"].model == settings.nemotron_model
    assert model_backed_steps["generate_files"].model == settings.nemotron_model
    assert all(step.model_mode == "mock" for step in model_backed_steps.values())
    assert all(step.decision_trace for step in detail.agent_steps)
    assert detail.final_report["readme"]["content"]
    assert detail.final_report["demo_script"]["content"]
    assert detail.final_report["pitch"]["content"]
    assert detail.final_report["blocker_analysis"]["recoverable"] is True


def test_route_after_tool_result_continues_after_success():
    route = route_after_tool_result(
        {"last_tool_result": {"status": "success", "recoverable": False}}
    )

    assert route == "generate_final_package"


def test_route_after_tool_result_handles_recoverable_blocker():
    route = route_after_tool_result(
        {"last_tool_result": {"status": "failed", "recoverable": True}}
    )

    assert route == "handle_blocker"


def test_route_after_tool_result_fails_unrecoverable_result():
    route = route_after_tool_result(
        {"last_tool_result": {"status": "failed", "recoverable": False}}
    )

    assert route == "failed"


@pytest.mark.asyncio
async def test_workflow_plan_repo_prompt_uses_recommended_stack_after_selector(monkeypatch):
    async def fake_search(query: str, top_k: int = 5, doc_types=None):
        return []

    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)
    monkeypatch.setattr("agent.live_adapters.search_rag", fake_search)
    from unittest.mock import AsyncMock, MagicMock

    mock_store = MagicMock()
    mock_store.search_memories = AsyncMock(return_value=[])
    mock_store.write_memory = AsyncMock()
    monkeypatch.setattr("agent.rag.store.get_rag_store", lambda: mock_store)
    monkeypatch.setattr("agent.rag.embed.embed_text", AsyncMock(return_value=[0.1, 0.2]))

    settings = Settings(_env_file=None, adapter_mode="mock")
    model_client = CapturingModelClient()
    workflow = build_workflow(
        settings,
        model_client=model_client,
        retrieval=LiveRagMemoryAdapter(),
        tools=InMemoryToolAdapter(),
    )
    initial_state = build_initial_state(
        task_id="task-default-stack",
        idea=VALID_IDEA,
        repo_visibility="public",
        demo_mode=True,
        settings=settings,
    )

    final_state = await workflow.ainvoke(initial_state)
    plan_prompt = model_client.prompts["plan_repo"]

    assert final_state["build_context"]["requiredTechStackPieces"]
    assert final_state["recommended_stack"]
    assert final_state["build_context"]["resolvedTechStack"]["source"] == "stack_recommendation"
    assert "Recommended stack (binding)" in plan_prompt or "recommendedStack" in plan_prompt
    assert "recommend_stack" in model_client.prompts
    assert "Do not assume the generated project must use MVPilot" in model_client.prompts["recommend_stack"]
    assert final_state["recommended_stack"]
    assert final_state["repo_plan"]["selected_stack"] == stack_items_from_recommended(
        final_state["recommended_stack"]
    )


@pytest.mark.asyncio
async def test_workflow_uses_frontend_intake_and_surfaces_source_warnings(mock_live_rag_search):
    settings = Settings(_env_file=None, adapter_mode="mock")
    task_store = InMemoryTaskStore()
    service = AgentService(task_store, settings)
    response = await service.start_task(
        RunAgentRequest(
            title="Study Planner",
            idea=FRONTEND_IDEA,
            repo_visibility="public",
            demo_mode=True,
            source="mvpilot_frontend",
            primary_rules_url="http://127.0.0.1:9/missing-rules",
        )
    )

    await service.run_task_workflow(response.task_id)

    detail = await task_store.get_task(response.task_id)
    assert detail.task.status == TaskStatus.COMPLETED
    assert detail.build_context["frontendIntake"]["title"] == "Study Planner"
    assert detail.build_context["frontendIntake"]["idea"] == FRONTEND_IDEA
    assert detail.build_context["sourceContext"]["warnings"]
    assert detail.final_report["readme"]["title"] == "Study Planner"
    assert "Study Planner" in detail.final_report["readme"]["content"]
    assert FRONTEND_IDEA in detail.final_report["readme"]["content"]
    assert "Study Planner" in detail.final_report["demo_script"]["title"]
    assert detail.final_report["pitch"]["title"] == "Study Planner"
    assert detail.final_report["source_warnings"] == detail.build_context["sourceContext"]["warnings"]
    assert "healthcare" not in str(detail.final_report).lower()


@pytest.mark.asyncio
async def test_live_workflow_missing_github_connection_retrieves_context_then_fails_on_repo_creation():
    from agent.live_adapters import LiveToolAdapter

    settings = Settings(_env_file=None, adapter_mode="live")
    workflow = build_workflow(
        settings,
        model_client=DeterministicModelClient(mode="mock"),
        retrieval=EmptyRagAdapter(),
        tools=LiveToolAdapter(),
    )
    initial_state = build_initial_state(
        task_id="task-live-missing-github",
        idea=VALID_IDEA,
        repo_visibility="public",
        demo_mode=False,
        settings=settings,
    )

    final_state = await workflow.ainvoke(initial_state)

    assert final_state["status"] == "failed"
    assert [step.node_name for step in final_state["graph_trace"]] == [
        "receive_idea",
        "exchange_github_code",
        "failed",
    ]
    assert "GitHub is not connected" in final_state["failure_reason"]
    assert not [
        tool_call
        for tool_call in final_state["tool_calls"]
        if tool_call.get("tool") == "github.create_repo"
    ]


@pytest.mark.asyncio
async def test_live_workflow_exchanges_github_connection_before_create_repo(mock_live_rag_search):
    settings = Settings(_env_file=None, adapter_mode="live")
    github_connections = FakeGitHubConnections()
    tools = RecordingToolAdapter()
    workflow = build_workflow(
        settings,
        model_client=DeterministicModelClient(mode="mock"),
        retrieval=EmptyRagAdapter(),
        tools=tools,
        github_connections=github_connections,
    )
    initial_state = build_initial_state(
        task_id="task-live-ready",
        idea=VALID_IDEA,
        repo_visibility="public",
        demo_mode=False,
        settings=settings,
        frontend_intake={
            "idea": VALID_IDEA,
            "githubConnected": True,
            "githubConnectionId": "conn-ready",
        },
    )

    final_state = await workflow.ainvoke(initial_state)

    assert github_connections.exchanged == [("conn-ready", "task-live-ready")]
    assert tools.create_repo_owner == "octocat"
    assert final_state["github_connection"]["login"] == "octocat"
    assert [step.node_name for step in final_state["graph_trace"]][:3] == [
        "receive_idea",
        "exchange_github_code",
        "retrieve_context",
    ]


@pytest.mark.asyncio
async def test_unrecoverable_tool_result_marks_workflow_failed(mock_live_rag_search):
    class UnrecoverableToolAdapter(InMemoryToolAdapter):
        def verify_build(self, *, recovered: bool, repo_name: str | None = None) -> dict:
            return {
                "tool": "build.verify",
                "status": "failed",
                "mock_mode": True,
                "recoverable": False,
                "error": "Mock mode: unrecoverable build failure.",
                "summary": "Mock mode: build verification cannot recover.",
            }

    settings = Settings(_env_file=None, adapter_mode="mock")
    workflow = build_workflow(settings, tools=UnrecoverableToolAdapter())
    initial_state = build_initial_state(
        task_id="task-1",
        idea=VALID_IDEA,
        repo_visibility="public",
        demo_mode=True,
        settings=settings,
    )

    final_state = await workflow.ainvoke(initial_state)

    assert final_state["status"] == "failed"
    assert final_state["graph_trace"][-1].node_name == "failed"
    assert final_state["final_report"]["status"] == "failed"


@pytest.mark.asyncio
async def test_workflow_memory_integration(mock_live_rag_search):
    # This proves remember_outcome calls write_memory() and find_similar_builds returns a prior memory
    settings = Settings(_env_file=None, adapter_mode="mock")
    task_store = InMemoryTaskStore()
    service = AgentService(task_store, settings)

    from unittest.mock import patch, MagicMock, AsyncMock
    with patch("agent.rag.store.get_rag_store") as mock_get_store, \
         patch("agent.rag.embed.embed_text", new_callable=AsyncMock) as mock_embed:
        
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.search_memories = AsyncMock(return_value=[{"id": "mock_memory"}])
        mock_store.write_memory = AsyncMock()
        mock_embed.return_value = [0.1, 0.2]

        # First run
        response = await service.start_task(
            RunAgentRequest(
                idea="Memory test idea",
                repo_visibility="public",
                demo_mode=True,
            )
        )
        await service.run_task_workflow(response.task_id)
        detail1 = await task_store.get_task(response.task_id)
    
        # Ensure memory_matches from first run contains the mock memory we injected
        assert len(detail1.memory_matches) > 0
        assert detail1.memory_matches[0]["id"] == "mock_memory"
    
        # Verify write_memory was called
        mock_store.write_memory.assert_awaited_once()
        written_memory = mock_store.write_memory.call_args[0][0]
        assert written_memory["outcome"]["workflow_task_id"] == response.task_id
        assert written_memory["idea"] == "Memory test idea"
        assert "embedding" in written_memory
    
        # Second run
        response2 = await service.start_task(
            RunAgentRequest(
                idea="Memory test idea 2",
                repo_visibility="public",
                demo_mode=True,
            )
        )
        await service.run_task_workflow(response2.task_id)
        detail2 = await task_store.get_task(response2.task_id)
        
        assert detail2.task.status == TaskStatus.COMPLETED
        assert len(detail2.memory_matches) > 0
        assert detail2.memory_matches[0]["id"] == "mock_memory"
        assert mock_store.write_memory.call_count == 2


def test_require_live_allows_degraded_plan_repo_when_partial_enabled() -> None:
    settings = Settings(allow_idea_aware_partial=True, require_live_file_manifest=True)
    result = ModelCallResult(
        output=object(),
        mode="degraded",
        model="nvidia/test",
        purpose="plan_repo",
        latency_ms=0,
        fallback_reason="HTTP 504 from Nemotron.",
    )
    _require_live_nemotron_result(
        result, "plan_repo", enforced=True, settings=settings
    )


def test_require_live_blocks_degraded_file_manifest_when_required() -> None:
    settings = Settings(allow_idea_aware_partial=True, require_live_file_manifest=True)
    result = ModelCallResult(
        output=object(),
        mode="degraded",
        model="nvidia/test",
        purpose="file_manifest",
        latency_ms=0,
        fallback_reason="HTTP 504 from Nemotron.",
    )
    with pytest.raises(RuntimeError, match="file_manifest"):
        _require_live_nemotron_result(
            result, "file_manifest", enforced=True, settings=settings
        )
