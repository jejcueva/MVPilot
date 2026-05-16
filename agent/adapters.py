from typing import Any, Protocol
from datetime import UTC, datetime

from agent.schemas import AgentStep

class RagMemoryAdapter(Protocol):
    async def index_source_urls(self, urls: list[str]) -> dict[str, Any]: ...
    async def retrieve_hackathon_context(self, idea: str) -> list[dict[str, Any]]: ...
    async def retrieve_nvidia_context(self, idea: str) -> list[dict[str, Any]]: ...
    async def find_similar_builds(self, issue: str) -> list[dict[str, Any]]: ...
    async def retrieve_build_context(
        self,
        project_id: str,
        idea: str,
        *,
        optional_params: dict[str, Any] | None = None,
        rules_url: str | None = None,
        reference_urls: list[str] | None = None,
        context_needed: list[str] | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]: ...
    async def write_memory(self, memory: dict[str, Any]) -> None: ...

class ToolAdapter(Protocol):
    def create_repo(
        self,
        task_id: str,
        visibility: str,
        *,
        repo_preference: str = "create_new_repo",
        repo_name: str | None = None,
        repo_description: str | None = None,
        repo_url: str | None = None,
    ) -> dict[str, Any]: ...
    def commit_files(self, repo_name: str, files: list[dict[str, Any]], message: str) -> dict[str, Any]: ...
    def check_repo_health(self, repo_name: str) -> dict[str, Any]: ...
    def detect_blocker(self, logs: list[dict[str, Any]]) -> dict[str, Any]: ...
    def verify_commit(self, repo_name: str, commit_sha: str) -> dict[str, Any]: ...
    def verify_build(self, recovered: bool, repo_name: str | None = None) -> dict[str, Any]: ...
    def recover_build(self) -> dict[str, Any]: ...

class AuditAdapter(Protocol):
    def write_audit_log(
        self,
        node_name: str,
        message: str,
        decision_trace: list[str],
        status: str,
        *,
        project_id: str | None = None,
        flight_stage: str | None = None,
        agent: str | None = None,
    ) -> AgentStep: ...


class InMemoryRagMemoryAdapter:
    """Delegates all retrieval to live Supabase + NVIDIA RAG (no mock snippets)."""

    def __init__(self) -> None:
        from agent.live_adapters import LiveRagMemoryAdapter

        self._live = LiveRagMemoryAdapter()

    async def index_source_urls(self, urls: list[str]) -> dict[str, Any]:
        return await self._live.index_source_urls(urls)

    async def retrieve_hackathon_context(self, idea: str) -> list[dict[str, Any]]:
        return await self._live.retrieve_hackathon_context(idea)

    async def retrieve_nvidia_context(self, idea: str) -> list[dict[str, Any]]:
        return await self._live.retrieve_nvidia_context(idea)

    async def retrieve_build_context(
        self,
        project_id: str,
        idea: str,
        *,
        optional_params: dict[str, Any] | None = None,
        rules_url: str | None = None,
        reference_urls: list[str] | None = None,
        context_needed: list[str] | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        return await self._live.retrieve_build_context(
            project_id,
            idea,
            optional_params=optional_params,
            rules_url=rules_url,
            reference_urls=reference_urls,
            context_needed=context_needed,
            top_k=top_k,
        )

    async def find_similar_builds(self, issue: str) -> list[dict[str, Any]]:
        return await self._live.find_similar_builds(issue)

    async def write_memory(self, memory: dict[str, Any]) -> None:
        await self._live.write_memory(memory)


class InMemoryToolAdapter:
    def create_repo(
        self,
        task_id: str,
        visibility: str,
        *,
        repo_preference: str = "create_new_repo",
        repo_name: str | None = None,
        repo_description: str | None = None,
        repo_url: str | None = None,
    ) -> dict[str, Any]:
        del repo_description
        repo_name = repo_name or f"mvpilot-generated-{task_id[:8]}"
        return {
            "tool": "github.create_repo" if repo_preference == "create_new_repo" else "github.use_existing_repo",
            "status": "success",
            "mock_mode": True,
            "recoverable": False,
            "repo": {
                "name": repo_name,
                "visibility": visibility,
                "url": repo_url or f"https://github.com/mock-org/{repo_name}",
            },
            "summary": (
                "Test mode: created deterministic GitHub repository record."
                if repo_preference == "create_new_repo"
                else "Test mode: attached deterministic existing GitHub repository record."
            ),
        }

    def commit_files(self, repo_name: str, files: list[dict[str, Any]], message: str) -> dict[str, Any]:
        return {
            "tool": "github.commit_files",
            "status": "success",
            "mock_mode": True,
            "recoverable": False,
            "repo": repo_name,
            "files": files,
            "commit_sha": "mock-commit-0001",
            "commit_url": f"https://github.com/mock-org/{repo_name}/commit/mock-commit-0001",
            "summary": "Test mode: committed generated MVP package files.",
        }

    def check_repo_health(self, repo_name: str) -> dict[str, Any]:
        return {
            "tool": "build.check_repo_health",
            "status": "success",
            "mock_mode": True,
            "healthy": True,
        }

    def detect_blocker(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "has_blocker": True,
            "blocker_type": "missing_dependency",
            "summary": "Test mode: missing idea-specific generated artifact.",
            "recommended_fix": "Add deterministic idea-specific artifact stub."
        }

    def verify_commit(self, repo_name: str, commit_sha: str) -> dict[str, Any]:
        return {
            "tool": "github.verify_commit",
            "status": "success",
            "mock_mode": True,
            "verification_status": "verified"
        }

    def verify_build(self, recovered: bool, repo_name: str | None = None) -> dict[str, Any]:
        if not recovered:
            return {
                "tool": "build.verify",
                "status": "failed",
                "mock_mode": True,
                "recoverable": True,
                "error": "Test mode: missing idea-specific generated artifact.",
                "summary": "Test mode: build failed with a recoverable artifact gap.",
            }
        return {
            "tool": "build.verify",
            "status": "success",
            "mock_mode": True,
            "recoverable": False,
            "checks": ["unit", "lint", "package"],
            "summary": "Test mode: build verification passed after recovery.",
        }

    def recover_build(self) -> dict[str, Any]:
        return {
            "tool": "build.apply_recovery_patch",
            "status": "success",
            "mock_mode": True,
            "recoverable": False,
            "patch": "Add deterministic idea-specific artifact stub.",
            "summary": "Test mode: applied recovery patch for the blocked build.",
        }


class InMemoryAuditAdapter:
    def __init__(self, model_name: str = "mock-model"):
        self._model_name = model_name

    def write_audit_log(
        self,
        node_name: str,
        message: str,
        decision_trace: list[str],
        status: str = "completed",
        *,
        project_id: str | None = None,
        flight_stage: str | None = None,
        agent: str | None = None,
    ) -> AgentStep:
        return AgentStep(
            project_id=project_id,
            flight_stage=flight_stage,
            agent=agent,
            node_name=node_name,
            status=status,
            message=message,
            model=self._model_name,
            decision_trace=[
                "Mock mode: deterministic Nemotron-style reasoning.",
                *decision_trace,
            ],
            timestamp=datetime.now(UTC),
        )
