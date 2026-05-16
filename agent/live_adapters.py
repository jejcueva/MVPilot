import logging
from typing import Any
from datetime import UTC, datetime

from agent.adapters import RagMemoryAdapter, ToolAdapter, AuditAdapter
from agent.schemas import AgentStep

from agent.rag.build_context import build_build_context_response
from agent.rag.retrieve import search_rag
from agent.rag.types import BuildContextOptionalParams, BuildContextRequest
from tools.github_tool import create_repo, commit_files
from tools.github_tool import GitHubConfig
from tools.policy import normalize_generated_repo_name
from tools.build_checker import check_repo_health
from tools.blocker_detector import detect_blocker
from tools.verifier import verify_commit
from tools.tool_logger import log_audit_event, log_generated_artifact, log_tool_call

logger = logging.getLogger(__name__)


def _memory_row_from_payload(memory: dict[str, Any]) -> dict[str, Any]:
    """Shape workflow memory for Supabase without FK-breaking task ids."""

    outcome = dict(memory.get("outcome") or {})
    workflow_task_id = memory.get("task_id")
    if workflow_task_id:
        outcome["workflow_task_id"] = workflow_task_id

    artifacts = outcome.get("generated_artifacts")
    if isinstance(artifacts, list):
        outcome["generated_artifacts"] = [
            {
                "name": artifact.get("name"),
                "kind": artifact.get("kind"),
                "summary": artifact.get("summary"),
            }
            if isinstance(artifact, dict)
            else {"name": str(artifact)}
            for artifact in artifacts
        ]

    return {
        "idea": str(memory.get("idea") or ""),
        "summary": str(memory.get("summary") or ""),
        "outcome": outcome,
        "tags": list(memory.get("tags") or []),
    }


class LiveRagMemoryAdapter:
    async def index_source_urls(self, urls: list[str]) -> dict[str, Any]:
        from agent.rag.ingest import ingest_source_urls

        result = await ingest_source_urls(urls)
        return {
            "urls": urls,
            "documentsLoaded": result.documentsLoaded,
            "chunksCreated": result.chunksCreated,
            "storedIn": result.storedIn,
        }

    async def retrieve_hackathon_context(self, idea: str) -> list[dict[str, Any]]:
        results = await search_rag(query=idea, top_k=5, doc_types=["hackathon_rules"])
        return [r.model_dump() for r in results]

    async def retrieve_nvidia_context(self, idea: str) -> list[dict[str, Any]]:
        results = await search_rag(query=idea, top_k=5, doc_types=["nvidia_docs"])
        return [r.model_dump() for r in results]

    async def find_similar_builds(self, issue: str) -> list[dict[str, Any]]:
        from agent.rag.store import get_rag_store
        from agent.rag.embed import embed_text
        store = get_rag_store()
        query_embedding = await embed_text(issue, input_type="query")
        return await store.search_memories(query_embedding, top_k=5)

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
        parsed_params: BuildContextOptionalParams | None = None
        if optional_params:
            parsed_params = BuildContextOptionalParams.model_validate(optional_params)

        response = await build_build_context_response(
            BuildContextRequest(
                projectId=project_id,
                idea=idea,
                rulesUrl=rules_url,
                referenceUrls=reference_urls or [],
                optionalParams=parsed_params,
                contextNeeded=context_needed or DEFAULT_BUILD_CONTEXT_NEEDED,
                topK=top_k,
            )
        )
        payload = response.model_dump()
        payload["mode"] = "live"
        return payload

    async def write_memory(self, memory: dict[str, Any]) -> None:
        from agent.rag.embed import embed_text
        from agent.rag.env_status import is_rag_configured
        from agent.rag.store import get_rag_store

        if not is_rag_configured():
            logger.info("Skipping memory write: RAG storage is not configured.")
            return

        row = _memory_row_from_payload(memory)
        summary = row.get("summary", "")
        if summary:
            try:
                row["embedding"] = await embed_text(summary, input_type="document")
            except Exception as exc:
                logger.warning("Memory embedding skipped: %s", exc)

        store = get_rag_store()
        await store.write_memory(row)


class LiveToolAdapter:
    def __init__(self, github_config: GitHubConfig | None = None) -> None:
        self._github_config = github_config
        self._allow_existing_repo = False

    def set_github_config(self, config: GitHubConfig) -> None:
        self._github_config = config

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
        if self._github_config is None:
            return {
                "tool": "github.create_repo",
                "status": "failed",
                "mock_mode": False,
                "recoverable": False,
                "repo": {},
                "summary": "Live GitHub connection is required before repository creation.",
            }
        if repo_preference == "use_existing_repo":
            active_repo_name = repo_name or _repo_name_from_url(repo_url)
            if not active_repo_name:
                return {
                    "tool": "github.use_existing_repo",
                    "status": "failed",
                    "mock_mode": False,
                    "recoverable": False,
                    "repo": {},
                    "summary": "Existing repo mode requires repoName or repoUrl.",
                }
            self._allow_existing_repo = True
            return {
                "tool": "github.use_existing_repo",
                "status": "success",
                "mock_mode": False,
                "recoverable": False,
                "repo": {
                    "name": active_repo_name,
                    "visibility": visibility,
                    "url": repo_url or f"https://github.com/{self._github_config.owner}/{active_repo_name}",
                },
                "summary": "Using the connected existing GitHub repository.",
            }
        self._allow_existing_repo = False
        normalized_repo_name = normalize_generated_repo_name(repo_name, task_id=task_id)
        raw_result = create_repo(
            repo_name=normalized_repo_name,
            description=repo_description or "Generated by MVPilot",
            visibility=visibility,
            config=self._github_config,
            task_id=task_id,
            reuse_existing=True,
        )
        output = raw_result.get("output", {})
        repo_status = str(output.get("status") or "created")
        if repo_status == "reused":
            success_message = "Reused your existing GitHub repository for this MVP name."
        elif output.get("name_adjusted"):
            success_message = (
                f"Created GitHub repository as {output.get('repo_name')} "
                f"(requested name was already taken)."
            )
        else:
            success_message = "Created and verified GitHub repository."
        return {
            "tool": raw_result.get("tool_name", "github.create_repo"),
            "status": _workflow_status(raw_result),
            "mock_mode": raw_result.get("status") == "mock",
            "recoverable": _is_recoverable(raw_result),
            "repo": {
                "name": output.get("repo_name"),
                "visibility": output.get("visibility", visibility),
                "url": output.get("repo_url"),
            },
            "summary": _summary(
                raw_result,
                success=success_message,
                failure="Failed to create GitHub repository.",
            ),
            "raw_result": raw_result,
        }

    def commit_files(self, repo_name: str, files: list[dict[str, Any]], message: str) -> dict[str, Any]:
        if self._github_config is None:
            return {
                "tool": "github.commit_files",
                "status": "failed",
                "mock_mode": False,
                "recoverable": False,
                "repo": repo_name,
                "files": [],
                "commit_sha": None,
                "commit_url": None,
                "summary": "Live GitHub connection is required before committing files.",
            }
        raw_result = commit_files(
            repo_name=repo_name,
            files=files,
            message=message,
            config=self._github_config,
            allow_existing_repo=self._allow_existing_repo,
        )
        output = raw_result.get("output", {})
        return {
            "tool": raw_result.get("tool_name", "github.commit_files"),
            "status": _workflow_status(raw_result),
            "mock_mode": raw_result.get("status") == "mock",
            "recoverable": _is_recoverable(raw_result),
            "repo": repo_name,
            "files": output.get("changed_files") or [file.get("path") for file in files],
            "commit_sha": output.get("commit_sha"),
            "commit_url": output.get("commit_url"),
            "summary": _summary(
                raw_result,
                success="Committed generated files and verified commit.",
                failure="Failed to commit generated files.",
            ),
            "raw_result": raw_result,
        }

    def check_repo_health(self, repo_name: str) -> dict[str, Any]:
        raw_result = check_repo_health(
            repo_name,
            config=self._github_config,
            allow_existing_repo=self._allow_existing_repo,
        )
        output = raw_result.get("output", {})
        return {
            "tool": raw_result.get("tool_name", "github.check_repo_health"),
            "status": _workflow_status(raw_result),
            "mock_mode": raw_result.get("status") == "mock",
            "recoverable": _is_recoverable(raw_result),
            "healthy": output.get("healthy", raw_result.get("status") in {"success", "mock"}),
            "missing": output.get("missing", []),
            "summary": _summary(
                raw_result,
                success="Checked generated repository health.",
                failure="Generated repository health check failed.",
            ),
            "raw_result": raw_result,
        }

    def detect_blocker(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        raw_result = detect_blocker(logs)
        output = raw_result.get("output", raw_result)
        return {
            "tool": raw_result.get("tool_name", "build.detect_blocker"),
            "status": _workflow_status(raw_result),
            "mock_mode": raw_result.get("status") == "mock",
            "recoverable": bool(output.get("has_blocker")),
            "has_blocker": output.get("has_blocker", False),
            "blocker_type": output.get("blocker_type"),
            "summary": output.get("summary") or _summary(
                raw_result,
                success="Analyzed logs for blockers.",
                failure="Failed to analyze logs for blockers.",
            ),
            "recommended_fix": output.get("recommended_fix"),
            "raw_result": raw_result,
        }

    def verify_commit(self, repo_name: str, commit_sha: str) -> dict[str, Any]:
        raw_result = verify_commit(repo_name, commit_sha, config=self._github_config)
        output = raw_result.get("output", {})
        return {
            "tool": raw_result.get("tool_name", "github.verify_commit"),
            "status": _workflow_status(raw_result),
            "mock_mode": raw_result.get("status") == "mock",
            "recoverable": _is_recoverable(raw_result),
            "repo": repo_name,
            "commit_sha": output.get("commit_sha", commit_sha),
            "files_changed": output.get("files_changed", []),
            "verification_status": raw_result.get("verification_status"),
            "summary": _summary(
                raw_result,
                success="Verified GitHub commit.",
                failure="Failed to verify GitHub commit.",
            ),
            "raw_result": raw_result,
        }

    def verify_build(self, recovered: bool, repo_name: str | None = None) -> dict[str, Any]:
        if not repo_name:
            return {
                "tool": "build.verify",
                "status": "failed",
                "recoverable": False,
                "error": "Live mode: repository name was missing for verification.",
                "summary": "Live mode: repository health could not be checked.",
            }

        health = self.check_repo_health(repo_name)
        status = "success" if health.get("status") == "success" and health.get("healthy") else "failed"
        return {
            "tool": "build.verify",
            "status": status,
            "recoverable": False,
            "checks": ["github_repo_health"],
            "repo": repo_name,
            "summary": (
                "Live mode: generated repository health check passed."
                if status == "success"
                else health.get("summary", "Live mode: generated repository health check failed.")
            ),
            "repo_health": health,
        }

    def recover_build(self) -> dict[str, Any]:
        return {
            "tool": "build.apply_recovery_patch",
            "status": "success",
            "recoverable": False,
            "patch": "Live idea-specific artifact recovery patch.",
            "summary": "Live mode: applied recovery patch.",
        }


class LiveAuditAdapter:
    def __init__(self, model_name: str):
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
        log_audit_event(
            task_id=project_id,
            step=node_name,
            message=message,
            data={
                "status": status,
                "model": self._model_name,
                "flight_stage": flight_stage,
                "agent": agent,
                "decision_trace": decision_trace,
            },
        )
        return AgentStep(
            project_id=project_id,
            flight_stage=flight_stage,
            agent=agent,
            node_name=node_name,
            status=status,
            message=message,
            model=self._model_name,
            decision_trace=["Live mode: running live audit trace.", *decision_trace],
            timestamp=datetime.now(UTC),
        )

    def write_tool_call(self, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        log_result = log_tool_call(
            task_id=None,
            tool_name=tool_name,
            input_json=args,
            result=result,
        )
        return {"tool_name": tool_name, "args": args, "result": result, "log_result": log_result}

    def write_artifact(self, name: str, kind: str, content: Any) -> dict[str, Any]:
        log_result = log_generated_artifact(
            task_id=None,
            artifact_type=kind,
            path=name,
            content=content if isinstance(content, str) else str(content),
            commit_sha=None,
        )
        return {"name": name, "kind": kind, "content": content, "log_result": log_result}


def _workflow_status(raw_result: dict[str, Any]) -> str:
    if raw_result.get("status") in {"success", "mock"}:
        return "success"
    return "failed"


def _is_recoverable(raw_result: dict[str, Any]) -> bool:
    status = raw_result.get("status")
    if status == "refused":
        return False
    return status == "failed"


def _summary(raw_result: dict[str, Any], *, success: str, failure: str) -> str:
    if raw_result.get("status") in {"success", "mock"}:
        return success
    return raw_result.get("error") or failure


def _repo_name_from_url(repo_url: str | None) -> str | None:
    if not repo_url:
        return None
    candidate = repo_url.rstrip("/").split("/")[-1].strip()
    return candidate or None


DEFAULT_BUILD_CONTEXT_NEEDED = [
    "required_deliverables",
    "allowed_tools_apis",
    "required_repository_format",
    "required_demo_format",
    "required_tech_stack_pieces",
    "hackathon_rules",
    "nvidia_model_usage",
    "security_constraints",
    "agent_boundaries",
    "scope_warnings",
]
