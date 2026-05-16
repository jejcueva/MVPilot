from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from agent.adapters import ToolAdapter
from agent.config import Settings


OPENCLAW_TOOL_NAMES = [
    "github.create_repo",
    "github.commit_files",
    "github.append_build_log",
    "github.check_repo_health",
    "github.verify_commit",
    "build.detect_blocker",
    "build.verify",
    "build.apply_recovery_patch",
    "rag.get_build_context",
    "rag.search",
    "rag.reindex_logs",
]


def registered_openclaw_tools() -> list[str]:
    return list(OPENCLAW_TOOL_NAMES)


def openclaw_runtime_ready(settings: Settings) -> bool:
    return settings.openclaw_configured


def openclaw_runtime_status(settings: Settings) -> dict[str, Any]:
    ready = openclaw_runtime_ready(settings)
    return {
        "openclaw_runtime_ready": ready,
        "openclaw_registered_tools": registered_openclaw_tools() if ready else [],
    }


def runtime_name_for_settings(settings: Settings) -> str:
    return "openclaw" if openclaw_runtime_ready(settings) else "langgraph"


def registered_tools_for_settings(settings: Settings) -> list[str]:
    return registered_openclaw_tools() if openclaw_runtime_ready(settings) else []


class OpenClawToolAdapter:
    """OpenClaw-compatible tool boundary around MVPilot's existing tool adapter.

    Pair with `agent.openclaw_orchestrator.OpenClawOrchestrator` for phased MVP build
    telemetry; LangGraph nodes call the orchestrator while tools execute here.
    """

    def __init__(self, wrapped: ToolAdapter, *, environment: str | None = None) -> None:
        self._wrapped = wrapped
        self._environment = environment or "development"

    def set_github_config(self, config: Any) -> None:
        configure = getattr(self._wrapped, "set_github_config", None)
        if configure is None:
            raise AttributeError("Wrapped tool adapter cannot accept GitHub config.")
        configure(config)

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
        return self._with_openclaw_trace(
            "github.create_repo",
            self._wrapped.create_repo(
                task_id=task_id,
                visibility=visibility,
                repo_preference=repo_preference,
                repo_name=repo_name,
                repo_description=repo_description,
                repo_url=repo_url,
            ),
        )

    def commit_files(self, repo_name: str, files: list[dict[str, Any]], message: str) -> dict[str, Any]:
        return self._with_openclaw_trace(
            "github.commit_files",
            self._wrapped.commit_files(repo_name=repo_name, files=files, message=message),
        )

    def check_repo_health(self, repo_name: str) -> dict[str, Any]:
        return self._with_openclaw_trace(
            "github.check_repo_health",
            self._wrapped.check_repo_health(repo_name=repo_name),
        )

    def append_build_log(self, task_id: str, repo_name: str, message: str, data: dict[str, Any]) -> dict[str, Any]:
        from tools.repo_writer import append_build_log

        return self._with_openclaw_trace(
            "github.append_build_log",
            append_build_log(
                task_id=task_id,
                repo_name=repo_name,
                message=message,
                data=data,
            ),
        )

    def detect_blocker(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        return self._with_openclaw_trace(
            "build.detect_blocker",
            self._wrapped.detect_blocker(logs=logs),
        )

    def verify_commit(self, repo_name: str, commit_sha: str) -> dict[str, Any]:
        return self._with_openclaw_trace(
            "github.verify_commit",
            self._wrapped.verify_commit(repo_name=repo_name, commit_sha=commit_sha),
        )

    def verify_build(self, recovered: bool, repo_name: str | None = None) -> dict[str, Any]:
        return self._with_openclaw_trace(
            "build.verify",
            self._wrapped.verify_build(recovered=recovered, repo_name=repo_name),
        )

    def recover_build(self) -> dict[str, Any]:
        return self._with_openclaw_trace(
            "build.apply_recovery_patch",
            self._wrapped.recover_build(),
        )

    def _with_openclaw_trace(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        traced = deepcopy(result)
        traced["runtime"] = "openclaw"
        traced["openclaw_tool"] = tool_name
        traced["openclaw_trace"] = [
            {
                "runtime": "openclaw",
                "environment": self._environment,
                "event": "tool_executed",
                "tool": tool_name,
                "status": traced.get("status", "unknown"),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
        return traced
