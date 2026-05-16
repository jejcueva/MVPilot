"""Verification helpers for Person 3 GitHub actions."""

from __future__ import annotations

from pydantic import ValidationError

from tools import mock_store
from tools.github_tool import GitHubClient, GitHubConfig
from tools.policy import validate_generated_repo_name
from tools.schemas import ToolResult, VerificationResult


def _mock_verify_commit(repo_name: str, commit_sha: str) -> ToolResult:
    commit = mock_store.get_commit(repo_name, commit_sha)
    verified = commit is not None
    output = VerificationResult(
        commit_sha=commit_sha,
        verified=verified,
        files_changed=commit.files_changed if commit else [],
        error=None if verified else "Mock verifier only accepts synthetic mock commit SHAs.",
    ).model_dump()

    if verified:
        return ToolResult.mock("github.verify_commit", output)
    return ToolResult.failure(
        "github.verify_commit",
        "Mock verifier only accepts synthetic mock commit SHAs.",
        output,
        verification_status="failed",
    )


def verify_commit(
    repo_name: str,
    commit_sha: str,
    *,
    config: GitHubConfig | None = None,
    allow_existing_repo: bool = False,
) -> dict:
    """Confirm a commit exists in the generated repository and list changed files."""

    if not allow_existing_repo:
        repo_error = validate_generated_repo_name(repo_name)
        if repo_error:
            return repo_error.model_dump(mode="json")

    if not commit_sha.strip():
        return ToolResult.failure("github.verify_commit", "Commit SHA must not be empty.").model_dump(mode="json")

    active_config = config or GitHubConfig.from_env()
    if active_config.mock_tools:
        return _mock_verify_commit(repo_name, commit_sha).model_dump(mode="json")
    if not active_config.token:
        output = VerificationResult(
            commit_sha=commit_sha,
            verified=False,
            files_changed=[],
            error="GitHub OAuth token is required before MVPilot can verify commits.",
        ).model_dump()
        return ToolResult.failure(
            "github.verify_commit",
            "GitHub OAuth token is required before MVPilot can verify commits.",
            output,
            verification_status="failed",
        ).model_dump(mode="json")

    client = GitHubClient(active_config)
    try:
        commit = client.get_commit(repo_name, commit_sha)
        files_changed = [file_info["filename"] for file_info in commit.get("files", []) if "filename" in file_info]
        output = VerificationResult(
            commit_sha=commit.get("sha", commit_sha),
            verified=True,
            files_changed=files_changed,
        ).model_dump()
        return ToolResult.success("github.verify_commit", output).model_dump(mode="json")
    except (ValidationError, Exception) as exc:
        output = VerificationResult(
            commit_sha=commit_sha,
            verified=False,
            files_changed=[],
            error=str(exc),
        ).model_dump()
        return ToolResult.failure(
            "github.verify_commit",
            str(exc),
            output,
            verification_status="failed",
        ).model_dump(mode="json")
