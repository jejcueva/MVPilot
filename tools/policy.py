"""Safety policy for MVPilot's live tool actions."""

from __future__ import annotations

import os
import re
from typing import Iterable

from pydantic import ValidationError

from tools.schemas import FilePayload, ToolResult, safe_validation_errors


DEFAULT_REPO_PREFIX = "mvpilot-generated-"
ALLOWED_ACTIONS = {
    "create_repo",
    "commit_files",
    "append_build_log",
    "check_repo_health",
    "verify_commit",
    "detect_blocker",
    "read_repo",
}
BLOCKED_ACTIONS = {
    "delete_repo",
    "force_push",
    "rewrite_history",
    "change_secrets",
    "modify_unrelated_repo",
}
_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")


def repo_prefix() -> str:
    return os.getenv("GITHUB_REPO_PREFIX", DEFAULT_REPO_PREFIX)


def repo_name_candidates(
    repo_name: str | None,
    *,
    task_id: str | None = None,
    max_candidates: int = 6,
) -> list[str]:
    """Return ordered repo names to try (base first, then unique suffixes)."""

    base = normalize_generated_repo_name(repo_name, task_id=task_id)
    candidates = [base]
    slug = re.sub(r"[^a-z0-9]", "", (task_id or "").lower())[:6]
    if slug:
        tagged = f"{base}-{slug}"
        if tagged not in candidates and len(tagged) <= 100 and _REPO_NAME_RE.fullmatch(tagged):
            candidates.append(tagged)
    for index in range(2, max_candidates + 1):
        candidate = f"{base}-{index}"
        if candidate in candidates:
            continue
        if len(candidate) > 100 or not _REPO_NAME_RE.fullmatch(candidate):
            break
        candidates.append(candidate)
    return candidates


def normalize_generated_repo_name(
    repo_name: str | None,
    *,
    task_id: str | None = None,
) -> str:
    """Ensure user-supplied names satisfy the generated-repo prefix policy."""

    prefix = repo_prefix()
    generated_default = f"{prefix}{(task_id or 'generated')[:8]}"
    name = (repo_name or generated_default).strip()
    if not name:
        return generated_default
    if name.startswith(prefix):
        return name
    if name.startswith("mvpilot-") and not name.startswith(prefix):
        name = prefix + name[len("mvpilot-") :]
    elif not name.startswith(prefix):
        name = f"{prefix}{name}"
    if not _REPO_NAME_RE.fullmatch(name):
        return generated_default
    return name


def validate_action(action: str) -> ToolResult | None:
    normalized = action.strip().lower()
    if normalized in BLOCKED_ACTIONS:
        return ToolResult.refused("policy.validate_action", f"Action '{action}' is blocked by policy.")
    if normalized not in ALLOWED_ACTIONS:
        return ToolResult.refused("policy.validate_action", f"Action '{action}' is not allowlisted.")
    return None


def validate_generated_repo_name(repo_name: str) -> ToolResult | None:
    name = repo_name.strip()
    prefix = repo_prefix()

    if not name:
        return ToolResult.refused("policy.validate_repo", "Repository name must not be empty.")
    if "/" in name or "\\" in name:
        return ToolResult.refused("policy.validate_repo", "Repository name must not include an owner or path.")
    if not name.startswith(prefix):
        return ToolResult.refused(
            "policy.validate_repo",
            f"Repository name must start with '{prefix}'.",
            {"repo_name": name},
        )
    if not _REPO_NAME_RE.fullmatch(name):
        return ToolResult.refused("policy.validate_repo", "Repository name contains unsupported characters.")
    return None


def validate_file_payloads(files: Iterable[dict | FilePayload]) -> ToolResult | None:
    for file_payload in files:
        try:
            FilePayload.model_validate(file_payload)
        except ValidationError as exc:
            return ToolResult.refused(
                "policy.validate_file_payloads",
                "File payload failed safety validation.",
                {"validation_error": safe_validation_errors(exc.errors(include_url=False))},
            )
    return None


def validate_github_mutation(
    action: str,
    repo_name: str,
    files: Iterable[dict | FilePayload] | None = None,
) -> ToolResult | None:
    action_error = validate_action(action)
    if action_error:
        return action_error

    repo_error = validate_generated_repo_name(repo_name)
    if repo_error:
        return repo_error

    if files is not None:
        file_error = validate_file_payloads(files)
        if file_error:
            return file_error

    return None
