from __future__ import annotations

from typing import Any

from agent.generated_project import build_project_artifacts, merge_with_project_artifacts


def generate_mvp_artifacts(
    *,
    idea: str,
    title: str | None,
    resolved_stack: str,
    repo_plan: dict[str, Any] | None = None,
    source_warnings: list[dict[str, str]] | None = None,
    target_users: str | None = None,
    required_features: list[str] | None = None,
    tech_stack_preference: str | None = None,
) -> list[dict[str, str]]:
    return build_project_artifacts(
        idea=idea,
        title=title,
        resolved_stack=resolved_stack,
        repo_plan=repo_plan,
        source_warnings=source_warnings,
        target_users=target_users,
        required_features=required_features,
        tech_stack_preference=tech_stack_preference,
    )


def merge_model_manifest(
    artifacts: list[dict[str, Any]],
    *,
    idea: str,
    title: str | None,
    resolved_stack: str,
    repo_plan: dict[str, Any] | None = None,
    source_warnings: list[dict[str, str]] | None = None,
    target_users: str | None = None,
    required_features: list[str] | None = None,
    tech_stack_preference: str | None = None,
) -> list[dict[str, Any]]:
    return merge_with_project_artifacts(
        artifacts,
        idea=idea,
        title=title,
        resolved_stack=resolved_stack,
        repo_plan=repo_plan,
        source_warnings=source_warnings,
        target_users=target_users,
        required_features=required_features,
        tech_stack_preference=tech_stack_preference,
    )


def artifact_groups(artifacts: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "frontend": [],
        "backend": [],
        "docs": [],
        "tests": [],
        "config": [],
    }
    for artifact in artifacts:
        name = str(artifact.get("name") or "")
        if not name:
            continue
        if name.startswith("src/") or name.endswith((".jsx", ".tsx", ".css", ".html")):
            groups["frontend"].append(name)
        elif name.startswith("backend/"):
            groups["backend"].append(name)
        elif name.startswith("docs/") or name.startswith("demo/"):
            groups["docs"].append(name)
        elif name.startswith("tests/"):
            groups["tests"].append(name)
        else:
            groups["config"].append(name)
    return groups
