"""Repo health checks for generated MVP repositories."""

from __future__ import annotations

from tools import mock_store
from tools.github_tool import GitHubClient, GitHubConfig
from tools.policy import validate_generated_repo_name
from tools.schemas import RepoHealthResult, ToolResult


README_FILES = ["README.md"]
BUILD_LOG_FILES = ["docs/BUILD_LOG.md", "logs/build_log.md"]
DEMO_SCRIPT_FILES = ["demo/demo_script.md", "demo_script.md"]
ARCHITECTURE_FILES = ["docs/ARCHITECTURE.md"]
PACKAGE_FILES = ["package.json", "requirements.txt"]
SOURCE_PREFIXES = ["src/", "backend/"]

REPO_HEALTH_SCAFFOLD: dict[str, str] = {
    "README.md": "# Generated MVP\n\nTailored scaffold for the submitted idea.\n",
    "docs/BUILD_LOG.md": "# Build Log\n\n- Repository scaffold created by MVPilot.\n",
    "docs/ARCHITECTURE.md": "# Architecture\n\nIdea-specific MVP structure and integration notes.\n",
    "demo/demo_script.md": "# Demo Script\n\n1. Run the app.\n2. Show the core workflow.\n",
    "requirements.txt": "fastapi\nuvicorn\npytest\n",
    "src/__init__.py": "",
    "src/app.py": (
        '"""MVPilot generated entrypoint."""\n\n\n'
        "def main() -> None:\n"
        '    print("Generated MVP entrypoint")\n\n\n'
        'if __name__ == "__main__":\n'
        "    main()\n"
    ),
}


def idea_repo_health_scaffold(*, title: str, idea: str) -> dict[str, str]:
    """Fill only missing health-check paths with idea-specific stubs."""

    label = (title or "Generated MVP").strip()
    idea_line = " ".join((idea or "the submitted MVP idea").split())
    if len(idea_line) > 240:
        idea_line = f"{idea_line[:237].rstrip()}..."

    return {
        "README.md": (
            f"# {label}\n\n"
            f"MVPilot generated this package for: {idea_line}\n\n"
            "See docs/ARCHITECTURE.md and docs/BUILD_LOG.md for planning evidence.\n"
        ),
        "docs/BUILD_LOG.md": (
            f"# Build Log\n\n"
            f"- Scoped MVP: {label}\n"
            f"- Idea: {idea_line}\n"
            "- Health-check scaffold paths filled where the model omitted files.\n"
        ),
        "docs/ARCHITECTURE.md": (
            f"# Architecture\n\n"
            f"## {label}\n\n"
            f"Core concept: {idea_line}\n\n"
            "Frontend, API modules, and mock integrations are generated from the scoped MVP plan.\n"
        ),
        "demo/demo_script.md": (
            f"# Demo Script — {label}\n\n"
            f"1. Open with the idea: {idea_line}\n"
            "2. Walk through the primary UI workflow.\n"
            "3. Show mock API/data labeled in the UI.\n"
        ),
        "requirements.txt": "fastapi\nuvicorn\npytest\n",
        "src/__init__.py": "",
        "src/app.py": (
            f'"""API entrypoint for {label}."""\n\n\n'
            "def main() -> None:\n"
            f'    print("Serving partial MVP scaffold for: {label}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n"
        ),
    }


def _has_any_file_with_prefix(files: set[str], prefixes: list[str]) -> bool:
    return any(any(path.startswith(prefix) for path in files) for prefix in prefixes)


def _has_build_log(files: set[str]) -> bool:
    return any(path in files for path in BUILD_LOG_FILES)


def _has_demo_script(files: set[str]) -> bool:
    return any(path in files for path in DEMO_SCRIPT_FILES)


def _has_package_manifest(files: set[str]) -> bool:
    return any(path in files for path in PACKAGE_FILES)


def merge_repo_health_scaffold(
    files: list[dict[str, str]],
    *,
    idea: str | None = None,
    title: str | None = None,
) -> list[dict[str, str]]:
    """Ensure committed files satisfy repo health checks."""

    scaffold = (
        idea_repo_health_scaffold(title=title or "Generated MVP", idea=idea or "")
        if (idea or title)
        else REPO_HEALTH_SCAFFOLD
    )

    merged: dict[str, str] = {}
    for file in files:
        path = str(file.get("path") or file.get("name") or "").strip()
        if not path or path.endswith("/") or path.split("/")[-1] == ".env":
            continue
        content = file.get("content")
        if content is None:
            content = file.get("summary", "")
        if not isinstance(content, str):
            content = str(content)
        merged[path] = content

    paths = set(merged)
    for path, content in scaffold.items():
        if path in merged:
            continue
        if path in BUILD_LOG_FILES and _has_build_log(paths):
            continue
        if path in DEMO_SCRIPT_FILES and _has_demo_script(paths):
            continue
        if path in PACKAGE_FILES and _has_package_manifest(paths):
            continue
        if path.startswith("src/") and _has_any_file_with_prefix(paths, SOURCE_PREFIXES):
            continue
        merged[path] = content

    return [{"path": path, "content": content} for path, content in sorted(merged.items())]


def _health_result(repo_name: str, files: set[str], commit_count: int) -> dict:
    checks = {
        "README.md exists": "README.md" in files,
        "build log exists": _has_build_log(files),
        "architecture doc exists": any(path in files for path in ARCHITECTURE_FILES),
        "demo script exists": _has_demo_script(files),
        "package.json or requirements.txt exists": _has_package_manifest(files),
        "src/ or backend/ exists": _has_any_file_with_prefix(files, SOURCE_PREFIXES),
        "at least one commit exists": commit_count > 0,
    }
    missing = [name for name, passed in checks.items() if not passed]
    return RepoHealthResult(
        repo_name=repo_name,
        healthy=not missing,
        checks=checks,
        missing=missing,
    ).model_dump()


def check_repo_health(
    repo_name: str,
    *,
    config: GitHubConfig | None = None,
    allow_existing_repo: bool = False,
) -> dict:
    """Check whether a generated repo has the minimum demo artifacts."""

    if not allow_existing_repo:
        repo_error = validate_generated_repo_name(repo_name)
        if repo_error:
            return repo_error.model_dump(mode="json")

    active_config = config or GitHubConfig.from_env()
    if active_config.mock_tools:
        output = _health_result(repo_name, mock_store.list_files(repo_name), mock_store.commit_count(repo_name))
        status = "mock" if output["healthy"] else "failed"
        if status == "failed":
            return ToolResult.failure(
                "github.check_repo_health",
                "Generated repo is missing required files.",
                output,
                verification_status="failed",
            ).model_dump(mode="json")
        return ToolResult.mock("github.check_repo_health", output).model_dump(mode="json")
    if not active_config.token:
        return ToolResult.failure(
            "github.check_repo_health",
            "GitHub OAuth token is required before MVPilot can verify the generated repository.",
            {
                "repo_name": repo_name,
                "healthy": False,
                "authenticated": False,
                "required_action": "Reconnect GitHub through OAuth and retry.",
            },
            verification_status="failed",
        ).model_dump(mode="json")

    client = GitHubClient(active_config)
    files: set[str] = set()
    errors: list[str] = []
    try:
        client.get_repo(repo_name)
        for path in README_FILES + BUILD_LOG_FILES + DEMO_SCRIPT_FILES + ARCHITECTURE_FILES + PACKAGE_FILES:
            try:
                client.get_contents(repo_name, path)
                files.add(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        for prefix in SOURCE_PREFIXES:
            try:
                contents = client.get_contents(repo_name, prefix.rstrip("/"))
                if isinstance(contents, list) or contents.get("type") == "dir":
                    files.add(prefix)
            except Exception as exc:
                errors.append(f"{prefix}: {exc}")

        # If the repository exists, it has at least the auto-init commit created by create_repo.
        output = _health_result(repo_name, files, commit_count=1)
        if errors:
            output["errors"] = errors
        if not output["healthy"]:
            return ToolResult.failure(
                "github.check_repo_health",
                "Generated repo is missing required files.",
                output,
                verification_status="failed",
            ).model_dump(mode="json")
        return ToolResult.success("github.check_repo_health", output).model_dump(mode="json")
    except Exception as exc:
        output = RepoHealthResult(
            repo_name=repo_name,
            healthy=False,
            error=str(exc),
        ).model_dump()
        return ToolResult.failure(
            "github.check_repo_health",
            str(exc),
            output,
            verification_status="failed",
        ).model_dump(mode="json")
