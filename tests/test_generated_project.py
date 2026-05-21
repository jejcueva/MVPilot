from agent.generated_project import (
    build_project_artifacts,
    merge_with_project_artifacts,
    title_from_idea,
)
from agent.project_generation import hydrate_file_manifest


def test_title_from_idea_strips_build_prefix() -> None:
    assert title_from_idea("Build a study planner for students").startswith("Study planner")


def test_build_project_artifacts_includes_runnable_stack() -> None:
    artifacts = build_project_artifacts(
        idea="Build a referral coordinator",
        title="Referral Coordinator",
        resolved_stack="React, FastAPI, Supabase",
    )
    names = {artifact["name"] for artifact in artifacts}
    assert "README.md" in names
    assert "package.json" in names
    assert "src/App.jsx" in names or "frontend/src/App.jsx" in names
    assert "backend/main.py" in names
    assert "docs/BUILD_LOG.md" in names
    assert "docs/WALKTHROUGH.md" in names or "demo/demo_script.md" in names


def test_hydrate_file_manifest_fills_scaffold_from_nemotron_outline() -> None:
    artifacts = hydrate_file_manifest(
        [{"name": "README.md", "kind": "markdown", "summary": "Nemotron overview."}],
        idea="Build StudyPilot for college students.",
        title="StudyPilot",
        resolved_stack="React, FastAPI",
        project_requirements={"must_have": ["weekly plan"]},
    )
    readme = next(item for item in artifacts if item["name"] == "README.md")
    assert readme["content"]
    assert readme["summary"] == "Nemotron overview."
    assert "StudyPilot" in readme["content"]
    assert any(item["name"] == "backend/main.py" for item in artifacts)

    package_json = next(artifact for artifact in artifacts if artifact["name"] == "package.json")
    assert '"vite": "^6.0.0"' in package_json["content"]
    assert '"@vitejs/plugin-react": "^4.3.4"' in package_json["content"]


def test_merge_with_project_artifacts_keeps_model_files() -> None:
    merged = merge_with_project_artifacts(
        [{"name": "custom/feature.md", "kind": "markdown", "content": "# Custom", "summary": "extra"}],
        idea="Build a planner",
        title="Planner",
        resolved_stack="React",
    )
    names = {artifact["name"] for artifact in merged}
    assert "custom/feature.md" in names
    assert "package.json" in names
