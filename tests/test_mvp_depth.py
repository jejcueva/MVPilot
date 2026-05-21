from agent.mvp_depth import classify_vertical, default_demo_path, enrich_mvp_scope
from agent.generated_project import build_project_artifacts


def test_classify_vertical_prefers_study_planner_pack() -> None:
    assert classify_vertical(idea="Build a study planner for college students") == "planner"


def test_enrich_mvp_scope_adds_demo_path_and_routes() -> None:
    scope = enrich_mvp_scope(
        {"must_have": ["weekly plan", "course intake"], "target_user": "students"},
        idea="Study planner for exams",
        intake={},
    )
    assert scope["vertical_pack"] == "planner"
    assert len(scope["demo_path"]) >= 2
    assert scope["api_routes"]


def test_build_project_artifacts_includes_sqlite_and_deploy_docs() -> None:
    scope = enrich_mvp_scope({"must_have": ["plan courses"]}, idea="Study planner", intake={})
    artifacts = build_project_artifacts(
        idea="Study planner",
        title="Study Planner",
        resolved_stack="React, FastAPI",
        mvp_scope=scope,
    )
    names = {artifact["name"] for artifact in artifacts}
    assert "backend/db.py" in names
    assert "docs/DEPLOY.md" in names
    assert "docs/WALKTHROUGH.md" in names
    app = next(item for item in artifacts if item["name"] == "src/App.jsx")
    assert "collectionRoute" in app["content"]
    assert "collectionRoute" in app["content"] or "reviewQueue" in app["content"]
