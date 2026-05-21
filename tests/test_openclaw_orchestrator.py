from agent.build_timeline import default_build_timeline
from agent.config import Settings
from agent.openclaw_orchestrator import OpenClawOrchestrator


def test_orchestrator_initial_timeline_has_enriched_phases(monkeypatch) -> None:
    monkeypatch.delenv("OPENCLAW_API_KEY", raising=False)
    orchestrator = OpenClawOrchestrator(Settings(_env_file=None))
    extras = orchestrator.initial_state_extras()
    assert len(extras["build_timeline"]) == 21
    assert extras["build_timeline"][0]["id"] == "idea_intake"
    assert extras["build_timeline"][-1]["id"] == "final_project_report"
    assert extras["project_agents"]
    assert extras["runtime"] == "langgraph"


def test_orchestrator_records_completed_phase() -> None:
    orchestrator = OpenClawOrchestrator(Settings(_env_file=None))
    state = {"build_timeline": default_build_timeline()}
    update = orchestrator.record_phase(
        state,
        phase_id="requirement_expansion",
        status="completed",
        detail="Scoped project requirements",
        artifacts=["feature-a"],
    )
    planned = next(item for item in update["build_timeline"] if item["id"] == "requirement_expansion")
    assert planned["status"] == "completed"
    assert planned["detail"] == "Scoped project requirements"
    assert planned["artifacts"] == ["feature-a"]


def test_compose_mvp_plan_merges_intake_features() -> None:
    orchestrator = OpenClawOrchestrator(Settings(_env_file=None))
    plan = orchestrator.compose_mvp_plan(
        idea="Build a referral coordinator",
        intake={
            "title": "Referral MVP",
            "targetUsers": "Clinic staff",
            "techStackPreference": "React + FastAPI",
            "requiredFeatures": ["Intake", "Dashboard"],
        },
        mvp_scope={"must_have": ["Track referrals"]},
        repo_plan={
            "selected_stack": "React, FastAPI",
            "implementation_steps": ["Ship intake"],
            "files": [{"path": "src/App.jsx"}],
        },
        build_context={"evidence": [{"id": 1}]},
    )
    assert plan["title"] == "Referral MVP"
    assert plan["target_users"] == "Clinic staff"
    assert "Intake" in plan["features"]
    assert "Track referrals" in plan["features"]
    assert plan["rag_evidence_count"] == 1


def test_compose_mvp_plan_accepts_string_file_paths() -> None:
    orchestrator = OpenClawOrchestrator(Settings(_env_file=None))
    plan = orchestrator.compose_mvp_plan(
        idea="Build StudyPilot",
        intake={},
        mvp_scope={},
        repo_plan={"files": ["src/App.jsx", "backend/main.py"]},
        build_context={},
    )
    assert plan["files_planned"] == ["src/App.jsx", "backend/main.py"]
