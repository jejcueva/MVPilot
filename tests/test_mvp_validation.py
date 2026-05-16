from agent.generated_project import build_project_artifacts
from agent.mvp_validation import validate_mvp_output


def test_validate_mvp_output_passes_for_idea_specific_artifacts():
    idea = "StudyPilot helps students plan study sessions with spaced repetition."
    artifacts = build_project_artifacts(
        idea=idea,
        title="StudyPilot",
        resolved_stack="React, FastAPI",
        required_features=[
            "Capture study goals and session length",
            "Generate a spaced-repetition schedule",
        ],
    )
    validation = validate_mvp_output(
        idea=idea,
        intake={
            "title": "StudyPilot",
            "requiredFeatures": [
                "Capture study goals and session length",
                "Generate a spaced-repetition schedule",
            ],
        },
        mvp_scope={
            "must_have": [
                "Capture study goals and session length",
                "Generate a spaced-repetition schedule",
            ]
        },
        repo_plan={"files": ["README.md", "src/App.jsx"], "implementation_steps": ["a"]},
        generated_artifacts=artifacts,
        model_modes=["partial"],
    )

    assert validation["passed"] is True
    assert validation["project_title"] == "StudyPilot"


def test_validate_mvp_output_rejects_generic_feature_trio():
    idea = "Fleet maintenance copilot for mechanics."
    validation = validate_mvp_output(
        idea=idea,
        intake={"title": "Fleet Copilot"},
        mvp_scope={
            "must_have": [
                "Capture intake details for the target workflow",
                "Generate a prioritized mvp action plan",
                "Expose demo data through a small fastapi surface",
            ]
        },
        repo_plan={"files": ["README.md"]},
        generated_artifacts=[],
        model_modes=["fallback"],
    )

    assert validation["passed"] is False
    assert any(check["name"] == "no_generic_fallback_features" for check in validation["checks"])


def test_validate_mvp_output_prefers_latest_repair_artifact():
    idea = "Fleet maintenance copilot for mechanics."
    repaired = build_project_artifacts(
        idea=idea,
        title="Fleet Maintenance Copilot",
        resolved_stack="React, FastAPI",
        required_features=["Track fleet maintenance requests", "Prioritize mechanic repair work"],
    )
    validation = validate_mvp_output(
        idea=idea,
        intake={
            "title": "Fleet Maintenance Copilot",
            "requiredFeatures": ["Track fleet maintenance requests", "Prioritize mechanic repair work"],
        },
        mvp_scope={"must_have": ["Track fleet maintenance requests", "Prioritize mechanic repair work"]},
        repo_plan={"files": ["README.md", "src/App.jsx"], "implementation_steps": ["Generate workflow"]},
        generated_artifacts=[
            {
                "name": "README.md",
                "content": "# Starter app\n\nA generic todo app.",
            },
            *repaired,
        ],
        model_modes=["partial"],
    )

    assert validation["passed"] is True
