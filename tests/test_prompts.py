from agent.prompts import (
    build_demo_script_prompt,
    build_file_manifest_prompt,
    build_final_readme_prompt,
    build_pitch_prompt,
    build_plan_repo_prompt,
    build_scope_mvp_prompt,
)


def _default_build_context() -> dict:
    return {
        "requiredDeliverables": [],
        "allowedToolsAndAPIs": [],
        "requiredRepositoryFormat": [],
        "requiredDemoFormat": [],
        "requiredTechStackPieces": [],
        "resolvedTechStack": {
            "source": "default",
            "items": [
                "Next.js",
                "React",
                "TypeScript",
                "Tailwind CSS",
                "Python 3.12",
                "FastAPI",
                "Uvicorn",
                "Supabase Postgres",
                "pgvector",
                "NVIDIA Nemotron",
                "pytest",
                "npm run build",
            ],
            "requiredItems": [],
            "defaultItems": [
                "Next.js",
                "React",
                "TypeScript",
                "Tailwind CSS",
                "Python 3.12",
                "FastAPI",
                "Uvicorn",
                "Supabase Postgres",
                "pgvector",
                "NVIDIA Nemotron",
                "pytest",
                "npm run build",
            ],
            "reason": "No required or preferred stack was found.",
        },
        "frontendIntake": {
            "title": "Study Planner",
            "idea": "Build a study planner",
            "source": "mvpilot_frontend",
            "primaryRulesUrl": "https://example.com/rules",
            "additionalUrls": [],
            "uploadedFiles": [],
            "githubConnected": True,
            "githubConnectionId": "gh_conn_123",
        },
        "sourceContext": {
            "primaryRulesUrl": {
                "url": "https://example.com/rules",
                "summary": "Must include a working demo.",
            },
            "additionalUrls": [],
            "uploadedFiles": [],
            "warnings": [
                {
                    "source": "https://example.com/missing",
                    "message": "Could not read submitted URL: HTTP 404",
                }
            ],
            "sourceCounts": {"warnings": 1},
        },
    }


def test_scope_mvp_prompt_includes_resolved_tech_stack_and_override_rule() -> None:
    prompt = build_scope_mvp_prompt(
        idea="Build a judge helper",
        build_context=_default_build_context(),
        memory_matches=[],
    )

    assert "resolvedTechStack" in prompt
    assert "frontendIntake" in prompt
    assert "sourceContext" in prompt
    assert "Frontend intake is the user's source of truth" in prompt
    assert "missing or unreadable sources as warnings" in prompt
    assert "recommendedStack" in prompt or "RAG stack hints" in prompt
    assert "Do not assume the generated project must use MVPilot" in prompt
    assert "Retrieved docs" not in prompt


def test_plan_repo_prompt_includes_default_stack_when_rag_is_silent() -> None:
    prompt = build_plan_repo_prompt(
        idea="Build a judge helper",
        project_requirements={"must_have": ["intake"]},
        build_context=_default_build_context(),
    )

    assert "Recommended stack (binding)" in prompt or "recommendedStack" in prompt
    assert "Study Planner" in prompt
    assert "Frontend intake is the user's source of truth" in prompt
    assert "Do not assume the generated project must use MVPilot" in prompt
    assert "full generated repository architecture" in prompt


def test_artifact_prompts_include_intake_source_context_and_stack_rules() -> None:
    build_context = _default_build_context()

    manifest_prompt = build_file_manifest_prompt(
        idea="Build a study planner",
        project_requirements={
            "vertical_pack": "planner",
            "demo_path": [{"step": "1", "screen": "Plan", "action": "Review plan", "api": "GET /api/items"}],
            "api_routes": ["/api/items"],
        },
        architecture_plan={"files": ["README.md"]},
        build_context=build_context,
    )
    readme_prompt = build_final_readme_prompt(
        idea="Build a study planner",
        project_requirements={"must_have": ["intake"]},
        architecture_plan={"files": ["README.md"]},
        generated_artifacts=[],
        build_context=build_context,
    )
    demo_prompt = build_demo_script_prompt(
        idea="Build a study planner",
        blocker_analysis=None,
        build_context=build_context,
    )
    pitch_prompt = build_pitch_prompt(
        idea="Build a study planner",
        final_readme={"title": "Study Planner"},
        walkthrough={"title": "Demo"},
        build_context=build_context,
    )

    for prompt in [manifest_prompt, readme_prompt, demo_prompt, pitch_prompt]:
        assert "frontendIntake" in prompt
        assert "sourceContext" in prompt
        assert "recommendedStack" in prompt or "resolvedTechStack" in prompt
        assert "Required RAG rules override user preference" in prompt
        assert "MVPilot" in prompt
