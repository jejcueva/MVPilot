"""Stack Selector Agent — project-specific stack, not MVPilot host defaults."""

import pytest

from agent.config import Settings
from agent.openclaw_orchestrator import OpenClawOrchestrator
from agent.stack_recommendation import (
    HOST_PLATFORM_STACK_LABEL,
    align_architecture_plan_with_recommended_stack,
    apply_recommended_stack_to_build_context,
    recommend_stack_heuristic,
    stack_items_from_recommended,
)
from agent.rag.build_context import build_build_context_response
from agent.rag.types import BuildContextRequest


STUDYPILOT_IDEA = (
    "StudyPilot is an AI study platform that turns uploaded lecture notes into summaries, "
    "quizzes, flashcards, spaced repetition plans, and personalized exam prep dashboards."
)


def test_heuristic_studypilot_prefers_nemotron_when_hackathon_rules_mention_sponsors() -> None:
    build_context = {
        "frontendIntake": {
            "projectDepth": "Hackathon-Winning Project",
            "targetPlatform": "web app",
        },
        "requiredTechStackPieces": [
            {"item": "Use NVIDIA Nemotron for reasoning and embeddings", "priority": "critical"},
        ],
        "allowedToolsAndAPIs": [
            {"item": "OpenClaw orchestration for agent workflows", "priority": "high"},
        ],
        "scopeWarnings": [],
        "evidence": [],
    }
    stack = recommend_stack_heuristic(
        idea=STUDYPILOT_IDEA,
        project_requirements={
            "project_depth": "Hackathon-Winning Project",
            "target_platform": "web app",
        },
        build_context=build_context,
    )

    items_text = " ".join(stack_items_from_recommended(stack)).lower()
    assert "nemotron" in items_text
    assert "openclaw" in items_text or "orchestration" in items_text
    assert any("study" in reason.lower() or "studypilot" in reason.lower() for reason in stack["reasonForChoices"])
    assert any("mvpilot" in alt.lower() or "host" in alt.lower() for alt in stack["rejectedAlternatives"])
    assert "next.js" not in items_text or "study" in stack["frontend"].lower() or "next" in stack["frontend"].lower()


def test_apply_recommended_stack_sets_binding_resolved_stack() -> None:
    recommended = recommend_stack_heuristic(
        idea=STUDYPILOT_IDEA,
        project_requirements={"project_depth": "Advanced Project"},
        build_context={"frontendIntake": {}, "evidence": []},
    )
    merged = apply_recommended_stack_to_build_context(
        {
            "resolvedTechStack": {
                "source": "default",
                "items": [],
                "requiredItems": [],
                "defaultItems": ["Next.js", "FastAPI", "Supabase Postgres"],
                "reason": "host only",
            }
        },
        recommended,
    )
    assert merged["recommendedStack"] == recommended
    assert merged["resolvedTechStack"]["source"] == "stack_recommendation"
    assert merged["resolvedTechStack"]["items"]
    assert "Next.js" in merged["resolvedTechStack"]["defaultItems"]
    assert "nemotron" in " ".join(merged["resolvedTechStack"]["items"]).lower()


@pytest.mark.asyncio
async def test_build_context_does_not_merge_host_defaults_into_items(monkeypatch) -> None:
    async def empty_search(*_args, **_kwargs):
        return []

    monkeypatch.setattr("agent.rag.build_context.search_rag", empty_search)
    response = await build_build_context_response(
        BuildContextRequest(projectId="p1", idea=STUDYPILOT_IDEA, topK=4)
    )
    assert response.resolvedTechStack.source == "default"
    assert response.resolvedTechStack.items == []
    assert "Next.js" in response.resolvedTechStack.defaultItems
    assert HOST_PLATFORM_STACK_LABEL  # module constant exists for rejected alternatives


def test_align_architecture_plan_overwrites_selected_stack() -> None:
    recommended = recommend_stack_heuristic(
        idea=STUDYPILOT_IDEA,
        project_requirements={"project_depth": "Hackathon-Winning Project"},
        build_context={
            "allowedToolsAndAPIs": [{"item": "NVIDIA Nemotron required", "priority": "critical"}],
            "frontendIntake": {},
        },
    )
    plan = align_architecture_plan_with_recommended_stack(
        {"selected_stack": ["Next.js", "FastAPI", "Supabase Postgres"], "architecture_overview": []},
        recommended,
    )
    assert plan["selected_stack"] == stack_items_from_recommended(recommended)
    assert "nemotron" in " ".join(plan["selected_stack"]).lower()


def test_compose_project_plan_includes_recommended_stack() -> None:
    orchestrator = OpenClawOrchestrator(Settings(_env_file=None))
    recommended = recommend_stack_heuristic(
        idea=STUDYPILOT_IDEA,
        project_requirements={"project_depth": "Hackathon-Winning Project"},
        build_context={
            "allowedToolsAndAPIs": [{"item": "NVIDIA Nemotron required", "priority": "critical"}],
            "frontendIntake": {},
        },
    )
    plan = orchestrator.compose_project_plan(
        idea=STUDYPILOT_IDEA,
        intake={"title": "StudyPilot"},
        mvp_scope={"core_features": ["upload notes", "quizzes", "flashcards"]},
        repo_plan=None,
        build_context={"recommendedStack": recommended},
        recommended_stack=recommended,
    )
    assert plan.get("recommended_stack") == recommended
    assert plan.get("selected_stack")
    assert "nemotron" in str(plan.get("selected_stack")).lower() or any(
        "nemotron" in str(v).lower() for v in (recommended.get("aiModels") or [])
    )
