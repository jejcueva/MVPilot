from __future__ import annotations

from time import perf_counter
from typing import TypeVar

from pydantic import BaseModel

from agent.generated_project import build_project_artifacts
from agent.project_depth import enrich_project_requirements
from agent.idea_context import (
    extract_idea_from_prompt,
    extract_json_section,
    features_from_context,
    project_title_from_context,
    target_users_from_context,
    tech_stack_from_context,
)
from agent.model_client import (
    ModelCallResult,
    ModelMode,
    _blocker_analysis_payload,
    _demo_script_payload,
    _final_readme_payload,
    _pitch_payload,
    _repo_plan_payload,
    _trace,
)
from agent.idea_context import title_from_idea as _idea_label
from agent.model_outputs import (
    BlockerAnalysisOutput,
    DemoScriptOutput,
    FileManifestOutput,
    FinalReadmeOutput,
    MvpScopeOutput,
    PitchOutput,
    RecommendedStackOutput,
    RepoPlanOutput,
)
from agent.stack_recommendation import (
    align_architecture_plan_with_recommended_stack,
    recommend_stack_heuristic,
)


OutputT = TypeVar("OutputT", bound=BaseModel)


class IdeaAwarePartialClient:
    """Explicit degraded mode: project-specific outputs, never a generic template."""

    def __init__(self, *, partial_reason: str) -> None:
        self._partial_reason = partial_reason

    async def complete_structured(
        self,
        *,
        purpose: str,
        model: str,
        prompt: str,
        response_model: type[OutputT],
        max_tokens: int = 1200,
        reasoning_effort: str = "medium",
    ) -> ModelCallResult[OutputT]:
        del max_tokens, reasoning_effort
        started = perf_counter()
        idea = extract_idea_from_prompt(prompt)
        intake = extract_json_section(prompt, "Frontend intake:\n")
        project_requirements = extract_json_section(prompt, "Project requirements:\n") or extract_json_section(prompt, "MVP scope:\n")
        architecture_plan = extract_json_section(prompt, "Architecture plan:\n") or extract_json_section(prompt, "Repo plan:\n")
        build_context = extract_json_section(prompt, "Structured build context:\n")
        resolved_stack = _resolved_stack_from_prompt(prompt, build_context)

        data = self._payload_for_purpose(
            purpose=purpose,
            idea=idea,
            intake=intake,
            project_requirements=project_requirements,
            architecture_plan=architecture_plan,
            build_context=build_context,
            resolved_stack=resolved_stack,
            prompt=prompt,
        )
        output = response_model.model_validate(data)
        return ModelCallResult(
            output=output,
            model=model,
            purpose=purpose,
            mode="degraded",
            latency_ms=max(0, round((perf_counter() - started) * 1000)),
            fallback_reason=self._partial_reason,
        )

    def _payload_for_purpose(
        self,
        *,
        purpose: str,
        idea: str,
        intake: dict,
        project_requirements: dict,
        architecture_plan: dict,
        build_context: dict,
        resolved_stack: str,
        prompt: str,
    ) -> dict:
        mode: ModelMode = "degraded"
        reason = self._partial_reason
        features = features_from_context(
            idea=idea,
            intake=intake,
            mvp_scope=project_requirements,
            repo_plan=architecture_plan,
        )
        title = project_title_from_context(idea=idea, intake=intake)
        target_users = target_users_from_context(intake) or "users described in the submitted idea"
        tech_stack = tech_stack_from_context(intake, architecture_plan) or resolved_stack

        if purpose in {"scope_mvp", "requirement_expansion"}:
            requirements_payload = enrich_project_requirements(
                {
                    "target_users": target_users,
                    "core_features": features,
                    "advanced_features": [
                        "Authenticated workspace",
                        "Operational dashboard",
                        "Generated project evidence log",
                    ],
                    "success_criteria": [
                        "Primary user can complete the end-to-end workflow.",
                        "Generated repository includes source, tests, docs, and deployment notes.",
                    ],
                    "project_depth": intake.get("projectDepth") or "Advanced Project",
                    "target_platform": intake.get("targetPlatform") or "web app",
                    "mode": mode,
                    "decision_trace": _trace(
                        mode,
                        reason,
                        [
                            f"Explicit degraded generation for: {_idea_label(idea)}",
                            "Expanded features from intake and plan because live Nemotron was unavailable.",
                            "Kept requirements specific to the submitted idea rather than a generic starter app.",
                        ],
                    ),
                },
                idea=idea,
                intake=intake,
            )
            requirements_payload["mode"] = mode
            requirements_payload["decision_trace"] = _trace(
                mode,
                reason,
                [
                    f"Explicit degraded generation for: {_idea_label(idea)}",
                    "Expanded features from intake and plan because live Nemotron was unavailable.",
                    "Kept requirements specific to the submitted idea rather than a generic starter app.",
                ],
            )
            return MvpScopeOutput.model_validate(requirements_payload).model_dump()

        if purpose == "recommend_stack":
            payload = recommend_stack_heuristic(
                idea=idea,
                project_requirements=project_requirements,
                build_context=build_context if isinstance(build_context, dict) else {},
            )
            payload["mode"] = mode
            payload["decision_trace"] = _trace(
                mode,
                reason,
                [
                    f"Explicit degraded stack recommendation for: {_idea_label(idea)}",
                    "Selected project-specific technologies from idea and RAG hints.",
                    "Did not default to MVPilot host stack.",
                ],
            )
            return RecommendedStackOutput.model_validate(payload).model_dump()

        if purpose in {"plan_repo", "architecture_plan"}:
            if not architecture_plan:
                plan_payload = _repo_plan_payload(mode, reason, idea, prompt)
            else:
                plan_payload = RepoPlanOutput.model_validate(
                    {
                        **architecture_plan,
                        "mode": mode,
                        "decision_trace": _trace(
                            mode,
                            reason,
                            [
                                "Reused repository plan from earlier orchestration step.",
                                f"Plan remains tied to {title}.",
                            ],
                        ),
                    }
                ).model_dump()
            recommended = (
                build_context.get("recommendedStack")
                if isinstance(build_context, dict)
                else None
            )
            return align_architecture_plan_with_recommended_stack(
                plan_payload,
                recommended if isinstance(recommended, dict) else None,
            )

        if purpose == "file_manifest":
            requirements = enrich_project_requirements(project_requirements or {}, idea=idea, intake=intake)
            artifacts = build_project_artifacts(
                idea=idea,
                title=title,
                resolved_stack=resolved_stack,
                architecture_plan=architecture_plan or requirements,
                source_warnings=_warnings_from_context(build_context),
                target_users=target_users,
                required_features=features,
                tech_stack_preference=tech_stack,
                project_requirements=requirements,
            )
            return FileManifestOutput(
                artifacts=artifacts,
                mode=mode,
                decision_trace=_trace(
                    mode,
                    reason,
                    [
                        f"Generated project-specific scaffold for {title} (degraded mode).",
                        f"Included {len(artifacts)} files aligned to the user's features.",
                        "External integrations are documented where credentials are not configured.",
                    ],
                ),
            ).model_dump()

        if purpose == "final_readme":
            return _final_readme_payload(mode, reason, idea, prompt)
        if purpose in {"demo_script", "walkthrough"}:
            return _demo_script_payload(mode, reason, idea, prompt)
        if purpose == "pitch":
            return _pitch_payload(mode, reason, idea, prompt)
        if purpose == "blocker_analysis":
            return _blocker_analysis_payload(mode, reason, idea)

        raise ValueError(f"Unsupported model purpose: {purpose}")


def _resolved_stack_from_prompt(prompt: str, build_context: dict) -> str:
    resolved = build_context.get("resolvedTechStack", {})
    if isinstance(resolved, dict):
        items = resolved.get("items")
        if isinstance(items, list):
            stack = [str(item).strip() for item in items if str(item).strip()]
            if stack:
                return ", ".join(stack)
    marker = "Resolved tech stack:\n"
    if marker in prompt:
        section = extract_json_section(prompt, marker)
        items = section.get("items") if isinstance(section, dict) else None
        if isinstance(items, list):
            stack = [str(item).strip() for item in items if str(item).strip()]
            if stack:
                return ", ".join(stack)
    recommended = build_context.get("recommendedStack")
    if isinstance(recommended, dict):
        from agent.stack_recommendation import recommended_stack_summary

        summary = recommended_stack_summary(recommended)
        if summary:
            return summary
    return "Project-specific stack pending Stack Selector Agent"


def _warnings_from_context(build_context: dict) -> list[dict[str, str]]:
    source_context = build_context.get("sourceContext", {})
    if not isinstance(source_context, dict):
        return []
    warnings = source_context.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [w for w in warnings if isinstance(w, dict)]
