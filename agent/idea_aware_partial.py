from __future__ import annotations

from time import perf_counter
from typing import TypeVar

from pydantic import BaseModel

from agent.generated_project import build_project_artifacts
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
    RepoPlanOutput,
)


OutputT = TypeVar("OutputT", bound=BaseModel)


class IdeaAwarePartialClient:
    """Graceful degradation: idea-specific partial outputs, never a generic template."""

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
        mvp_scope = extract_json_section(prompt, "MVP scope:\n")
        repo_plan = extract_json_section(prompt, "Repo plan:\n")
        build_context = extract_json_section(prompt, "Structured build context:\n")
        resolved_stack = _resolved_stack_from_prompt(prompt, build_context)

        data = self._payload_for_purpose(
            purpose=purpose,
            idea=idea,
            intake=intake,
            mvp_scope=mvp_scope,
            repo_plan=repo_plan,
            build_context=build_context,
            resolved_stack=resolved_stack,
            prompt=prompt,
        )
        output = response_model.model_validate(data)
        return ModelCallResult(
            output=output,
            model=model,
            purpose=purpose,
            mode="partial",
            latency_ms=max(0, round((perf_counter() - started) * 1000)),
            fallback_reason=self._partial_reason,
        )

    def _payload_for_purpose(
        self,
        *,
        purpose: str,
        idea: str,
        intake: dict,
        mvp_scope: dict,
        repo_plan: dict,
        build_context: dict,
        resolved_stack: str,
        prompt: str,
    ) -> dict:
        mode: ModelMode = "partial"
        reason = self._partial_reason
        features = features_from_context(idea=idea, intake=intake, mvp_scope=mvp_scope, repo_plan=repo_plan)
        title = project_title_from_context(idea=idea, intake=intake)
        target_users = target_users_from_context(intake) or "users described in the submitted idea"
        tech_stack = tech_stack_from_context(intake, repo_plan) or resolved_stack

        if purpose == "scope_mvp":
            return MvpScopeOutput(
                target_user=target_users,
                must_have=features,
                demo_boundary=(
                    f"One end-to-end MVP path for {title}, with labeled mock integrations "
                    "only where APIs or credentials are unavailable."
                ),
                mode=mode,
                decision_trace=_trace(
                    mode,
                    reason,
                    [
                        f"Partial generation for: {_idea_label(idea)}",
                        "Scoped features from intake and plan because live Nemotron was unavailable.",
                        "Kept scope specific to the submitted idea rather than a generic starter app.",
                    ],
                ),
            ).model_dump()

        if purpose == "plan_repo":
            if not repo_plan:
                return _repo_plan_payload(mode, reason, idea, prompt)
            return RepoPlanOutput.model_validate(
                {
                    **repo_plan,
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

        if purpose == "file_manifest":
            artifacts = build_project_artifacts(
                idea=idea,
                title=title,
                resolved_stack=resolved_stack,
                repo_plan=repo_plan or mvp_scope,
                source_warnings=_warnings_from_context(build_context),
                target_users=target_users,
                required_features=features,
                tech_stack_preference=tech_stack,
            )
            return FileManifestOutput(
                artifacts=artifacts,
                mode=mode,
                decision_trace=_trace(
                    mode,
                    reason,
                    [
                        f"Generated idea-specific scaffold for {title} (partial mode).",
                        f"Included {len(artifacts)} files aligned to the user's features.",
                        "Mock data is labeled in code where external APIs are not wired.",
                    ],
                ),
            ).model_dump()

        if purpose == "final_readme":
            return _final_readme_payload(mode, reason, idea, prompt)
        if purpose == "demo_script":
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
    return "React, FastAPI, Postgres-ready schema, Pytest"


def _warnings_from_context(build_context: dict) -> list[dict[str, str]]:
    source_context = build_context.get("sourceContext", {})
    if not isinstance(source_context, dict):
        return []
    warnings = source_context.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [w for w in warnings if isinstance(w, dict)]
