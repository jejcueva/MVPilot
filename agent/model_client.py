from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agent.config import Settings
from agent.generated_project import build_project_artifacts
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


OutputT = TypeVar("OutputT", bound=BaseModel)
ModelMode = Literal["mock", "live", "degraded", "partial"]


class ModelClientError(Exception):
    def __init__(self, safe_message: str) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message


class RetryableModelClientError(ModelClientError):
    pass


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str | None:
    cleaned = _strip_json_fences(text)
    start = cleaned.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return None


def _parse_json_text(raw: str) -> Any:
    candidates: list[str] = []
    stripped = _strip_json_fences(raw)
    candidates.append(stripped)
    extracted = _extract_json_object(stripped)
    if extracted and extracted not in candidates:
        candidates.insert(0, extracted)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found in Nemotron response.", raw, 0)


def _json_decode_looks_truncated(exc: BaseException | None) -> bool:
    if not isinstance(exc, json.JSONDecodeError):
        return False
    return "unterminated" in str(exc).lower()


def _bump_max_tokens(current: int, *, cap: int = 12_000) -> int:
    return min(max(current * 2, current + 2000), cap)


@dataclass(frozen=True)
class ModelCallResult[OutputT]:
    output: OutputT
    model: str
    purpose: str
    mode: ModelMode
    latency_ms: int
    fallback_reason: str | None = None


class ModelClient(Protocol):
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
        """Return a structured model output for the requested purpose."""


class DeterministicModelClient:
    def __init__(
        self,
        *,
        mode: ModelMode = "mock",
        fallback_reason: str | None = None,
    ) -> None:
        self._mode = mode
        self._fallback_reason = fallback_reason

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
        data = _deterministic_payload(
            purpose,
            self._mode,
            self._fallback_reason,
            prompt,
        )
        output = response_model.model_validate(data)
        return ModelCallResult(
            output=output,
            model=model,
            purpose=purpose,
            mode=self._mode,
            latency_ms=_elapsed_ms(started),
            fallback_reason=self._fallback_reason,
        )


class NemotronModelClient:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._settings = settings

    def _allows_partial_fallback(self, purpose: str) -> bool:
        if not self._settings.allow_idea_aware_partial:
            return False
        if purpose == "file_manifest" and self._settings.require_live_file_manifest:
            return False
        return True

    @staticmethod
    async def _backoff_before_retry(reason: str, attempt: int, *, purpose: str) -> None:
        if "504" not in reason:
            return
        cap = 90.0 if purpose == "plan_repo" else 45.0
        await asyncio.sleep(min(cap, 2.0 ** (attempt + 1)))

    async def complete_structured(
        self,
        *,
        purpose: str,
        model: str,
        prompt: str,
        response_model: type[OutputT],
        max_tokens: int = 1200,
        reasoning_effort: str | None = None,
    ) -> ModelCallResult[OutputT]:
        if not self._settings.nvidia_configured:
            return await self._idea_specific_partial(
                purpose=purpose,
                model=model,
                prompt=prompt,
                response_model=response_model,
                reason="Missing NVIDIA_API_KEY for live mode.",
            )

        started = perf_counter()
        attempts = max(0, self._settings.nemotron_max_retries_for(purpose)) + 1
        last_reason = "Nemotron request failed."
        effort = reasoning_effort or self._settings.nemotron_reasoning_effort
        effective_max_tokens = max(
            max_tokens,
            self._settings.nemotron_max_tokens_for(purpose),
        )

        for attempt in range(attempts):
            try:
                payload = await self._send_completion_request(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    max_tokens=effective_max_tokens,
                    reasoning_effort=effort,
                )
                return self._parse_completion_payload(
                    payload=payload,
                    purpose=purpose,
                    model=model,
                    response_model=response_model,
                    started=started,
                )
            except RetryableModelClientError as exc:
                last_reason = exc.safe_message
                if (
                    last_reason == "Nemotron returned truncated JSON."
                    and effective_max_tokens < 12_000
                ):
                    effective_max_tokens = _bump_max_tokens(effective_max_tokens)
                if attempt + 1 < attempts:
                    await self._backoff_before_retry(
                        last_reason, attempt, purpose=purpose
                    )
                    continue
                if not self._allows_partial_fallback(purpose):
                    raise ModelClientError(
                        f"Live Nemotron is required for full project generation ({last_reason}). "
                        "Set NEMOTRON_API_KEY/NVIDIA_API_KEY or enable degraded mode explicitly."
                    ) from exc
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=last_reason,
                    started=started,
                )
            except ModelClientError as exc:
                if not self._allows_partial_fallback(purpose):
                    raise
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=exc.safe_message,
                    started=started,
                )
            except httpx.TimeoutException as exc:
                last_reason = "Nemotron request timed out."
                if attempt + 1 < attempts:
                    await self._backoff_before_retry(
                        last_reason, attempt, purpose=purpose
                    )
                    continue
                if not self._allows_partial_fallback(purpose):
                    raise ModelClientError(
                        f"Live Nemotron is required for full project generation ({last_reason}). "
                        "Set NEMOTRON_API_KEY/NVIDIA_API_KEY or enable degraded mode explicitly."
                    ) from exc
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=last_reason,
                    started=started,
                )
            except httpx.HTTPError as exc:
                last_reason = "Nemotron request failed."
                if attempt + 1 < attempts:
                    await self._backoff_before_retry(
                        last_reason, attempt, purpose=purpose
                    )
                    continue
                if not self._allows_partial_fallback(purpose):
                    raise ModelClientError(
                        f"Live Nemotron is required for full project generation ({last_reason}). "
                        "Set NEMOTRON_API_KEY/NVIDIA_API_KEY or enable degraded mode explicitly."
                    ) from exc
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=last_reason,
                    started=started,
                )

        if not self._allows_partial_fallback(purpose):
            raise ModelClientError(
                f"Live Nemotron is required for full project generation ({last_reason}). "
                "Set NEMOTRON_API_KEY/NVIDIA_API_KEY or enable degraded mode explicitly."
            )
        return await self._idea_specific_partial(
            purpose=purpose,
            model=model,
            prompt=prompt,
            response_model=response_model,
            reason=last_reason,
            started=started,
        )

    async def _send_completion_request(
        self,
        *,
        purpose: str,
        model: str,
        prompt: str,
        response_model: type[OutputT],
        max_tokens: int,
        reasoning_effort: str,
    ) -> dict:
        url = f"{self._settings.nemotron_base_url.rstrip('/')}/chat/completions"
        if purpose == "file_manifest":
            system_content = (
                "You are MVPilot's full project repository manifest planner. Return only one "
                "complete JSON object matching the guided schema. List every required "
                "file path with kind and a short summary. Do not include file bodies or "
                "content fields. No markdown fences, prose, or trailing commas."
            )
        else:
            system_content = (
                "You are MVPilot's full project planning model. Return only "
                "one complete JSON object that matches the guided schema. "
                "No markdown fences, prose, or trailing commas. Close all "
                "strings, arrays, and objects."
            )
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": False,
            "reasoning_effort": reasoning_effort,
            "guided_json": response_model.model_json_schema(),
        }
        headers = {
            "Authorization": (
                f"Bearer {self._settings.nvidia_api_key.get_secret_value()}"
            ),
            "Content-Type": "application/json",
        }

        read_timeout = self._settings.nemotron_read_timeout_seconds(purpose)
        timeout = httpx.Timeout(
            connect=30.0,
            read=read_timeout,
            write=120.0,
            pool=60.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=request_body)
            if response.status_code == 202:
                request_id = self._extract_request_id(response)
                return await self._poll_status(
                    client=client,
                    request_id=request_id,
                    purpose=purpose,
                )
            if response.status_code == 200:
                return self._safe_response_json(response)
            if response.status_code == 429 or response.status_code >= 500:
                raise RetryableModelClientError(
                    f"HTTP {response.status_code} from Nemotron."
                )
            raise ModelClientError(f"HTTP {response.status_code} from Nemotron.")

    async def _poll_status(
        self,
        *,
        client: httpx.AsyncClient,
        request_id: str,
        purpose: str,
    ) -> dict:
        status_url = (
            f"{self._settings.nemotron_base_url.rstrip('/')}/status/{request_id}"
        )
        deadline = perf_counter() + self._settings.nemotron_poll_max_seconds_for(purpose)
        max_attempts = max(1, self._settings.nemotron_poll_attempts)
        interval = self._settings.nemotron_poll_interval_seconds

        for attempt in range(max_attempts):
            if perf_counter() >= deadline:
                break
            if interval > 0 and attempt > 0:
                await asyncio.sleep(interval)

            response = await client.get(status_url)
            if response.status_code == 202:
                continue
            if response.status_code != 200:
                raise ModelClientError(
                    f"HTTP {response.status_code} from Nemotron status."
                )

            payload = self._safe_response_json(response)
            status = str(payload.get("status", "")).lower()
            if status in {"failed", "error", "cancelled"}:
                raise ModelClientError("Nemotron async request failed.")
            if status in {"", "fulfilled", "completed", "succeeded", "success"}:
                try:
                    self._extract_content(payload)
                except ModelClientError:
                    continue
                return payload

        raise RetryableModelClientError("Nemotron request stayed pending.")

    def _parse_completion_payload(
        self,
        *,
        payload: dict,
        purpose: str,
        model: str,
        response_model: type[OutputT],
        started: float,
    ) -> ModelCallResult[OutputT]:
        raw_content = self._extract_content(payload)
        try:
            if isinstance(raw_content, str):
                parsed = _parse_json_text(raw_content)
            else:
                parsed = raw_content
        except json.JSONDecodeError as exc:
            message = (
                "Nemotron returned truncated JSON."
                if _json_decode_looks_truncated(exc)
                else "Nemotron returned invalid JSON."
            )
            raise RetryableModelClientError(message) from exc

        try:
            output = response_model.model_validate(parsed)
        except ValidationError as exc:
            raise RetryableModelClientError(
                "Nemotron response failed schema validation."
            ) from exc

        return ModelCallResult(
            output=output,
            model=model,
            purpose=purpose,
            mode="live",
            latency_ms=_elapsed_ms(started),
        )

    async def _idea_specific_partial(
        self,
        *,
        purpose: str,
        model: str,
        prompt: str,
        response_model: type[OutputT],
        reason: str,
        started: float | None = None,
    ) -> ModelCallResult[OutputT]:
        if not self._settings.allow_idea_aware_partial:
            raise ModelClientError(
                f"Live Nemotron is required for full project generation ({reason}). "
                "Set NEMOTRON_API_KEY/NVIDIA_API_KEY or enable degraded mode explicitly."
            )

        from agent.idea_aware_partial import IdeaAwarePartialClient

        partial_reason = reason
        if self._settings.nemotron_fast_fallback_active:
            partial_reason = (
                f"{reason} Using explicit degraded project output after a short live Nemotron attempt."
            )

        partial_client = IdeaAwarePartialClient(partial_reason=partial_reason)
        result = await partial_client.complete_structured(
            purpose=purpose,
            model=model,
            prompt=prompt,
            response_model=response_model,
        )
        if started is None:
            return result
        return replace(result, latency_ms=_elapsed_ms(started))

    @staticmethod
    def _safe_response_json(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ModelClientError("Nemotron returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ModelClientError("Nemotron returned invalid JSON.")
        return payload

    @staticmethod
    def _extract_request_id(response: httpx.Response) -> str:
        payload = NemotronModelClient._safe_response_json(response)
        request_id = payload.get("requestId")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ModelClientError("Nemotron accepted request without requestId.")
        return request_id

    @staticmethod
    def _extract_content(payload: dict) -> str | dict | list:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    for key in ("reasoning_content", "reasoning"):
                        alternate = message.get(key)
                        if isinstance(alternate, str) and alternate.strip():
                            return alternate
                if "text" in first_choice:
                    return first_choice["text"]

        for key in ("response", "result", "output"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return NemotronModelClient._extract_content(nested)
            if isinstance(nested, (str, list)):
                return nested

        if "content" in payload:
            return payload["content"]

        raise ModelClientError("Nemotron response missing completion content.")


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))


def _trace(
    mode: ModelMode,
    fallback_reason: str | None,
    entries: list[str],
) -> list[str]:
    if mode in {"degraded", "partial"}:
        prefix = f"Degraded mode: {fallback_reason or 'live model unavailable; project-specific scaffold used'}"
        return [prefix, *entries]
    return entries


def _extract_idea(prompt: str) -> str:
    marker = "Idea:\n"
    if marker not in prompt:
        return _clean_idea(prompt)

    value = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    return _clean_idea(value)


def _clean_idea(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    return cleaned or "the submitted project idea"


def _idea_label(idea: str) -> str:
    label = idea.rstrip(".")
    return label if len(label) <= 90 else f"{label[:87].rstrip()}..."


def _idea_title(idea: str) -> str:
    label = _idea_label(idea)
    lowered = label.lower()
    for prefix in ("build a ", "build an ", "create a ", "make a "):
        if lowered.startswith(prefix):
            label = label[len(prefix):]
            break
    return label[:1].upper() + label[1:] if label else "Submitted Project"


def _deterministic_payload(
    purpose: str,
    mode: ModelMode,
    fallback_reason: str | None,
    prompt: str,
) -> dict:
    idea = _extract_idea(prompt)
    if purpose == "recommend_stack":
        return _stack_recommendation_payload(mode, fallback_reason, idea, prompt)
    if purpose in {"plan_repo", "architecture_plan"}:
        return _repo_plan_payload(mode, fallback_reason, idea, prompt)
    if purpose == "file_manifest":
        return _file_manifest_payload(mode, fallback_reason, idea, prompt)
    if purpose == "final_readme":
        return _final_readme_payload(mode, fallback_reason, idea, prompt)
    if purpose in {"demo_script", "walkthrough"}:
        return _demo_script_payload(mode, fallback_reason, idea, prompt)
    if purpose == "pitch":
        return _pitch_payload(mode, fallback_reason, idea, prompt)

    payloads = {
        "scope_mvp": _scope_payload,
        "requirement_expansion": _scope_payload,
        "blocker_analysis": _blocker_analysis_payload,
    }
    try:
        return payloads[purpose](mode, fallback_reason, idea, prompt)
    except KeyError as exc:
        raise ModelClientError(f"Unsupported model purpose: {purpose}.") from exc


def _scope_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str = "",
) -> dict:
    from agent.idea_context import extract_json_section, features_from_context, target_users_from_context

    label = _idea_label(idea)
    intake = extract_json_section(prompt, "Frontend intake:\n")
    features = features_from_context(idea=idea, intake=intake)
    from agent.project_depth import enrich_project_requirements

    target = target_users_from_context(intake) or "users described in the submitted project"
    requirements = enrich_project_requirements(
        {
            "target_users": target,
            "user_personas": [target, "Admin or operator"],
            "core_features": features,
            "advanced_features": [
                "Authenticated workspace",
                "Operational dashboard",
                "Generated project evidence log",
            ],
            "success_criteria": [
                "The primary user can complete the end-to-end workflow.",
                "The codebase includes frontend, backend, database schema, tests, and docs.",
            ],
            "project_depth": intake.get("projectDepth") or "Advanced Project",
            "target_platform": intake.get("targetPlatform") or "web app",
            "mode": mode,
            "decision_trace": _trace(
                mode,
                fallback_reason,
                [
                    f"Grounded requirements in the submitted idea: {label}.",
                    "Expanded the idea into a multi-feature project brief.",
                    "Kept fallback behavior explicit as degraded mode when live generation is unavailable.",
                ],
            ),
        },
        idea=idea,
        intake=intake,
    )
    requirements["mode"] = mode
    requirements["decision_trace"] = _trace(
        mode,
        fallback_reason,
        [
            f"Grounded requirements in the submitted idea: {label}.",
            "Expanded the idea into a multi-feature project brief.",
            "Kept fallback behavior explicit as degraded mode when live generation is unavailable.",
        ],
    )
    return MvpScopeOutput.model_validate(requirements).model_dump()


def _stack_recommendation_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    from agent.idea_context import extract_json_section
    from agent.stack_recommendation import recommend_stack_heuristic

    intake = extract_json_section(prompt, "Frontend intake:\n")
    requirements = extract_json_section(prompt, "Project requirements:\n")
    build_context = extract_json_section(prompt, "Structured build context:\n")
    payload = recommend_stack_heuristic(
        idea=idea,
        project_requirements=requirements,
        build_context=build_context if isinstance(build_context, dict) else {},
    )
    payload["mode"] = mode
    payload["decision_trace"] = _trace(
        mode,
        fallback_reason,
        [
            f"Stack Selector Agent grounded stack in: {_idea_label(idea)}.",
            "Did not copy MVPilot host platform defaults.",
            "Aligned stack with idea, depth, platform, and RAG hints.",
        ],
    )
    return RecommendedStackOutput.model_validate(payload).model_dump()


def _repo_plan_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    from agent.idea_context import extract_json_section
    from agent.stack_recommendation import align_architecture_plan_with_recommended_stack

    label = _idea_label(idea)
    title = _project_title(idea, prompt)
    resolved_stack = _extract_resolved_stack_summary(prompt)
    warning_summary = _source_warning_summary(prompt)
    build_context = extract_json_section(prompt, "Structured build context:\n")
    recommended = (
        build_context.get("recommendedStack")
        if isinstance(build_context, dict)
        else None
    )
    files = [
        "README.md",
        "package.json",
        "index.html",
        "src/App.jsx",
        "src/main.jsx",
        "src/lib/api.js",
        "src/state/projectState.js",
        "src/styles.css",
        "backend/main.py",
        "backend/models.py",
        "backend/services.py",
        "backend/db.py",
        "requirements.txt",
        "tests/test_backend.py",
        "docs/PROJECT_PLAN.md",
        "docs/ARCHITECTURE.md",
        "docs/API_SPEC.md",
        "docs/DATABASE_SCHEMA.sql",
        "docs/TESTING_STRATEGY.md",
        "docs/DEPLOY.md",
        "docs/AGENT_LOG.md",
        "docs/BUILD_LOG.md",
        "docs/KNOWN_LIMITATIONS.md",
        "docs/WALKTHROUGH.md",
        ".env.example",
    ]
    plan_payload = RepoPlanOutput(
        files=files,
        file_tree=files,
        selected_stack=[item.strip() for item in resolved_stack.split(",") if item.strip()],
        architecture_overview=[
            f"Use submitted title as project identity: {title}.",
            f"Use resolvedTechStack for generated files, tests, and architecture: {resolved_stack}.",
            "Generate a complete full-stack project with source, tests, docs, database schema, and deployment notes.",
        ],
        frontend_architecture=[
            "React workspace with authenticated project flow, generation surface, review queue, and dashboard.",
            "API client module isolates fetch calls from UI state.",
        ],
        backend_architecture=[
            "FastAPI app with auth, upload, generation, review, dashboard, and health routes.",
            "Service layer owns domain logic and provider adapters.",
        ],
        data_model=[
            "users",
            "project_assets",
            "generated_items",
            "activity_logs",
        ],
        api_design=[
            "POST /api/auth/login",
            "POST /api/uploads",
            "POST /api/quizzes",
            "GET /api/flashcards/review",
            "GET /api/dashboard",
        ],
        auth_design=[
            "Development login route for local testing.",
            "Production-ready auth replacement documented for Supabase/Auth.js/Clerk.",
        ],
        database_schema=[
            "Postgres tables for users, assets, generated items, and activity logs.",
        ],
        state_management=[
            "React local state for current session, generated artifacts, review queue, and dashboard metrics.",
        ],
        integration_points=[
            "AI provider hooks in backend services.",
            "Supabase/Postgres persistence boundary in backend db adapter.",
        ],
        implementation_steps=[
            "Create the product-specific frontend workspace.",
            "Generate backend/API routes that serve the full project data flow.",
            "Add auth, upload, generation, dashboard, and review workflows.",
            "Document architecture, setup, env vars, testing, deployment, and limitations.",
            "Commit generated files through the GitHub Agent.",
        ],
        agent_assignments=[
            "Product Strategist Agent: requirements, personas, features, success criteria",
            "Research/RAG Agent: source URLs, uploaded docs, memory, and context",
            "System Architect Agent: architecture, stack, file tree, integration points",
            "Data/API Agent: models, API routes, validation, auth, storage",
            "Frontend Agent: screens, components, state, user experience",
            "Backend Agent: routes, services, persistence, provider boundaries",
            "QA Agent: validation, test plan, build verification",
            "Documentation Agent: README, setup, env, architecture, usage",
            "GitHub Agent: repo create/update and commit",
            "Logger Agent: live progress and stored logs",
        ],
        github_actions_needed=[
            "create_new_repo or use_existing_repo from frontend intake",
            "commit generated project files",
            "return repoUrl, commitUrl, branch, filesUploaded, and validation status",
        ],
        generated_artifacts=files,
        security_constraints=[
            "Never commit .env files or real secrets.",
            "Only include placeholder values in .env.example.",
            "Keep GitHub and Supabase service credentials server-side.",
        ],
        test_plan=["service unit tests", "API integration tests", "frontend build check", "repository health validation"],
        deployment_plan=[
            "Deploy frontend as a static Vite app.",
            "Deploy backend as a Python web service.",
            "Run Postgres/Supabase schema and set backend-only secrets.",
        ],
        documentation_plan=[
            "README with setup and usage",
            "Architecture, API, database, testing, deployment, limitations, and walkthrough docs",
        ],
        mode=mode,
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Selected a complete generated project package for the submitted idea: {label}.",
                "Planned around the RAG build context before any GitHub action.",
                warning_summary or "No submitted source warnings were present.",
                "Kept generated files secret-safe, testable, and deployable.",
            ],
        ),
    ).model_dump()
    return align_architecture_plan_with_recommended_stack(
        plan_payload,
        recommended if isinstance(recommended, dict) else None,
    )


def _extract_resolved_stack_summary(prompt: str) -> str:
    marker = "Resolved tech stack:\n"
    if marker not in prompt:
        return "the resolved project stack from build context"

    raw_block = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    try:
        payload = json.loads(raw_block)
    except json.JSONDecodeError:
        return "the resolved project stack from build context"

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return "the resolved project stack from build context"

    stack_items = [item for item in items if isinstance(item, str) and item.strip()]
    if not stack_items:
        return "the resolved project stack from build context"

    return ", ".join(stack_items)


def _extract_json_section(prompt: str, marker: str) -> dict:
    if marker not in prompt:
        return {}

    raw_block = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    try:
        payload = json.loads(raw_block)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_frontend_intake(prompt: str) -> dict:
    return _extract_json_section(prompt, "Frontend intake:\n")


def _extract_source_context(prompt: str) -> dict:
    return _extract_json_section(prompt, "Source context:\n")


def _project_title(idea: str, prompt: str) -> str:
    intake = _extract_frontend_intake(prompt)
    title = intake.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return _idea_title(idea)


def _source_warning_summary(prompt: str) -> str:
    warnings = _source_warnings(prompt)
    if not warnings:
        return ""

    messages = []
    for warning in warnings:
        source = warning.get("source", "source")
        message = warning.get("message", "unreadable source")
        messages.append(f"{source}: {message}")

    if not messages:
        return ""
    return "Source warnings: " + "; ".join(messages)


def _source_warnings(prompt: str) -> list[dict[str, str]]:
    source_context = _extract_source_context(prompt)
    warnings = source_context.get("warnings")
    if not isinstance(warnings, list) or not warnings:
        return []

    parsed: list[dict[str, str]] = []
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        source = warning.get("source", "source")
        message = warning.get("message", "unreadable source")
        parsed.append({"source": str(source), "message": str(message)})
    return parsed


def _file_manifest_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    title = _project_title(idea, prompt)
    resolved_stack = _extract_resolved_stack_summary(prompt)
    warning_summary = _source_warning_summary(prompt)
    source_warnings = _source_warnings(prompt)
    architecture_plan = _extract_json_section(prompt, "Architecture plan:\n") or _extract_json_section(prompt, "Repo plan:\n")
    project_requirements = _extract_json_section(prompt, "Project requirements:\n") or _extract_json_section(prompt, "MVP scope:\n")
    artifacts = build_project_artifacts(
        idea=idea,
        title=title,
        resolved_stack=resolved_stack,
        architecture_plan=architecture_plan,
        source_warnings=source_warnings,
        project_requirements=project_requirements,
    )
    return FileManifestOutput(
        artifacts=artifacts,
        mode=mode,
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Mapped {title} into complete frontend, backend, database, tests, docs, deployment, and walkthrough artifacts.",
                f"Used resolved stack summary: {resolved_stack}.",
                warning_summary or "No submitted source warnings were present.",
                "Kept artifact content secret-safe, testable, and deployment-ready.",
            ],
        ),
    ).model_dump()


def _blocker_analysis_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str = "",
) -> dict:
    del prompt
    label = _idea_label(idea)
    return BlockerAnalysisOutput(
        blocker_type="repo_health_recovery",
        severity="medium",
        recoverable=True,
        root_cause=(
            "Repository health verification found a missing or incomplete generated artifact "
            "after the full project package was committed."
        ),
        recovery_plan=[
            "Classify the build result as recoverable.",
            "Regenerate or patch only the missing project-specific artifact.",
            "Run build verification again before final packaging.",
        ],
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Kept blocker analysis tied to the submitted idea: {label}.",
                "Read the build tool result before applying recovery.",
                "Chose a minimal recovery because the repo plan is otherwise sound.",
                "Kept the blocker visible as proof of autonomous validation and recovery.",
            ],
        ),
    ).model_dump()


def _final_readme_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    title = _project_title(idea, prompt)
    resolved_stack = _extract_resolved_stack_summary(prompt)
    warning_summary = _source_warning_summary(prompt)
    warning_note = f"\n\n{warning_summary}" if warning_summary else ""
    return FinalReadmeOutput(
        title=title,
        content=(
            f"# {title}\n\n"
            f"MVPilot turned this submitted idea into a complete generated software project: {idea}\n\n"
            f"Resolved stack: {resolved_stack}.\n\n"
            "The package includes expanded requirements, architecture, source files, API/data plans, "
            "tests, setup instructions, deployment guidance, validation evidence, and a final project report."
            f"{warning_note}"
        ),
        setup_steps=[
            "Install dependencies.",
            "Run the FastAPI backend.",
            "Submit the idea through the dashboard.",
        ],
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Summarized the workflow around {title}.",
                f"Included resolved stack summary: {resolved_stack}.",
                "Included setup, testing, and deployment steps for a complete project handoff.",
            ],
        ),
    ).model_dump()


def _demo_script_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    title = _project_title(idea, prompt)
    warning_summary = _source_warning_summary(prompt)
    warning_note = f" Mention source warnings: {warning_summary}." if warning_summary else ""
    return DemoScriptOutput(
        title=f"Three-minute {title} walkthrough",
        content=(
            f"Open with the submitted idea: {idea}. "
            "Show MVPilot retrieving context, expanding requirements, designing architecture, "
            "generating project files, validating the codebase, committing to GitHub, and ending "
            "with README, architecture, testing, deployment, and report content."
            f"{warning_note}"
        ),
        beats=[
            "Enter the project idea.",
            "Watch the graph trace fill with model-backed decisions.",
            "Show validation and any explicit degraded-mode integrations.",
            "Close on the generated repository and project report.",
        ],
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Built the script around observable {title} dashboard moments.",
                "Kept validation and delivery evidence visible instead of hiding generation work.",
            ],
        ),
    ).model_dump()


def _pitch_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    label = _idea_label(idea)
    title = _project_title(idea, prompt)
    warning_summary = _source_warning_summary(prompt)
    content = (
        f"MVPilot helps teams move from idea to complete generated software projects. "
        f"For {title}, {label}, it expands requirements, designs the architecture, "
        "generates a full codebase, validates the output, and produces README, setup, "
        "testing, deployment, and project-report evidence."
    )
    if warning_summary:
        content = f"{content} {warning_summary}"
    return PitchOutput(
        title=title,
        tagline="A Nemotron-powered project studio that turns ideas into committed full-stack repositories.",
        content=content,
        proof_points=[
            "Stable FastAPI task contracts for frontend integration.",
            "Structured Nemotron-style reasoning on model-backed steps.",
            "Visible validation and recovery before final packaging.",
        ],
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Focused the {title} pitch on speed, traceability, and recovery.",
                "Used generated artifacts as the proof instead of broad claims.",
            ],
        ),
    ).model_dump()
