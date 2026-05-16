from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal, Protocol, TypeVar

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
    RepoPlanOutput,
)


OutputT = TypeVar("OutputT", bound=BaseModel)
ModelMode = Literal["mock", "live", "partial"]


class ModelClientError(Exception):
    def __init__(self, safe_message: str) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message


class RetryableModelClientError(ModelClientError):
    pass


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
        if not self._settings.nvidia_configured:
            return await self._idea_specific_partial(
                purpose=purpose,
                model=model,
                prompt=prompt,
                response_model=response_model,
                reason="Missing NVIDIA_API_KEY for live mode.",
            )

        started = perf_counter()
        attempts = max(0, self._settings.nemotron_max_retries) + 1
        last_reason = "Nemotron request failed."

        for attempt in range(attempts):
            try:
                payload = await self._send_completion_request(
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
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
                if attempt + 1 < attempts:
                    continue
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=last_reason,
                    started=started,
                )
            except ModelClientError as exc:
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=exc.safe_message,
                    started=started,
                )
            except httpx.TimeoutException:
                last_reason = "Nemotron request timed out."
                if attempt + 1 < attempts:
                    continue
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=last_reason,
                    started=started,
                )
            except httpx.HTTPError:
                last_reason = "Nemotron request failed."
                if attempt + 1 < attempts:
                    continue
                return await self._idea_specific_partial(
                    purpose=purpose,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    reason=last_reason,
                    started=started,
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
        model: str,
        prompt: str,
        response_model: type[OutputT],
        max_tokens: int,
        reasoning_effort: str,
    ) -> dict:
        url = f"{self._settings.nemotron_base_url.rstrip('/')}/chat/completions"
        request_body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are MVPilot's structured planning model. Return only "
                        "JSON that matches the guided schema."
                    ),
                },
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

        timeout = httpx.Timeout(self._settings.nemotron_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=request_body)
            if response.status_code == 202:
                request_id = self._extract_request_id(response)
                return await self._poll_status(client=client, request_id=request_id)
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
    ) -> dict:
        status_url = (
            f"{self._settings.nemotron_base_url.rstrip('/')}/status/{request_id}"
        )
        attempts = max(1, self._settings.nemotron_poll_attempts)

        for _ in range(attempts):
            if self._settings.nemotron_poll_interval_seconds > 0:
                await asyncio.sleep(self._settings.nemotron_poll_interval_seconds)

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

        raise ModelClientError("Nemotron request stayed pending.")

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
            parsed = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
        except json.JSONDecodeError as exc:
            raise ModelClientError("Nemotron returned invalid JSON.") from exc

        try:
            output = response_model.model_validate(parsed)
        except ValidationError as exc:
            raise ModelClientError(
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
                f"Live Nemotron is required for MVP generation ({reason}). "
                "Set NVIDIA_API_KEY or enable ALLOW_IDEA_AWARE_PARTIAL for graceful degradation."
            )

        from agent.idea_aware_partial import IdeaAwarePartialClient

        partial_client = IdeaAwarePartialClient(partial_reason=reason)
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
                if isinstance(message, dict) and "content" in message:
                    return message["content"]
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
    if mode == "partial":
        prefix = f"Partial mode: {fallback_reason or 'live model unavailable; idea-specific scaffold used'}"
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
    return cleaned or "the submitted MVP idea"


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
    return label[:1].upper() + label[1:] if label else "Submitted MVP"


def _deterministic_payload(
    purpose: str,
    mode: ModelMode,
    fallback_reason: str | None,
    prompt: str,
) -> dict:
    idea = _extract_idea(prompt)
    if purpose == "plan_repo":
        return _repo_plan_payload(mode, fallback_reason, idea, prompt)
    if purpose == "file_manifest":
        return _file_manifest_payload(mode, fallback_reason, idea, prompt)
    if purpose == "final_readme":
        return _final_readme_payload(mode, fallback_reason, idea, prompt)
    if purpose == "demo_script":
        return _demo_script_payload(mode, fallback_reason, idea, prompt)
    if purpose == "pitch":
        return _pitch_payload(mode, fallback_reason, idea, prompt)

    payloads = {
        "scope_mvp": _scope_payload,
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
    target = target_users_from_context(intake) or "users described in the submitted MVP"
    return MvpScopeOutput(
        target_user=target,
        must_have=features,
        demo_boundary=(
            f"one idea-specific workflow for: {label}; use live integrations where configured "
            "and clearly labeled mocks where credentials are unavailable"
        ),
        mode=mode,
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Grounded the scope in the submitted idea: {label}.",
                "Kept the MVP to one visible workflow for a short hackathon run.",
                "Deferred broad integrations only when the idea-specific core path needed credentials or more time.",
            ],
        ),
    ).model_dump()


def _repo_plan_payload(
    mode: ModelMode,
    fallback_reason: str | None,
    idea: str,
    prompt: str,
) -> dict:
    label = _idea_label(idea)
    title = _project_title(idea, prompt)
    resolved_stack = _extract_resolved_stack_summary(prompt)
    warning_summary = _source_warning_summary(prompt)
    files = [
        "README.md",
        "package.json",
        "index.html",
        "src/App.jsx",
        "src/styles.css",
        "src/data/mockRecords.js",
        "backend/main.py",
        "backend/mvp_engine.py",
        "backend/requirements.txt",
        "tests/test_backend.py",
        "database/schema.sql",
        "docs/ARCHITECTURE.md",
        "docs/BUILD_LOG.md",
        "demo/demo_script.md",
        ".env.example",
    ]
    return RepoPlanOutput(
        files=files,
        required_files=files,
        selected_stack=[item.strip() for item in resolved_stack.split(",") if item.strip()],
        repo_structure=[
            "README.md",
            "src/ for the React MVP experience",
            "backend/ for FastAPI routes and idea-specific service logic",
            "database/ for schema notes and seedable tables",
            "tests/ for smoke tests",
            "docs/ for architecture, build logs, and validation notes",
            "demo/ for the walkthrough script",
        ],
        implementation_steps=[
            "Create the idea-specific frontend workflow.",
            "Generate backend/API routes that serve the scoped MVP state.",
            "Add labeled mock integrations when live credentials are unavailable.",
            "Document architecture, build evidence, validation, and setup.",
            "Commit generated files through the GitHub Agent.",
        ],
        agent_assignments=[
            "orchestrator: scope, plan, and validate the package",
            "rag: provide rules, stack, and safety evidence before planning",
            "github: create or update the repository and commit files",
            "black_box: store decisions, logs, artifacts, and final summary",
        ],
        github_actions_needed=[
            "create_new_repo or use_existing_repo from frontend intake",
            "commit generated project files",
            "return repoUrl, commitUrl, branch, filesUploaded, and errors",
        ],
        generated_artifacts=files,
        security_constraints=[
            "Never commit .env files or real secrets.",
            "Only include placeholder values in .env.example.",
            "Keep GitHub and Supabase service credentials server-side.",
        ],
        demo_requirements=[
            "Show the submitted idea becoming requirements, architecture, and files.",
            "Show RAG evidence before plan generation when source context exists.",
            "End with GitHub repo, commit links, and validation status.",
        ],
        test_plan=["unit workflow", "API integration", "repo health smoke check"],
        architecture_notes=[
            f"Use submitted title as project identity: {title}.",
            f"Use resolvedTechStack for generated files, tests, and architecture: {resolved_stack}.",
            "Keep model calls behind a client protocol.",
            "Keep GitHub and build actions behind mockable tool adapters.",
            warning_summary or "No submitted source warnings were present.",
        ],
        mode=mode,
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Selected a complete but small repo package for the submitted idea: {label}.",
                "Planned around the RAG build context before any GitHub action.",
                "Kept generated files secret-safe and repo-health checkable.",
            ],
        ),
    ).model_dump()


def _extract_resolved_stack_summary(prompt: str) -> str:
    marker = "Resolved tech stack:\n"
    if marker not in prompt:
        return "the resolved MVPilot stack from build context"

    raw_block = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    try:
        payload = json.loads(raw_block)
    except json.JSONDecodeError:
        return "the resolved MVPilot stack from build context"

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return "the resolved MVPilot stack from build context"

    stack_items = [item for item in items if isinstance(item, str) and item.strip()]
    if not stack_items:
        return "the resolved MVPilot stack from build context"

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
    repo_plan = _extract_json_section(prompt, "Repo plan:\n")
    artifacts = build_project_artifacts(
        idea=idea,
        title=title,
        resolved_stack=resolved_stack,
        repo_plan=repo_plan,
        source_warnings=source_warnings,
    )
    return FileManifestOutput(
        artifacts=artifacts,
        mode=mode,
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Mapped {title} into runnable frontend, backend, database, tests, docs, and walkthrough artifacts.",
                f"Used resolved stack summary: {resolved_stack}.",
                warning_summary or "No submitted source warnings were present.",
                "Kept artifact content compact, commit-safe, and health-checkable.",
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
            "after the idea-specific package was committed."
        ),
        recovery_plan=[
            "Classify the build result as recoverable.",
            "Regenerate or patch only the missing idea-specific artifact.",
            "Run build verification again before final packaging.",
        ],
        decision_trace=_trace(
            mode,
            fallback_reason,
            [
                f"Kept blocker analysis tied to the submitted idea: {label}.",
                "Read the build tool result before applying recovery.",
                "Chose a minimal recovery because the repo plan is otherwise sound.",
                "Kept the blocker visible as proof of autonomous error handling.",
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
            f"MVPilot turned this submitted idea into an idea-specific MVP package: {idea}\n\n"
            f"Resolved stack: {resolved_stack}.\n\n"
            "The package includes scoped requirements, generated artifacts, build "
            "verification, validation evidence, and a judge-ready final summary."
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
                "Included setup steps that work without live external services.",
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
            "Show MVPilot retrieving context, scoping the MVP, generating repo "
            "artifacts, validating idea-specific output, committing to GitHub, and ending "
            "with README, walkthrough, and pitch content."
            f"{warning_note}"
        ),
        beats=[
            "Enter the project idea.",
            "Watch the graph trace fill with model-backed decisions.",
            "Show validation and any labeled partial/mocked integrations.",
            "Close on the generated package and pitch.",
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
        f"MVPilot helps teams move from idea to credible MVP evidence fast. "
        f"For {title}, {label}, it scopes the workflow, plans "
        "the repo, generates idea-specific artifacts, validates the output, "
        "and produces the final README, walkthrough script, and pitch."
    )
    if warning_summary:
        content = f"{content} {warning_summary}"
    return PitchOutput(
        title=title,
        tagline="An AI teammate that turns messy hackathon ideas into committed MVP repositories.",
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
