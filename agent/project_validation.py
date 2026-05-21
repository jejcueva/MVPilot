from __future__ import annotations

import re
from typing import Any

from agent.idea_context import project_title_from_context
from agent.project_depth import PROJECT_ARCHETYPES, user_flow_checklist


GENERIC_MARKERS = (
    "generic todo",
    "basic todo",
    "hello world",
    "lorem ipsum",
    "placeholder app",
    "sample dashboard",
    "starter app",
    "baseline prototype",
)

GENERIC_FEATURE_PHRASES = (
    "capture intake details for the target workflow",
    "generate a prioritized action plan",
    "expose sample data through a small api surface",
)


def validate_project_output(
    *,
    idea: str,
    intake: dict[str, Any] | None,
    project_requirements: dict[str, Any] | None,
    architecture_plan: dict[str, Any] | None,
    generated_artifacts: list[dict[str, Any]],
    model_modes: list[str],
    require_live_manifest: bool = False,
    manifest_model_mode: str | None = None,
) -> dict[str, Any]:
    intake = intake or {}
    requirements = project_requirements or {}
    plan = architecture_plan or {}
    title = project_title_from_context(idea=idea, intake=intake)
    idea_tokens = _idea_tokens(idea)
    title_tokens = _idea_tokens(title)

    readme = _artifact_content(generated_artifacts, "README.md")
    app_source = _artifact_content(generated_artifacts, "src/App.jsx")
    architecture = _artifact_content(generated_artifacts, "docs/ARCHITECTURE.md")
    api_spec = _artifact_content(generated_artifacts, "docs/API_SPEC.md")
    schema = _artifact_content(generated_artifacts, "docs/DATABASE_SCHEMA.sql")
    backend_main = _artifact_content(generated_artifacts, "backend/main.py")
    backend_services = _artifact_content(generated_artifacts, "backend/services.py")
    tests = _artifact_content(generated_artifacts, "tests/test_backend.py")
    deploy = _artifact_content(generated_artifacts, "docs/DEPLOY.md")
    agent_log = _artifact_content(generated_artifacts, "docs/AGENT_LOG.md") + _artifact_content(
        generated_artifacts, "docs/BUILD_LOG.md"
    )

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    features = _string_list(requirements.get("core_features") or requirements.get("must_have"))
    advanced = _string_list(requirements.get("advanced_features"))
    api_routes = _string_list(requirements.get("api_routes"))
    high_depth = str(requirements.get("project_depth") or "").lower() in {
        "advanced project",
        "production-style project",
        "hackathon-winning project",
    }

    add_check(
        "title_matches_idea",
        _title_matches_idea(title, idea, idea_tokens, title_tokens),
        f"Expected project title '{title}' to reflect the submitted idea.",
    )
    add_check(
        "readme_specific",
        bool(readme) and _text_reflects_idea(readme, idea_tokens) and not _looks_generic(readme),
        "README should mention the idea and describe a complete generated project.",
    )
    add_check(
        "ui_specific",
        bool(app_source) and _text_reflects_idea(app_source, idea_tokens) and not _looks_generic(app_source),
        "Frontend should reference the idea and expose real product flows.",
    )
    add_check(
        "requirements_expanded",
        len(features) >= (7 if high_depth else 4),
        "Project requirements should include enough core features for the requested depth.",
    )
    add_check(
        "advanced_features_present",
        bool(advanced) or not high_depth,
        "Advanced or higher depth should include advanced features.",
    )
    add_check(
        "no_generic_fallback_features",
        not _uses_generic_feature_set(features) and not _looks_generic(readme + app_source),
        "Output must not use legacy generic starter or fallback feature language.",
    )
    add_check(
        "architecture_documents_full_system",
        bool(architecture)
        and all(term in architecture.lower() for term in ("frontend", "backend", "data", "auth"))
        and (bool(plan.get("implementation_steps")) or bool(plan.get("files")) or bool(plan.get("file_tree"))),
        "Architecture doc should cover frontend, backend, data, auth, and the plan.",
    )
    add_check(
        "api_and_database_planned",
        bool(api_spec)
        and bool(schema)
        and ("create table" in schema.lower())
        and (bool(api_routes) or "/api/" in api_spec),
        "Generated project should include API route and database schema plans.",
    )
    add_check(
        "auth_data_flow_present",
        ("/api/auth/login" in backend_main and "users" in schema.lower()) if high_depth else True,
        "Advanced or higher depth should include authentication and user-backed data flow.",
    )
    add_check(
        "implementation_files_complete",
        _has_file(generated_artifacts, "backend/models.py")
        and _has_file(generated_artifacts, "backend/services.py")
        and _has_file(generated_artifacts, "src/lib/api.js")
        and _has_file(generated_artifacts, "src/state/projectState.js"),
        "Generated file tree should include frontend API/state and backend model/service layers.",
    )
    add_check(
        "testing_and_deployment_present",
        bool(tests) and bool(deploy),
        "Generated project should include tests and deployment instructions.",
    )
    add_check(
        "degraded_mode_explicit",
        "degraded" not in model_modes or "degraded" in agent_log.lower() or "degraded" in readme.lower(),
        "Any degraded mode must be explicit in logs or documentation.",
    )
    if require_live_manifest:
        manifest_mode = manifest_model_mode or (model_modes[-1] if model_modes else "")
        add_check(
            "live_manifest_only",
            manifest_mode == "live",
            "Live workflow requires Nemotron live file_manifest output.",
        )
    add_check(
        "user_flow_defined",
        _user_flow_defined(requirements),
        "Project requirements should include an end-to-end user flow.",
    )
    add_check(
        "project_archetype_selected",
        str(requirements.get("project_archetype") or "") in PROJECT_ARCHETYPES,
        "Requirements should select a project archetype.",
    )

    critical = {
        "title_matches_idea",
        "readme_specific",
        "ui_specific",
        "requirements_expanded",
        "no_generic_fallback_features",
        "architecture_documents_full_system",
        "api_and_database_planned",
        "implementation_files_complete",
        "testing_and_deployment_present",
        "user_flow_defined",
    }
    if require_live_manifest:
        critical.add("live_manifest_only")
    passed = all(item["passed"] for item in checks if item["name"] in critical)
    warnings = [item["detail"] for item in checks if not item["passed"]]
    return {
        "passed": passed,
        "project_title": title,
        "checks": checks,
        "user_flows": requirements.get("user_flows") or [],
        "project_archetype": requirements.get("project_archetype"),
        "project_depth": requirements.get("project_depth"),
        "warnings": warnings,
    }


def build_project_delivery_report(
    *,
    idea: str,
    intake: dict[str, Any] | None,
    project_requirements: dict[str, Any] | None,
    validation: dict[str, Any],
    model_modes: list[str],
    generated_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    del intake
    requirements = project_requirements or {}
    features = _string_list(requirements.get("core_features"))
    advanced = _string_list(requirements.get("advanced_features"))
    pending = []
    if "degraded" in model_modes or "partial" in model_modes:
        pending.append("Live Nemotron file content used explicit degraded mode")
    if not validation.get("passed"):
        pending.extend(validation.get("warnings") or [])

    return {
        "idea": idea,
        "project_title": validation.get("project_title"),
        "project_depth": requirements.get("project_depth"),
        "project_archetype": requirements.get("project_archetype"),
        "user_flow_checklist": user_flow_checklist(requirements.get("user_flows") or []),
        "model_modes": model_modes,
        "completed_features": features,
        "advanced_features": advanced,
        "degraded_features": [
            "External provider integrations require configured credentials",
            "Production auth provider replacement is documented",
        ]
        if "degraded" in model_modes or "partial" in model_modes
        else [],
        "pending_features": pending,
        "artifact_count": len(generated_artifacts),
        "validation_passed": validation.get("passed", False),
        "validation_checks": validation.get("checks", []),
    }


def _artifact_content(artifacts: list[dict[str, Any]], name: str) -> str:
    for artifact in reversed(artifacts):
        if str(artifact.get("name")) == name:
            return str(artifact.get("content") or "")
    return ""


def _has_file(artifacts: list[dict[str, Any]], name: str) -> bool:
    return any(str(artifact.get("name")) == name and bool(artifact.get("content")) for artifact in artifacts)


def _title_matches_idea(
    title: str,
    idea: str,
    idea_tokens: set[str],
    title_tokens: set[str],
) -> bool:
    if _text_reflects_idea(title, idea_tokens) or _text_reflects_idea(title, title_tokens):
        return True
    compact_title = re.sub(r"[^a-z0-9]", "", title.lower())
    compact_idea = re.sub(r"[^a-z0-9]", "", idea.lower())
    return bool(compact_title) and compact_title in compact_idea


def _idea_tokens(idea: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{4,}", idea.lower()) if token not in STOP_WORDS}


def _text_reflects_idea(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return True
    lowered = text.lower()
    hits = sum(1 for token in tokens if token in lowered)
    return hits >= min(2, len(tokens))


def _looks_generic(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in GENERIC_MARKERS)


def _uses_generic_feature_set(features: list[Any]) -> bool:
    normalized = [str(item).strip().lower() for item in features if str(item).strip()]
    if len(normalized) < 2:
        return False
    hits = sum(1 for phrase in GENERIC_FEATURE_PHRASES if any(phrase in item for item in normalized))
    return hits >= 2


def _user_flow_defined(requirements: dict[str, Any]) -> bool:
    flows = requirements.get("user_flows")
    return isinstance(flows, list) and len(flows) >= 2


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


STOP_WORDS = {
    "build",
    "create",
    "make",
    "that",
    "with",
    "from",
    "into",
    "help",
    "helps",
    "this",
    "your",
    "their",
    "platform",
    "project",
    "software",
    "system",
    "agent",
    "and",
    "the",
    "for",
}
