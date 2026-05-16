from __future__ import annotations

import re
from typing import Any

from agent.idea_context import project_title_from_context, title_from_idea


GENERIC_MARKERS = (
    "generic todo",
    "basic todo",
    "starter app",
    "hello world",
    "lorem ipsum",
    "placeholder app",
    "sample dashboard",
    "mvpilot demo ready",
)

GENERIC_FEATURE_PHRASES = (
    "capture intake details for the target workflow",
    "generate a prioritized mvp action plan",
    "expose demo data through a small fastapi surface",
)


def validate_mvp_output(
    *,
    idea: str,
    intake: dict[str, Any] | None,
    mvp_scope: dict[str, Any] | None,
    repo_plan: dict[str, Any] | None,
    generated_artifacts: list[dict[str, Any]],
    model_modes: list[str],
) -> dict[str, Any]:
    intake = intake or {}
    scope = mvp_scope or {}
    plan = repo_plan or {}
    title = project_title_from_context(idea=idea, intake=intake)
    idea_tokens = _idea_tokens(idea)
    title_tokens = _idea_tokens(title)

    readme = _artifact_content(generated_artifacts, "README.md")
    app_source = _artifact_content(generated_artifacts, "src/App.jsx") or _artifact_content(
        generated_artifacts, "src/app.py"
    )
    architecture = _artifact_content(generated_artifacts, "docs/ARCHITECTURE.md")

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    add_check(
        "title_matches_idea",
        _title_matches_idea(title, idea, idea_tokens, title_tokens),
        f"Expected project title '{title}' to reflect the submitted idea.",
    )
    add_check(
        "readme_specific",
        bool(readme) and _text_reflects_idea(readme, idea_tokens) and not _looks_generic(readme),
        "README should mention the idea and avoid generic starter copy.",
    )
    add_check(
        "ui_specific",
        bool(app_source) and _text_reflects_idea(app_source, idea_tokens) and not _looks_generic(app_source),
        "Frontend entry should reference the idea, not a generic dashboard.",
    )
    add_check(
        "features_aligned",
        _features_aligned(idea, scope.get("must_have") or [], intake.get("requiredFeatures") or []),
        "Scoped features should relate to the user's concept.",
    )
    add_check(
        "no_generic_fallback_features",
        not _uses_generic_feature_set(scope.get("must_have") or []),
        "Scope must not use the legacy generic MVP feature trio.",
    )
    add_check(
        "architecture_documents_plan",
        bool(architecture) and (bool(plan.get("implementation_steps")) or bool(plan.get("files"))),
        "Architecture doc should exist alongside a structured repo plan.",
    )
    add_check(
        "enriched_pipeline_used",
        "fallback" not in model_modes
        and (
            "live" in model_modes
            or "partial" in model_modes
            or "mock" in model_modes
        ),
        f"Model modes recorded: {', '.join(model_modes) or 'unknown'}. "
        "Legacy generic fallback must not be used.",
    )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "project_title": title,
        "checks": checks,
        "warnings": [] if passed else [item["detail"] for item in checks if not item["passed"]],
    }


def _artifact_content(artifacts: list[dict[str, Any]], name: str) -> str:
    for artifact in reversed(artifacts):
        if str(artifact.get("name")) == name:
            return str(artifact.get("content") or "")
    return ""


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


def _features_aligned(idea: str, scope_features: list[Any], intake_features: list[Any]) -> bool:
    combined = [str(item) for item in [*scope_features, *intake_features] if str(item).strip()]
    if not combined:
        return _text_reflects_idea(idea, _idea_tokens(idea))
    idea_tokens = _idea_tokens(idea)
    for feature in combined:
        if _text_reflects_idea(feature, idea_tokens):
            return True
    return len(combined) >= 2


def build_delivery_report(
    *,
    idea: str,
    intake: dict[str, Any] | None,
    mvp_scope: dict[str, Any] | None,
    validation: dict[str, Any],
    model_modes: list[str],
    generated_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    intake = intake or {}
    scope = mvp_scope or {}
    features = [str(item) for item in (scope.get("must_have") or []) if str(item).strip()]
    completed = features[:]
    mocked = [
        "External API integrations (labeled mock in generated code)",
        "Production auth/billing (requires API keys)",
    ]
    pending = []
    if "live" not in model_modes:
        pending.append("Live Nemotron file content (partial or scaffold path was used)")
    if not validation.get("passed"):
        pending.extend(validation.get("warnings") or [])

    return {
        "idea": idea,
        "project_title": validation.get("project_title"),
        "model_modes": model_modes,
        "completed_features": completed,
        "mocked_features": mocked,
        "pending_features": pending,
        "artifact_count": len(generated_artifacts),
        "validation_passed": validation.get("passed", False),
        "validation_checks": validation.get("checks", []),
    }


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
    "for",
    "and",
    "the",
    "agent",
}
