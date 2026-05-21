from __future__ import annotations

import json
import re
from typing import Any


def extract_idea_from_prompt(prompt: str) -> str:
    marker = "Idea:\n"
    if marker not in prompt:
        return _clean_text(prompt) or "the submitted project idea"
    value = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    return _clean_text(value) or "the submitted project idea"


def extract_json_section(prompt: str, marker: str) -> dict[str, Any]:
    if marker not in prompt:
        return {}
    raw = prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def project_title_from_context(
    *,
    idea: str,
    intake: dict[str, Any] | None = None,
) -> str:
    intake = intake or {}
    title = _clean_text(str(intake.get("title") or ""))
    if title:
        return title
    return title_from_idea(idea)


def title_from_idea(idea: str) -> str:
    label = idea.rstrip(".")
    lowered = label.lower()
    for prefix in ("build a ", "build an ", "create a ", "make a "):
        if lowered.startswith(prefix):
            label = label[len(prefix) :]
            break
    if len(label) > 72:
        label = f"{label[:69].rstrip()}..."
    return label[:1].upper() + label[1:] if label else "Submitted Project"


def features_from_context(
    *,
    idea: str,
    intake: dict[str, Any] | None = None,
    mvp_scope: dict[str, Any] | None = None,
    repo_plan: dict[str, Any] | None = None,
) -> list[str]:
    features: list[str] = []
    intake = intake or {}
    scope = mvp_scope or {}
    plan = repo_plan or {}

    for source in (
        intake.get("requiredFeatures") or intake.get("required_features") or [],
        scope.get("must_have") or [],
        plan.get("implementation_steps") or plan.get("implementationSteps") or [],
    ):
        if not isinstance(source, list):
            continue
        for item in source:
            label = _clean_text(str(item))
            if label and label not in features:
                features.append(label)

    if not features:
        features = [
            f"Core workflow for: {title_from_idea(idea)}",
            "Idea-specific UI and API surface",
            "Documented setup, architecture, testing, and deployment path",
        ]
    return features[:8]


def target_users_from_context(intake: dict[str, Any] | None) -> str | None:
    intake = intake or {}
    value = intake.get("targetUsers") or intake.get("target_users")
    return _clean_text(str(value)) if value else None


def tech_stack_from_context(
    intake: dict[str, Any] | None,
    repo_plan: dict[str, Any] | None = None,
) -> str | None:
    intake = intake or {}
    plan = repo_plan or {}
    preference = intake.get("techStackPreference") or intake.get("tech_stack_preference")
    if preference:
        return _clean_text(str(preference))
    stack = plan.get("selected_stack") or plan.get("selectedStack")
    if isinstance(stack, list):
        items = [str(item).strip() for item in stack if str(item).strip()]
        return ", ".join(items) if items else None
    return _clean_text(str(stack)) if stack else None


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
