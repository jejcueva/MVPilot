from __future__ import annotations

import json
from typing import Any


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _context_contract() -> str:
    return (
        "Frontend intake is the user's source of truth for project identity, "
        "idea, submitted URLs, uploaded files, GitHub connection markers, and final labels. "
        "Use frontendIntake, sourceContext, and resolvedTechStack as structured build context. "
        "Source material grounds the build. "
        "required RAG rules override user preference and MVPilot defaults. "
        "resolvedTechStack is binding for architecture/tests/files. "
        "Surface missing or unreadable sources as warnings, not silent omissions."
    )


def build_scope_mvp_prompt(
    *,
    idea: str,
    build_context: dict[str, Any],
    memory_matches: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> str:
    del retrieved_docs
    resolved_stack = build_context.get("resolvedTechStack", {})
    return (
        "Scope this hackathon idea into one MVP-ready build.\n"
        "Do NOT default to a generic todo app, starter dashboard, or unrelated template.\n"
        "Every feature must map to the submitted idea and intake fields.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(resolved_stack)}\n\n"
        f"Structured build context (highest priority — follow critical items first):\n"
        f"{_json_block(build_context)}\n\n"
        f"Memory matches:\n{_json_block(memory_matches)}\n\n"
        "Use resolvedTechStack as the stack decision. "
        "required stack items override MVPilot defaults. "
        "Honor required deliverables, allowed tools/APIs, repository format, walkthrough format, "
        "resolved tech stack, and scope warnings from build context. "
        "Return target user, must-have features, demo boundary, mode, and a short decision trace."
    )


def build_plan_repo_prompt(
    *,
    idea: str,
    mvp_scope: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    resolved_stack = build_context.get("resolvedTechStack", {})
    return (
        "Plan the smallest generated repository package for this MVP.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"MVP scope:\n{_json_block(mvp_scope)}\n\n"
        f"Resolved tech stack:\n{_json_block(resolved_stack)}\n\n"
        f"Structured build context:\n{_json_block(build_context)}\n\n"
        "Use resolvedTechStack as the binding stack decision; "
        "required stack items override MVPilot defaults. "
        "generated files, tests, and architecture must match resolvedTechStack. "
        "Align files and layout with requiredRepositoryFormat and allowedToolsAndAPIs. "
        "Return files, selected stack, required files, repo structure, implementation steps, "
        "agent assignments, GitHub actions needed, generated artifacts, security constraints, "
        "demo requirements, test plan, architecture notes, mode, and decision trace. "
        "Do not include secrets or real .env values in generated files."
    )


def build_file_manifest_prompt(
    *,
    idea: str,
    repo_plan: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    resolved_stack = build_context.get("resolvedTechStack", {})
    return (
        "Create the generated artifact manifest for the MVP repo.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(resolved_stack)}\n\n"
        f"Repo plan:\n{_json_block(repo_plan)}\n\n"
        f"Structured build context:\n{_json_block(build_context)}\n\n"
        "Return artifact names, kinds, summaries, full text content, mode, and "
        "decision trace. Generate a real runnable MVP repository tailored to this idea, "
        "not a generic placeholder or unrelated starter app. "
        "Label mock integrations clearly in code comments where external APIs are not wired. "
        "Include frontend files, backend/API files, database schema notes when persistence is useful, "
        "tests, README.md, docs/ARCHITECTURE.md, docs/IMPLEMENTATION_PLAN.md, "
        "docs/BUILD_LOG.md, demo/demo_script.md, package.json and/or requirements.txt, "
        "and at least one file under src/ or backend/. Do not include secrets or a real .env file."
    )


def build_blocker_analysis_prompt(
    *,
    idea: str,
    tool_result: dict[str, Any],
) -> str:
    return (
        "Analyze this blocked build result and decide whether it can recover.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Tool result:\n{_json_block(tool_result)}\n\n"
        "Return blocker type, severity, recoverable flag, root cause, recovery "
        "plan, and decision trace."
    )


def build_final_readme_prompt(
    *,
    idea: str,
    mvp_scope: dict[str, Any],
    repo_plan: dict[str, Any],
    generated_artifacts: list[dict[str, Any]],
    build_context: dict[str, Any],
) -> str:
    resolved_stack = build_context.get("resolvedTechStack", {})
    return (
        "Write the final README content for the generated MVP package.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(resolved_stack)}\n\n"
        f"MVP scope:\n{_json_block(mvp_scope)}\n\n"
        f"Repo plan:\n{_json_block(repo_plan)}\n\n"
        f"Artifacts:\n{_json_block(generated_artifacts)}\n\n"
        "Return title, README markdown content, setup steps, and decision trace."
    )


def build_demo_script_prompt(
    *,
    idea: str,
    blocker_analysis: dict[str, Any] | None,
    build_context: dict[str, Any],
) -> str:
    resolved_stack = build_context.get("resolvedTechStack", {})
    return (
        "Write a short demo script for judges.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(resolved_stack)}\n\n"
        f"Blocker analysis:\n{_json_block(blocker_analysis or {})}\n\n"
        "Return title, script content, demo beats, and decision trace."
    )


def build_pitch_prompt(
    *,
    idea: str,
    final_readme: dict[str, Any],
    demo_script: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    resolved_stack = build_context.get("resolvedTechStack", {})
    return (
        "Write the final hackathon pitch for this package.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(resolved_stack)}\n\n"
        f"README:\n{_json_block(final_readme)}\n\n"
        f"Demo script:\n{_json_block(demo_script)}\n\n"
        "Return title, tagline, pitch content, proof points, and decision trace."
    )
