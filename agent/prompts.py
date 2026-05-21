from __future__ import annotations

import json
from typing import Any


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _compact_planning_context(build_context: dict[str, Any]) -> dict[str, Any]:
    """Trim large RAG payloads for architecture calls to reduce gateway timeouts."""
    compact_keys = (
        "frontendIntake",
        "sourceContext",
        "recommendedStack",
        "resolvedTechStack",
        "projectDepth",
        "targetPlatform",
        "openclawEnabled",
        "scopeWarnings",
        "requiredDeliverables",
        "requiredTechStackPieces",
    )
    compact = {key: build_context[key] for key in compact_keys if key in build_context}
    evidence = build_context.get("evidence")
    if isinstance(evidence, list) and evidence:
        compact["evidence"] = evidence[:12]
    return compact


def _context_contract() -> str:
    return (
        "Frontend intake is the user's source of truth for project identity, idea, "
        "target platform, project depth, submitted URLs, uploaded files, GitHub repository "
        "context, and final labels. Use frontendIntake, sourceContext, recommendedStack, "
        "and retrieval evidence as binding build context. "
        "Do not assume the generated project must use MVPilot's internal host stack "
        "(Next.js/FastAPI/Supabase used to run MVPilot). Recommend a project-specific stack "
        "from the idea, scope, target platform, complexity, and retrieved hackathon rules. "
        "Required RAG rules override user preference. After stack recommendation, "
        "resolvedTechStack and recommendedStack are binding for architecture, tests, and files. "
        "defaultItems / hostPlatformDefaults in build context are orchestrator-only hints, "
        "not the generated product stack. Surface missing or unreadable sources as warnings."
    )


def build_stack_recommendation_prompt(
    *,
    idea: str,
    project_requirements: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    return (
        "You are the Stack Selector Agent. Recommend the most effective tech stack for the "
        "generated software project — not for MVPilot itself.\n"
        "Analyze the project idea, depth, target platform, required features, and all "
        "RAG-retrieved hackathon rules, sponsor requirements, judging criteria, allowed/banned "
        "tools, APIs, SDKs, and deployment constraints.\n"
        "Do not blindly copy MVPilot's host stack. Prefer sponsor-required or sponsor-rewarded "
        "technologies when they fit the product. Label any fallback option as an alternative, "
        "not a silent default.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Project requirements:\n{_json_block(project_requirements)}\n\n"
        f"RAG hints (required stack pieces — not yet the final stack):\n"
        f"{_json_block(build_context.get('resolvedTechStack', {}))}\n\n"
        f"Structured build context:\n{_json_block(build_context)}\n\n"
        "Return frontend, backend, database, authentication, aiModels (list), orchestration (list), "
        "ragRetrieval, vectorStorage, deployment, testing, reasonForChoices, "
        "hackathonRuleAlignment, rejectedAlternatives, ruleConflicts, mode, and decision_trace."
    )


def build_requirements_prompt(
    *,
    idea: str,
    build_context: dict[str, Any],
    memory_matches: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> str:
    del retrieved_docs
    return (
        "Expand this idea into a complete software project brief.\n"
        "Do not scope it down to a shallow prototype or generic starter app.\n"
        "The output must describe a full-scale generated project with real product depth, "
        "multiple features, data flows, authentication when relevant, and success criteria.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"RAG stack hints (Stack Selector finalizes the project stack next):\n"
        f"{_json_block(build_context.get('resolvedTechStack', {}))}\n\n"
        f"Structured build context:\n{_json_block(build_context)}\n\n"
        f"Memory matches:\n{_json_block(memory_matches)}\n\n"
        "Return target_users, user_personas, core_features, advanced_features, "
        "success_criteria, project_depth, target_platform, project_archetype, "
        "primary_entity, auth_required, database_required, data_entities, user_flows "
        "(3 or more steps with step, screen, action, api), api_routes, mode, and decision_trace. "
        "Default to Advanced Project or higher unless the user explicitly asks for Starter Project."
    )


def build_architecture_plan_prompt(
    *,
    idea: str,
    project_requirements: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    return (
        "Design the full generated repository architecture for this project.\n"
        "Plan a complete codebase, not a minimal demo. Include frontend, backend, data, API, auth, "
        "testing, docs, deployment readiness, and GitHub export tasks.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Project requirements:\n{_json_block(project_requirements)}\n\n"
        f"Recommended stack (binding):\n{_json_block(build_context.get('recommendedStack', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(build_context.get('resolvedTechStack', {}))}\n\n"
        f"Planning context (trimmed):\n{_json_block(_compact_planning_context(build_context))}\n\n"
        "Return files, file_tree, selected_stack, architecture_overview, frontend_architecture, "
        "backend_architecture, data_model, api_design, auth_design, database_schema, "
        "state_management, integration_points, implementation_steps, agent_assignments, "
        "github_actions_needed, generated_artifacts, security_constraints, test_plan, "
        "deployment_plan, documentation_plan, mode, and decision_trace. "
        "Do not include secrets or real .env values."
    )


def build_file_manifest_prompt(
    *,
    idea: str,
    project_requirements: dict[str, Any],
    architecture_plan: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    return (
        "Create the generated artifact manifest for a complete full-stack software project.\n"
        "The manifest must support the architecture plan, user flows, auth/data/API design, "
        "tests, README, setup guide, env guide, deployment notes, and final project report.\n"
        "Do not emit a generic todo app, starter dashboard, or shallow fallback template.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Project requirements:\n{_json_block(project_requirements)}\n\n"
        f"Architecture plan:\n{_json_block(architecture_plan)}\n\n"
        f"Recommended stack:\n{_json_block(build_context.get('recommendedStack', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(build_context.get('resolvedTechStack', {}))}\n\n"
        f"Structured build context:\n{_json_block(build_context)}\n\n"
        "Return artifact names, kinds, one-line summaries, mode, and decision_trace. "
        "You may include file content, but if content is omitted the backend hydrates complete "
        "production-style files locally. Required paths include README.md, package.json, index.html, "
        "src/App.jsx, src/main.jsx, src/lib/api.js, src/state/projectState.js, src/styles.css, "
        "backend/main.py, backend/models.py, backend/services.py, backend/db.py, requirements.txt, "
        "tests/test_backend.py, docs/PROJECT_PLAN.md, docs/ARCHITECTURE.md, docs/API_SPEC.md, "
        "docs/DATABASE_SCHEMA.sql, docs/TESTING_STRATEGY.md, docs/DEPLOY.md, docs/AGENT_LOG.md, "
        "docs/BUILD_LOG.md, docs/KNOWN_LIMITATIONS.md, docs/WALKTHROUGH.md, and .env.example."
    )


def build_blocker_analysis_prompt(
    *,
    idea: str,
    tool_result: dict[str, Any],
) -> str:
    return (
        "Analyze this blocked full-project generation result and decide whether it can recover.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Tool result:\n{_json_block(tool_result)}\n\n"
        "Return blocker type, severity, recoverable flag, root cause, recovery plan, and decision trace."
    )


def build_final_readme_prompt(
    *,
    idea: str,
    project_requirements: dict[str, Any],
    architecture_plan: dict[str, Any],
    generated_artifacts: list[dict[str, Any]],
    build_context: dict[str, Any],
) -> str:
    return (
        "Write the final README content for the generated full project.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Project requirements:\n{_json_block(project_requirements)}\n\n"
        f"Architecture plan:\n{_json_block(architecture_plan)}\n\n"
        f"Generated artifacts:\n{_json_block(generated_artifacts)}\n\n"
        "Return title, README markdown content, setup steps, and decision trace. "
        "Include product description, personas, features, architecture, env vars, setup, tests, "
        "deployment, limitations, and future improvements."
    )


def build_walkthrough_prompt(
    *,
    idea: str,
    blocker_analysis: dict[str, Any] | None,
    build_context: dict[str, Any],
) -> str:
    return (
        "Write a concise product walkthrough for the generated project.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(build_context.get('resolvedTechStack', {}))}\n\n"
        f"Blocker analysis:\n{_json_block(blocker_analysis or {})}\n\n"
        "Return title, walkthrough content, beats, and decision trace."
    )


def build_pitch_prompt(
    *,
    idea: str,
    final_readme: dict[str, Any],
    walkthrough: dict[str, Any],
    build_context: dict[str, Any],
) -> str:
    return (
        "Write the final project report pitch for this generated codebase.\n\n"
        f"Idea:\n{idea}\n\n"
        f"Prompt contract:\n{_context_contract()}\n\n"
        f"Frontend intake:\n{_json_block(build_context.get('frontendIntake', {}))}\n\n"
        f"Source context:\n{_json_block(build_context.get('sourceContext', {}))}\n\n"
        f"Resolved tech stack:\n{_json_block(build_context.get('resolvedTechStack', {}))}\n\n"
        f"README:\n{_json_block(final_readme)}\n\n"
        f"Walkthrough:\n{_json_block(walkthrough)}\n\n"
        "Return title, tagline, pitch content, proof points, and decision trace."
    )


# Compatibility aliases for older imports.
build_scope_mvp_prompt = build_requirements_prompt
build_plan_repo_prompt = build_architecture_plan_prompt
build_demo_script_prompt = build_walkthrough_prompt
build_recommend_stack_prompt = build_stack_recommendation_prompt
