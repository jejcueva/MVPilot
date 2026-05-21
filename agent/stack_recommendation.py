from __future__ import annotations

import re
from typing import Any

from agent.model_outputs import RecommendedStackOutput

# MVPilot host platform — never treat as the default generated-project stack.
HOST_PLATFORM_STACK_LABEL = (
    "MVPilot host platform defaults (orchestrator UI only; not the generated project stack)."
)

_SPONSOR_KEYWORDS = (
    ("nemotron", "aiModels", "NVIDIA Nemotron"),
    ("nvidia", "aiModels", "NVIDIA Nemotron"),
    ("openclaw", "orchestration", "OpenClaw"),
    ("pgvector", "vectorStorage", "pgvector on Supabase Postgres"),
    ("supabase", "database", "Supabase Postgres"),
)


def _collect_rule_text(build_context: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in (
        "requiredTechStackPieces",
        "allowedToolsAndAPIs",
        "requiredDeliverables",
        "requiredDemoFormat",
    ):
        for item in build_context.get(field) or []:
            if isinstance(item, dict):
                parts.append(str(item.get("item") or ""))
                parts.append(str(item.get("reason") or ""))
            else:
                parts.append(str(item))
    for warning in build_context.get("scopeWarnings") or []:
        if isinstance(warning, dict):
            parts.append(str(warning.get("item") or ""))
            parts.append(str(warning.get("reason") or ""))
    for evidence in build_context.get("evidence") or []:
        if isinstance(evidence, dict):
            parts.append(str(evidence.get("content") or ""))
    return " ".join(parts).lower()


def _mentions(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def recommend_stack_heuristic(
    *,
    idea: str,
    project_requirements: dict[str, Any] | None,
    build_context: dict[str, Any],
) -> dict[str, Any]:
    """Project-specific stack when live Nemotron is unavailable (explicit degraded path)."""
    requirements = project_requirements or {}
    intake = build_context.get("frontendIntake") or {}
    if not isinstance(intake, dict):
        intake = {}

    depth = str(
        requirements.get("project_depth")
        or intake.get("projectDepth")
        or "Advanced Project"
    )
    platform = str(
        requirements.get("target_platform")
        or intake.get("targetPlatform")
        or "web app"
    ).lower()
    preference = str(
        intake.get("techStackPreference") or intake.get("tech_stack_preference") or ""
    ).strip()

    rules_text = _collect_rule_text(build_context)
    idea_lower = idea.lower()
    study = _mentions(idea_lower, "study", "lecture", "quiz", "flashcard", "exam")
    sponsor_nemotron = _mentions(rules_text, "nemotron", "nvidia")
    sponsor_openclaw = "openclaw" in rules_text
    hackathon = _mentions(rules_text, "hackathon", "judging", "demo", "sponsor")

    if platform in {"api", "backend api"}:
        frontend = "Minimal admin UI or API docs (OpenAPI) only"
    elif platform in {"mobile app", "mobile"}:
        frontend = "React Native (Expo) with TypeScript"
    else:
        frontend = "Next.js 14 with React and TypeScript"

    if preference:
        backend = preference if len(preference) < 80 else "FastAPI (Python 3.12)"
    elif study and depth.lower().startswith("hackathon"):
        backend = "FastAPI (Python 3.12) for AI pipelines and file parsing"
    elif platform == "api":
        backend = "FastAPI (Python 3.12)"
    else:
        backend = "FastAPI (Python 3.12) or Node.js Express when team prefers JS end-to-end"

    database = (
        "Supabase Postgres"
        if sponsor_nemotron or _mentions(rules_text, "supabase", "postgres")
        else "PostgreSQL (managed) with migrations"
    )
    authentication = (
        "Supabase Auth"
        if "supabase" in database.lower()
        else "Clerk or JWT session auth"
    )
    ai_models: list[str] = []
    if sponsor_nemotron or study:
        ai_models.append("NVIDIA Nemotron for summarization, quiz generation, and tutoring")
    orchestration: list[str] = []
    if sponsor_openclaw or sponsor_nemotron:
        orchestration.append("OpenClaw tool orchestration for agent steps and GitHub export")
    rag_retrieval = (
        "URL + document ingestion with Nemotron embeddings"
        if sponsor_nemotron
        else "Reference URL fetch and chunking"
    )
    vector_storage = (
        "pgvector on Supabase Postgres"
        if sponsor_nemotron or _mentions(rules_text, "pgvector", "vector")
        else "Optional vector store if note search is required at scale"
    )
    if study:
        rag_retrieval = "Ingest lecture notes (PDF/Markdown); embed with Nemotron; rerank for quiz context"
        vector_storage = "pgvector on Supabase for note chunks and spaced-repetition metadata"

    deployment = (
        "Vercel (frontend) + Supabase (database/auth) + GPU-backed Nemotron API"
        if hackathon and sponsor_nemotron
        else "Vercel or Render for frontend; containerized FastAPI; managed Postgres"
    )
    testing = "pytest for backend; Playwright or Vitest for critical UI flows; contract tests for APIs"

    alignment: list[str] = []
    if sponsor_nemotron:
        alignment.append("Hackathon/RAG context rewards NVIDIA Nemotron — included in aiModels.")
    if sponsor_openclaw:
        alignment.append("OpenClaw called out in rules — included in orchestration.")
    if hackathon:
        alignment.append("Stack favors fast demo: managed auth/DB and a deployable web UI.")
    if not alignment:
        alignment.append("No sponsor-mandated stack in RAG; choices follow idea, depth, and platform.")

    rejected: list[str] = []
    rejected.append(
        f"Alternative (not default): {HOST_PLATFORM_STACK_LABEL}"
    )
    if not sponsor_nemotron:
        rejected.append("Skipped generic MVPilot host stack (Next.js orchestrator + internal LangGraph).")

    reasons = [
        f"Selected for '{idea[:80]}' at {depth} on {platform}.",
        "Stack is chosen for the generated product, not copied from MVPilot's host codebase.",
    ]
    if study:
        reasons.append(
            "Study/education domain needs upload parsing, AI generation, spaced repetition, and dashboards."
        )
    if preference:
        reasons.append(f"Honored user preference where compatible: {preference}.")

    output = RecommendedStackOutput(
        frontend=frontend,
        backend=backend,
        database=database,
        authentication=authentication,
        aiModels=ai_models or ["Use project-appropriate LLM APIs; configure in .env.example"],
        orchestration=orchestration or ["In-app workflow coordinator"],
        ragRetrieval=rag_retrieval,
        vectorStorage=vector_storage,
        deployment=deployment,
        testing=testing,
        reasonForChoices=reasons,
        hackathonRuleAlignment=alignment,
        rejectedAlternatives=rejected,
        ruleConflicts=[],
        mode="degraded",
        decision_trace=[
            "Stack Selector Agent used explicit degraded heuristics.",
            "Did not copy MVPilot host platform defaults into the generated project stack.",
            *reasons[:2],
        ],
    )
    return output.model_dump()


def apply_recommended_stack_to_build_context(
    build_context: dict[str, Any],
    recommended: dict[str, Any],
) -> dict[str, Any]:
    """Merge recommendedStack and binding resolvedTechStack into build context."""
    stack_items = stack_items_from_recommended(recommended)
    merged = dict(build_context)
    merged["recommendedStack"] = recommended
    host_defaults = []
    resolved = merged.get("resolvedTechStack")
    if isinstance(resolved, dict):
        host_defaults = list(resolved.get("defaultItems") or [])
        required = list(resolved.get("requiredItems") or [])
    else:
        required = []

    merged["resolvedTechStack"] = {
        "source": "stack_recommendation",
        "items": stack_items,
        "requiredItems": required,
        "defaultItems": host_defaults,
        "hostPlatformDefaults": host_defaults,
        "reason": "; ".join(recommended.get("reasonForChoices") or [])[:500]
        or "Project-specific stack from Stack Selector Agent.",
    }
    return merged


def stack_items_from_recommended(recommended: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in (
        "frontend",
        "backend",
        "database",
        "authentication",
        "deployment",
        "testing",
    ):
        value = recommended.get(key)
        if isinstance(value, str) and value.strip():
            items.append(value.strip())
    for key in ("aiModels", "orchestration", "ragRetrieval", "vectorStorage"):
        value = recommended.get(key)
        if isinstance(value, str) and value.strip():
            items.append(value.strip())
        elif isinstance(value, list):
            items.extend(str(v).strip() for v in value if str(v).strip())
    return _unique(items)


def recommended_stack_summary(recommended: dict[str, Any]) -> str:
    return ", ".join(stack_items_from_recommended(recommended))


def align_architecture_plan_with_recommended_stack(
    plan: dict[str, Any],
    recommended: dict[str, Any] | None,
) -> dict[str, Any]:
    """Ensure architecture plan selected_stack matches Stack Selector output."""
    if not recommended:
        return plan

    stack_list = stack_items_from_recommended(recommended)
    if not stack_list:
        return plan

    aligned = dict(plan)
    aligned["selected_stack"] = stack_list

    overview = list(aligned.get("architecture_overview") or [])
    binding_line = (
        f"Binding stack from Stack Selector Agent: {recommended_stack_summary(recommended)}."
    )
    if not overview:
        aligned["architecture_overview"] = [
            binding_line,
            "Architecture follows the recommended project stack, not MVPilot host defaults.",
        ]
    elif binding_line not in overview[0]:
        aligned["architecture_overview"] = [binding_line, *overview]

    trace = list(aligned.get("decision_trace") or [])
    trace.append(
        "Aligned selected_stack with recommendedStack from the Stack Selector Agent."
    )
    aligned["decision_trace"] = trace
    return aligned


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(value.strip())
    return items
