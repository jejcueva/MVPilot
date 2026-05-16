from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from agent.rag.retrieve import search_rag
from agent.rag.types import (
    BuildContextItem,
    BuildContextOptionalParams,
    BuildContextRequest,
    BuildContextResponse,
    BuildContextResponseCategory,
    DocType,
    EvidenceItem,
    Priority,
    RagSearchResult,
    ResolvedTechStack,
    ResolvedTechStackSource,
    ScopeWarningItem,
)

logger = logging.getLogger(__name__)

BUILD_CONTEXT_DOC_TYPES: list[DocType] = [
    "required_deliverables",
    "allowed_tools_apis",
    "repository_format",
    "demo_format",
    "tech_stack",
    "security_constraints",
    "agent_boundaries",
    "scope_warning",
    "hackathon_rules",
    "nvidia_docs",
    "nvidia_model_docs",
    "nvidia_model_usage",
    "agent_architecture",
    "implementation_constraints",
    "generated_project_doc",
]

DOC_TYPE_TO_CATEGORY: dict[DocType, BuildContextResponseCategory] = {
    "required_deliverables": "requiredDeliverables",
    "allowed_tools_apis": "allowedToolsAndAPIs",
    "repository_format": "requiredRepositoryFormat",
    "demo_format": "requiredDemoFormat",
    "tech_stack": "requiredTechStackPieces",
    "security_constraints": "scopeWarnings",
    "agent_boundaries": "scopeWarnings",
    "hackathon_rules": "requiredDeliverables",
    "nvidia_docs": "requiredTechStackPieces",
    "nvidia_model_docs": "requiredTechStackPieces",
    "nvidia_model_usage": "requiredTechStackPieces",
    "agent_architecture": "requiredTechStackPieces",
    "implementation_constraints": "scopeWarnings",
    "generated_project_doc": "requiredRepositoryFormat",
    "build_log": "scopeWarnings",
    "team_notes": "scopeWarnings",
    "unknown": "scopeWarnings",
}

_CATEGORY_FIELDS: dict[BuildContextResponseCategory, str] = {
    "requiredDeliverables": "requiredDeliverables",
    "allowedToolsAndAPIs": "allowedToolsAndAPIs",
    "requiredRepositoryFormat": "requiredRepositoryFormat",
    "requiredDemoFormat": "requiredDemoFormat",
    "requiredTechStackPieces": "requiredTechStackPieces",
}

DEFAULT_TECH_STACK_ITEMS: tuple[str, ...] = (
    "Next.js",
    "React",
    "TypeScript",
    "Tailwind CSS",
    "Python 3.12",
    "FastAPI",
    "Uvicorn",
    "Supabase Postgres",
    "pgvector",
    "NVIDIA Nemotron",
    "pytest",
    "npm run build",
)

_DEFAULT_STACK_CATEGORIES: dict[str, str] = {
    "Next.js": "frontend_framework",
    "React": "frontend_library",
    "TypeScript": "frontend_language",
    "Tailwind CSS": "frontend_style",
    "Python 3.12": "backend_language",
    "FastAPI": "backend_framework",
    "Uvicorn": "backend_server",
    "Supabase Postgres": "database",
    "pgvector": "vector_store",
    "NVIDIA Nemotron": "ai_model",
    "pytest": "python_tests",
    "npm run build": "js_build",
}

_DEFAULT_STACK_ALIASES: dict[str, tuple[str, ...]] = {
    "Next.js": ("next.js", "nextjs"),
    "React": ("react",),
    "TypeScript": ("typescript",),
    "Tailwind CSS": ("tailwind",),
    "Python 3.12": ("python", "python 3"),
    "FastAPI": ("fastapi",),
    "Uvicorn": ("uvicorn",),
    "Supabase Postgres": ("supabase", "postgres", "postgresql"),
    "pgvector": ("pgvector",),
    "NVIDIA Nemotron": ("nemotron", "nvidia"),
    "pytest": ("pytest",),
    "npm run build": ("npm run build", "npm build"),
}

FALLBACK_BUILD_ITEMS: dict[BuildContextResponseCategory, tuple[str, ...]] = {
    "requiredDeliverables": (
        "Working MVP that demonstrates the submitted idea end to end",
        "README.md with setup, architecture, and NVIDIA/Nemotron usage notes",
    ),
    "allowedToolsAndAPIs": (
        "Use NVIDIA/Nemotron APIs for reasoning, embeddings, and reranking when configured",
        "Use GitHub API through backend-owned OAuth or backend environment credentials only",
        "Use Supabase Postgres and pgvector for RAG, memory, audit, and shared project state",
    ),
    "requiredRepositoryFormat": (
        "README.md at the repository root",
        "docs/ARCHITECTURE.md and docs/BUILD_LOG.md for judge-visible evidence",
        ".env.example may be committed with placeholders; real .env files must never be committed",
    ),
    "requiredDemoFormat": (
        "Frontend launches the project and shows flight-stage progress",
        "Orchestrator retrieves RAG evidence before planning and GitHub actions",
        "Final landing card shows repository, commit, build log, and architecture links",
    ),
    "requiredTechStackPieces": (
        "Next.js + TypeScript + Tailwind frontend",
        "Python 3.12 + FastAPI orchestrator backend",
        "Supabase Postgres + pgvector RAG and memory store",
        "NVIDIA Nemotron reasoning, embedding, and reranking models",
    ),
}

FALLBACK_SCOPE_WARNINGS: tuple[str, ...] = (
    "No strong RAG evidence was retrieved; treat defaults as safe MVPilot assumptions.",
    "Never expose frontend-submitted or backend-owned secrets in generated files.",
)

_STACK_CONFLICT_TERMS: dict[str, tuple[str, ...]] = {
    "frontend_framework": ("vue", "svelte", "angular", "nuxt", "remix", "vite"),
    "frontend_library": ("vue", "svelte", "solid", "angular"),
    "frontend_language": ("javascript", "js only"),
    "frontend_style": (
        "bootstrap",
        "material ui",
        "mui",
        "chakra",
        "styled-components",
    ),
    "backend_language": (
        "node.js",
        "nodejs",
        "node ",
        "go ",
        "golang",
        "rust",
        "java",
        "kotlin",
        "ruby",
        "c#",
    ),
    "backend_framework": (
        "express",
        "nestjs",
        "django",
        "flask",
        "spring",
        "rails",
        "hono",
        "koa",
    ),
    "backend_server": (
        "express",
        "nestjs",
        "django",
        "flask",
        "spring",
        "rails",
        "hono",
        "koa",
    ),
    "database": ("firebase", "firestore", "mongodb", "mysql", "sqlite", "dynamodb", "redis"),
    "vector_store": ("pinecone", "weaviate", "qdrant", "milvus", "chromadb", "faiss"),
    "ai_model": ("openai", "anthropic", "claude", "gemini", "llama", "mistral", "gpt"),
    "python_tests": ("unittest", "jest", "vitest"),
    "js_build": ("pnpm build", "yarn build", "vite build", "next build"),
}


async def get_build_context(
    project_id: str,
    idea: str,
    optional_params: BuildContextOptionalParams | dict[str, Any] | None = None,
    *,
    top_k: int = 8,
) -> BuildContextResponse:
    parsed_params: BuildContextOptionalParams | None
    if isinstance(optional_params, dict):
        parsed_params = BuildContextOptionalParams.model_validate(optional_params)
    else:
        parsed_params = optional_params

    request = BuildContextRequest(
        projectId=project_id,
        idea=idea,
        optionalParams=parsed_params,
        topK=top_k,
    )
    return await build_build_context_response(request)


async def build_build_context_response(request: BuildContextRequest) -> BuildContextResponse:
    query = _build_search_query(request)
    chunks = await search_rag(query, request.topK, BUILD_CONTEXT_DOC_TYPES)
    reranked_count = len(chunks)

    categorized = _categorize_chunks(chunks)
    evidence = _build_evidence(chunks)
    empty_categories: list[str] = []

    response_items: dict[BuildContextResponseCategory, list[BuildContextItem]] = {
        category: categorized.get(category, []) for category in _CATEGORY_FIELDS
    }

    for category in _CATEGORY_FIELDS:
        if not response_items[category]:
            empty_categories.append(category)
            response_items[category] = _default_items(category)

    scope_warnings: list[ScopeWarningItem] = [
        ScopeWarningItem(item=item.item, reason=item.reason, source=item.source)
        for item in categorized.get("scopeWarnings", [])
    ]
    if not chunks:
        scope_warnings.extend(
            ScopeWarningItem(
                item=item,
                reason="Default safety context used because vector retrieval returned no evidence.",
                source="mvpilot_default_build_context",
            )
            for item in FALLBACK_SCOPE_WARNINGS
        )

    logger.info(
        "rag.get_build_context projectId=%s query=%r docTypes=%s chunks=%s reranked=%s emptyCategories=%s",
        request.projectId,
        query,
        BUILD_CONTEXT_DOC_TYPES,
        len(chunks),
        reranked_count,
        empty_categories,
    )

    return BuildContextResponse(
        requiredDeliverables=response_items["requiredDeliverables"],
        allowedToolsAndAPIs=response_items["allowedToolsAndAPIs"],
        requiredRepositoryFormat=response_items["requiredRepositoryFormat"],
        requiredDemoFormat=response_items["requiredDemoFormat"],
        requiredTechStackPieces=response_items["requiredTechStackPieces"],
        agentBoundaries=_build_agent_boundaries(chunks),
        resolvedTechStack=_resolve_tech_stack(
            required_items=categorized.get("requiredTechStackPieces", []),
            optional_params=request.optionalParams,
        ),
        scopeWarnings=scope_warnings,
        evidence=evidence,
    )


def _build_search_query(request: BuildContextRequest) -> str:
    parts = [
        request.idea,
        "hackathon required deliverables allowed tools APIs repository format demo format tech stack constraints",
    ]
    params = request.optionalParams
    if params:
        if params.stack:
            parts.append("stack: " + ", ".join(params.stack))
        if params.features:
            parts.append("features: " + ", ".join(params.features))
        if params.sourceUrls:
            parts.append("source urls: " + ", ".join(params.sourceUrls))
        if params.repoPreference:
            parts.append(f"repository preference: {params.repoPreference}")
        if params.repoName:
            parts.append(f"repository name: {params.repoName}")
        if params.repoUrl:
            parts.append(f"repository url: {params.repoUrl}")
        if params.visibility:
            parts.append(f"visibility: {params.visibility}")
        if params.demoPreference:
            parts.append(f"demo preference: {params.demoPreference}")
    if request.rulesUrl:
        parts.append(f"rules url: {request.rulesUrl}")
    if request.referenceUrls:
        parts.append("reference urls: " + ", ".join(request.referenceUrls))
    if request.contextNeeded:
        parts.append("context needed: " + ", ".join(request.contextNeeded))
    return " ".join(part for part in parts if part).strip()


def _default_items(category: BuildContextResponseCategory) -> list[BuildContextItem]:
    return [
        BuildContextItem(
            item=item,
            priority="high",
            reason="Default MVPilot build context used because RAG did not return this category.",
            source="mvpilot_default_build_context",
        )
        for item in FALLBACK_BUILD_ITEMS[category]
    ]


def _categorize_chunks(chunks: list[RagSearchResult]) -> dict[str, list[BuildContextItem]]:
    buckets: dict[str, list[BuildContextItem]] = {category: [] for category in _CATEGORY_FIELDS}
    buckets["scopeWarnings"] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        category = DOC_TYPE_TO_CATEGORY.get(chunk.doc_type, "scopeWarnings")
        if chunk.doc_type == "scope_warning":
            category = "scopeWarnings"

        for bullet in _extract_bullets(chunk.text):
            key = (category, _normalize_item(bullet))
            if key in seen:
                continue
            seen.add(key)

            item = BuildContextItem(
                item=bullet,
                priority=_infer_priority(bullet, chunk),
                reason=_build_reason(chunk),
                source=chunk.source,
            )

            if category == "scopeWarnings":
                buckets["scopeWarnings"].append(item)
            else:
                buckets[category].append(item)

    for category in _CATEGORY_FIELDS:
        buckets[category].sort(key=lambda item: _priority_rank(item.priority), reverse=True)

    return buckets


def _extract_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^[-*]\s+(.+)$", line.strip())
        if match:
            bullets.append(match.group(1).strip())
    if not bullets:
        condensed = re.sub(r"\s+", " ", text).strip()
        if condensed and len(condensed) <= 240:
            bullets.append(condensed)
    return bullets


def _infer_priority(text: str, chunk: RagSearchResult) -> Priority:
    lowered = text.lower()
    if any(word in lowered for word in ("must", "required", "critical", "never", "do not")):
        return "critical"
    if any(word in lowered for word in ("should", "need to", "ensure")):
        return "high"
    if chunk.doc_type in {"hackathon_rules", "required_deliverables", "scope_warning"}:
        return "high"
    if chunk.authority_score >= 0.9:
        return "high"
    if any(word in lowered for word in ("may", "optional", "can")):
        return "low"
    return "medium"


def _build_reason(chunk: RagSearchResult) -> str:
    section = chunk.metadata.get("section_heading")
    if section:
        return f"Grounded in section '{section}' from {chunk.source}."
    return f"Grounded in {chunk.doc_type} documentation from {chunk.source}."


def _build_evidence(chunks: list[RagSearchResult]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source=chunk.source,
            docType=chunk.doc_type,
            chunkId=chunk.chunk_id,
            content=chunk.text,
            score=chunk.score,
        )
        for chunk in chunks
    ]


def _build_agent_boundaries(chunks: list[RagSearchResult]) -> dict[str, Any]:
    boundaries = [
        bullet
        for chunk in chunks
        if chunk.doc_type in {"agent_boundaries", "agent_architecture", "implementation_constraints", "scope_warning"}
        for bullet in _extract_bullets(chunk.text)
    ]
    if not boundaries:
        boundaries = [
            "RAG provides grounded context and memory only; it does not decide the final project.",
            "Orchestrator owns planning and sequencing before GitHub actions.",
            "GitHub Agent owns repository mutations and must reject unsafe paths or secrets.",
            "Frontend never sends GitHub or Supabase service-role tokens.",
        ]
    return {
        "rules": _unique_strings(boundaries),
        "source": "rag_evidence" if chunks else "mvpilot_default_build_context",
    }


def _resolve_tech_stack(
    *,
    required_items: list[BuildContextItem],
    optional_params: BuildContextOptionalParams | None,
) -> ResolvedTechStack:
    required_stack_items = _unique_strings(item.item for item in required_items)
    preferred_stack_items = _unique_strings(optional_params.stack if optional_params else [])
    explicit_items = [*required_stack_items, *preferred_stack_items]

    default_items = [
        item
        for item in DEFAULT_TECH_STACK_ITEMS
        if _should_add_default_stack_item(item=item, explicit_items=explicit_items)
    ]
    items = _unique_strings([*required_stack_items, *preferred_stack_items, *default_items])
    source = _resolve_stack_source(
        has_required=bool(required_stack_items),
        has_preferences=bool(preferred_stack_items),
        has_defaults=bool(default_items),
    )

    return ResolvedTechStack(
        source=source,
        items=items,
        requiredItems=required_stack_items,
        defaultItems=default_items,
        reason=_resolved_stack_reason(source),
    )


def _should_add_default_stack_item(*, item: str, explicit_items: list[str]) -> bool:
    category = _DEFAULT_STACK_CATEGORIES[item]
    return not (
        _explicitly_covers_default(item=item, explicit_items=explicit_items)
        or _explicitly_blocks_category(category=category, explicit_items=explicit_items)
    )


def _explicitly_covers_default(*, item: str, explicit_items: list[str]) -> bool:
    aliases = _DEFAULT_STACK_ALIASES[item]
    return any(
        alias in explicit_item.lower()
        for explicit_item in explicit_items
        for alias in aliases
    )


def _explicitly_blocks_category(*, category: str, explicit_items: list[str]) -> bool:
    conflict_terms = _STACK_CONFLICT_TERMS.get(category, ())
    return any(
        term in explicit_item.lower()
        for explicit_item in explicit_items
        for term in conflict_terms
    )


def _resolve_stack_source(
    *,
    has_required: bool,
    has_preferences: bool,
    has_defaults: bool,
) -> ResolvedTechStackSource:
    if has_defaults and (has_required or has_preferences):
        return "mixed"
    if has_required:
        return "rag_required"
    if has_preferences:
        return "request_preference"
    return "default"


def _resolved_stack_reason(source: ResolvedTechStackSource) -> str:
    reasons = {
        "rag_required": "RAG required stack items covered the stack, so MVPilot defaults were not needed.",
        "request_preference": "Request stack preferences covered the stack, so MVPilot defaults were not needed.",
        "default": "No required or preferred stack was found, so MVPilot defaults were applied.",
        "mixed": (
            "Required or preferred stack items were kept, and MVPilot defaults filled "
            "silent categories without overriding explicit choices."
        ),
    }
    return reasons[source]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item:
            continue
        key = _normalize_item(item)
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _normalize_item(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _priority_rank(priority: Priority) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}[priority]
