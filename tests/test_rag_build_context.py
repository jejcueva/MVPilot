import pytest

from agent.rag.build_context import BUILD_CONTEXT_DOC_TYPES, build_build_context_response, get_build_context
from agent.rag.chunk import chunk_document, detect_doc_type_from_section_heading
from agent.rag.types import BuildContextRequest, RagSearchResult, SourceDocument


def _chunk(
    *,
    chunk_id: str,
    source: str,
    doc_type: str,
    text: str,
    score: float = 0.9,
    section_heading: str | None = None,
) -> RagSearchResult:
    metadata = {"section_heading": section_heading} if section_heading else {}
    return RagSearchResult(
        chunk_id=chunk_id,
        source=source,
        title="Title",
        doc_type=doc_type,
        authority_score=0.95,
        text=text,
        metadata=metadata,
        score=score,
        similarity=score,
    )


@pytest.mark.asyncio
async def test_get_build_context_returns_all_categories(monkeypatch) -> None:
    chunks = [
        _chunk(
            chunk_id="deliverables-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="required_deliverables",
            section_heading="Required Deliverables",
            text="# Required Deliverables\n\n- Working MVP\n- README.md",
        ),
        _chunk(
            chunk_id="tools-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="allowed_tools_apis",
            section_heading="Allowed Tools and APIs",
            text="# Allowed Tools and APIs\n\n- NVIDIA API\n- GitHub API",
        ),
        _chunk(
            chunk_id="repo-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="repository_format",
            text="# Required Repository Format\n\n- README.md\n- .env.example",
        ),
        _chunk(
            chunk_id="demo-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="demo_format",
            text="# Required Demo Format\n\n- User enters project idea\n- Orchestrator calls RAG",
        ),
        _chunk(
            chunk_id="stack-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="tech_stack",
            text="# Required Tech Stack Pieces\n\n- FastAPI backend\n- Supabase pgvector",
        ),
        _chunk(
            chunk_id="warn-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="scope_warning",
            text="# Scope Warnings\n\n- Do not expose SUPABASE_SERVICE_ROLE_KEY to the frontend",
        ),
    ]

    async def fake_search(query: str, top_k: int, doc_types=None):
        assert doc_types == BUILD_CONTEXT_DOC_TYPES
        return chunks[:top_k]

    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)

    response = await get_build_context(
        "project-123",
        "Build an autonomous hackathon teammate",
        {"stack": ["FastAPI", "Supabase"], "features": ["RAG", "GitHub"]},
    )

    assert isinstance(response.requiredDeliverables, list)
    assert isinstance(response.allowedToolsAndAPIs, list)
    assert isinstance(response.requiredRepositoryFormat, list)
    assert isinstance(response.requiredDemoFormat, list)
    assert isinstance(response.requiredTechStackPieces, list)
    assert len(response.requiredDeliverables) >= 1
    assert len(response.allowedToolsAndAPIs) >= 1
    assert response.requiredDeliverables[0].source.endswith("mvpilot_build_requirements.md")
    assert response.evidence
    assert response.evidence[0].chunkId
    assert response.evidence[0].docType
    assert response.evidence[0].score > 0
    assert response.resolvedTechStack.requiredItems == [
        "FastAPI backend",
        "Supabase pgvector",
    ]
    assert response.resolvedTechStack.source in {"rag_required", "request_preference", "mixed"}


@pytest.mark.asyncio
async def test_get_build_context_returns_empty_categories_when_no_chunks(monkeypatch) -> None:
    async def fake_search(query: str, top_k: int, doc_types=None):
        return []

    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)

    response = await build_build_context_response(
        BuildContextRequest(projectId="p1", idea="AI teammate", topK=8)
    )

    assert len(response.requiredDeliverables) == 2
    assert all(
        item.source == "mvpilot_default_build_context"
        for item in response.requiredDeliverables
    )
    assert len(response.allowedToolsAndAPIs) == 3
    assert len(response.requiredRepositoryFormat) == 3
    assert len(response.requiredDemoFormat) == 3
    assert len(response.requiredTechStackPieces) == 3
    assert len(response.scopeWarnings) == 3
    assert response.scopeWarnings[0].source == "mvpilot_default_build_context"
    assert response.evidence == []
    assert response.resolvedTechStack.source == "default"
    assert response.resolvedTechStack.requiredItems == []
    assert response.resolvedTechStack.defaultItems == [
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
    ]
    assert response.resolvedTechStack.items == []


@pytest.mark.asyncio
async def test_resolved_tech_stack_keeps_required_items_without_conflicting_defaults(monkeypatch) -> None:
    chunks = [
        _chunk(
            chunk_id="stack-0",
            source="rag/sources/hackathon_rules.md",
            doc_type="tech_stack",
            text="# Required Tech Stack Pieces\n\n- Must use Flask backend\n- Must use MongoDB",
        )
    ]

    async def fake_search(query: str, top_k: int, doc_types=None):
        return chunks

    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)

    response = await build_build_context_response(
        BuildContextRequest(projectId="p1", idea="AI teammate", topK=8)
    )

    assert response.resolvedTechStack.requiredItems == [
        "Must use Flask backend",
        "Must use MongoDB",
    ]
    assert response.resolvedTechStack.items == response.resolvedTechStack.requiredItems
    assert "flask" in " ".join(response.resolvedTechStack.items).lower()


@pytest.mark.asyncio
async def test_partial_required_stack_uses_mixed_source_and_fills_missing_defaults(monkeypatch) -> None:
    chunks = [
        _chunk(
            chunk_id="stack-0",
            source="rag/sources/mvpilot_build_requirements.md",
            doc_type="tech_stack",
            text="# Required Tech Stack Pieces\n\n- FastAPI backend",
        )
    ]

    async def fake_search(query: str, top_k: int, doc_types=None):
        return chunks

    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)

    response = await build_build_context_response(
        BuildContextRequest(projectId="p1", idea="AI teammate", topK=8)
    )

    assert response.resolvedTechStack.source == "rag_required"
    assert response.resolvedTechStack.requiredItems == ["FastAPI backend"]
    assert response.resolvedTechStack.items == ["FastAPI backend"]
    assert "Next.js" in response.resolvedTechStack.defaultItems
    assert "Supabase Postgres" in response.resolvedTechStack.defaultItems


@pytest.mark.asyncio
async def test_optional_stack_preferences_are_included_in_resolved_stack(monkeypatch) -> None:
    async def fake_search(query: str, top_k: int, doc_types=None):
        return []

    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)

    response = await build_build_context_response(
        BuildContextRequest(
            projectId="p1",
            idea="AI teammate",
            optionalParams={"stack": ["Vue", "Firebase"]},
            topK=8,
        )
    )

    assert "Vue" in response.resolvedTechStack.items
    assert "Firebase" in response.resolvedTechStack.items
    assert response.resolvedTechStack.source == "request_preference"
    assert "Next.js" in response.resolvedTechStack.defaultItems
    assert "Vue" not in response.resolvedTechStack.defaultItems


def test_section_headings_map_to_build_doc_types() -> None:
    assert detect_doc_type_from_section_heading("# Required Deliverables") == "required_deliverables"
    assert detect_doc_type_from_section_heading("## Allowed Tools and APIs") == "allowed_tools_apis"
    assert detect_doc_type_from_section_heading("# Required Tech Stack Pieces") == "tech_stack"


def test_chunk_document_preserves_section_heading_metadata() -> None:
    text = """# MVPilot Build Requirements

# Required Deliverables

- Working MVP
- README.md
"""
    document = SourceDocument(
        source="rag/sources/mvpilot_build_requirements.md",
        title="MVPilot Build Requirements",
        doc_type="implementation_constraints",
        text=text,
    )

    chunks = chunk_document(document)
    deliverable_chunk = next(
        chunk for chunk in chunks if chunk.metadata.get("section_heading") == "Required Deliverables"
    )

    assert deliverable_chunk.doc_type == "required_deliverables"
    assert deliverable_chunk.metadata["source_path"] == "rag/sources/mvpilot_build_requirements.md"
    assert deliverable_chunk.text.startswith("# Required Deliverables")
    assert "Working MVP" in deliverable_chunk.text


def test_get_build_context_endpoint(client, monkeypatch) -> None:
    async def fake_build(request: BuildContextRequest):
        from agent.rag.types import BuildContextItem, BuildContextResponse, ResolvedTechStack

        item = BuildContextItem(
            item="Working MVP",
            priority="critical",
            reason="test",
            source="rag/sources/mvpilot_build_requirements.md",
        )
        return BuildContextResponse(
            requiredDeliverables=[item],
            allowedToolsAndAPIs=[item],
            requiredRepositoryFormat=[item],
            requiredDemoFormat=[item],
            requiredTechStackPieces=[item],
            resolvedTechStack=ResolvedTechStack(
                source="rag_required",
                items=["Working MVP"],
                requiredItems=["Working MVP"],
                defaultItems=[],
                reason="test",
            ),
            scopeWarnings=[],
            evidence=[],
        )

    monkeypatch.setattr("agent.rag.routes.build_build_context_response", fake_build)

    response = client.post(
        "/rag/get-build-context",
        json={
            "projectId": "demo-project",
            "idea": "Autonomous hackathon agent",
            "optionalParams": {"stack": ["FastAPI"]},
            "topK": 8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["requiredDeliverables"], list)
    assert isinstance(body["allowedToolsAndAPIs"], list)
    assert isinstance(body["requiredRepositoryFormat"], list)
    assert isinstance(body["requiredDemoFormat"], list)
    assert isinstance(body["requiredTechStackPieces"], list)
    assert isinstance(body["resolvedTechStack"], dict)
