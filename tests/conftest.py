from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agent.config import Settings
from agent.main import create_app
from agent.rag.types import RagSearchResult


def _sample_rag_chunks() -> list[RagSearchResult]:
    return [
        RagSearchResult(
            chunk_id="deliverables-0",
            source="rag/sources/mvpilot_build_requirements.md",
            title="Required Deliverables",
            doc_type="required_deliverables",
            authority_score=0.98,
            text="# Required Deliverables\n\n- Working MVP\n- README.md",
            metadata={"section_heading": "Required Deliverables"},
            score=0.92,
            similarity=0.92,
        ),
        RagSearchResult(
            chunk_id="tools-0",
            source="rag/sources/mvpilot_build_requirements.md",
            title="Allowed Tools and APIs",
            doc_type="allowed_tools_apis",
            authority_score=0.93,
            text="# Allowed Tools and APIs\n\n- NVIDIA API\n- GitHub API",
            metadata={"section_heading": "Allowed Tools and APIs"},
            score=0.9,
            similarity=0.9,
        ),
        RagSearchResult(
            chunk_id="repo-0",
            source="rag/sources/mvpilot_build_requirements.md",
            title="Required Repository Format",
            doc_type="repository_format",
            authority_score=0.9,
            text="# Required Repository Format\n\n- README.md\n- .env.example",
            metadata={"section_heading": "Required Repository Format"},
            score=0.89,
            similarity=0.89,
        ),
        RagSearchResult(
            chunk_id="demo-0",
            source="rag/sources/mvpilot_build_requirements.md",
            title="Required Demo Format",
            doc_type="demo_format",
            authority_score=0.9,
            text="# Required Demo Format\n\n- User enters project idea\n- Orchestrator calls RAG",
            metadata={"section_heading": "Required Demo Format"},
            score=0.87,
            similarity=0.87,
        ),
        RagSearchResult(
            chunk_id="stack-0",
            source="rag/sources/mvpilot_build_requirements.md",
            title="Required Tech Stack Pieces",
            doc_type="tech_stack",
            authority_score=0.88,
            text="# Required Tech Stack Pieces\n\n- FastAPI backend\n- Supabase pgvector",
            metadata={"section_heading": "Required Tech Stack Pieces"},
            score=0.88,
            similarity=0.88,
        ),
        RagSearchResult(
            chunk_id="hackathon-0",
            source="rag/sources/hackathon_rules.md",
            title="Hackathon Rules",
            doc_type="hackathon_rules",
            authority_score=1.0,
            text="# Hackathon Rules\n\n- Build the smallest working demo.",
            metadata={},
            score=0.95,
            similarity=0.95,
        ),
        RagSearchResult(
            chunk_id="nvidia-0",
            source="rag/sources/nvidia_models.md",
            title="NVIDIA Models",
            doc_type="nvidia_docs",
            authority_score=0.95,
            text="# NVIDIA Models\n\n- Use Nemotron for orchestration.",
            metadata={},
            score=0.91,
            similarity=0.91,
        ),
        RagSearchResult(
            chunk_id="logs-0",
            source="logs/build_log.md",
            title="Build Log",
            doc_type="build_log",
            authority_score=0.75,
            text="# Build Log\n\n- Tests passed after recovery.",
            metadata={},
            score=0.7,
            similarity=0.7,
        ),
        RagSearchResult(
            chunk_id="warn-0",
            source="rag/sources/mvpilot_build_requirements.md",
            title="Scope Warnings",
            doc_type="scope_warning",
            authority_score=0.7,
            text="# Scope Warnings\n\n- Do not expose SUPABASE_SERVICE_ROLE_KEY to the frontend",
            metadata={"section_heading": "Scope Warnings"},
            score=0.8,
            similarity=0.8,
        ),
    ]


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.delenv("OPENCLAW_API_KEY", raising=False)
    return Settings(
        _env_file=None,
        adapter_mode="mock",
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings=settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_live_rag_search(monkeypatch):
    chunks = _sample_rag_chunks()

    async def fake_search(query: str, top_k: int = 5, doc_types=None):
        if doc_types:
            filtered = [chunk for chunk in chunks if chunk.doc_type in doc_types]
            return filtered[:top_k]
        return chunks[:top_k]

    monkeypatch.setattr("agent.live_adapters.search_rag", fake_search)
    monkeypatch.setattr("agent.rag.build_context.search_rag", fake_search)
    
    from unittest.mock import AsyncMock, MagicMock
    mock_store = MagicMock()
    mock_store.search_memories = AsyncMock(return_value=[{"id": "mock_memory"}])
    mock_store.write_memory = AsyncMock()
    
    monkeypatch.setattr("agent.rag.store.get_rag_store", lambda: mock_store)
    monkeypatch.setattr("agent.rag.embed.embed_text", AsyncMock(return_value=[0.1, 0.2]))

    return fake_search
