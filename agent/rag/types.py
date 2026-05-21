from typing import Any, Literal

from pydantic import BaseModel, Field

DocType = Literal[
    "hackathon_rules",
    "nvidia_docs",
    "nvidia_model_docs",
    "team_notes",
    "build_log",
    "generated_project_doc",
    "required_deliverables",
    "allowed_tools_apis",
    "repository_format",
    "demo_format",
    "tech_stack",
    "scope_warning",
    "agent_architecture",
    "agent_boundaries",
    "security_constraints",
    "implementation_constraints",
    "nvidia_model_usage",
    "unknown",
]

Priority = Literal["critical", "high", "medium", "low"]

BuildContextResponseCategory = Literal[
    "requiredDeliverables",
    "allowedToolsAndAPIs",
    "requiredRepositoryFormat",
    "requiredDemoFormat",
    "requiredTechStackPieces",
]

ResolvedTechStackSource = Literal[
    "rag_required",
    "request_preference",
    "default",
    "mixed",
    "stack_recommendation",
]


class SourceDocument(BaseModel):
    source: str
    title: str
    doc_type: DocType
    text: str
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagChunk(BaseModel):
    chunk_id: str
    source: str
    title: str
    doc_type: DocType
    authority_score: float = Field(ge=0, le=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class RagSearchResult(RagChunk):
    score: float
    similarity: float | None = None
    rerank_score: float | None = None


class RagSourceSummary(BaseModel):
    source: str
    doc_type: DocType
    authority_score: float = Field(ge=0, le=1)
    chunk_count: int


class IngestResponse(BaseModel):
    success: bool
    documentsLoaded: int
    chunksCreated: int
    storedIn: Literal["supabase"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    topK: int = Field(default=5, ge=1, le=25)
    docTypes: list[DocType] | None = None


class ApiChunk(BaseModel):
    source: str
    title: str
    doc_type: DocType
    authority_score: float
    text: str
    score: float
    similarity: float | None = None

    @classmethod
    def from_search_result(cls, chunk: RagSearchResult) -> "ApiChunk":
        return cls(
            source=chunk.source,
            title=chunk.title,
            doc_type=chunk.doc_type,
            authority_score=chunk.authority_score,
            text=chunk.text,
            score=chunk.score,
            similarity=chunk.similarity,
        )


class SearchResponse(BaseModel):
    query: str
    chunks: list[ApiChunk]


class AnswerContextRequest(BaseModel):
    task: str = Field(min_length=1)
    query: str = Field(min_length=1)
    topK: int = Field(default=5, ge=1, le=25)


class AnswerContextResponse(BaseModel):
    task: str
    query: str
    context: list[ApiChunk]
    recommended_context_summary: str


class SourcesResponse(BaseModel):
    sources: list[RagSourceSummary]


class BuildContextOptionalParams(BaseModel):
    stack: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    sourceUrls: list[str] = Field(default_factory=list)
    repoPreference: str | None = None
    repoName: str | None = None
    repoUrl: str | None = None
    visibility: str | None = None
    demoPreference: str | None = None


class BuildContextRequest(BaseModel):
    projectId: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    rulesUrl: str | None = None
    referenceUrls: list[str] = Field(default_factory=list)
    optionalParams: BuildContextOptionalParams | None = None
    contextNeeded: list[str] = Field(default_factory=list)
    topK: int = Field(default=8, ge=1, le=25)


class BuildContextItem(BaseModel):
    item: str
    priority: Priority
    reason: str
    source: str


class ScopeWarningItem(BaseModel):
    item: str
    reason: str
    source: str


class EvidenceItem(BaseModel):
    source: str
    docType: DocType
    chunkId: str
    content: str
    score: float


class ResolvedTechStack(BaseModel):
    source: ResolvedTechStackSource
    items: list[str]
    requiredItems: list[str]
    defaultItems: list[str]
    reason: str


class BuildContextResponse(BaseModel):
    requiredDeliverables: list[BuildContextItem]
    allowedToolsAndAPIs: list[BuildContextItem]
    requiredRepositoryFormat: list[BuildContextItem]
    requiredDemoFormat: list[BuildContextItem]
    requiredTechStackPieces: list[BuildContextItem]
    agentBoundaries: dict[str, Any] = Field(default_factory=dict)
    resolvedTechStack: ResolvedTechStack
    scopeWarnings: list[ScopeWarningItem]
    evidence: list[EvidenceItem]
