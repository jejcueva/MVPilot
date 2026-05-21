from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.rag.url_utils import collect_source_urls


MAX_ADDITIONAL_URLS = 5
MAX_UPLOADED_FILES = 5
RepoVisibility = Literal["public", "private"]
RepoPreference = Literal["create_new_repo", "use_existing_repo"]
FlightStage = Literal[
    "preflight",
    "radar_scan",
    "flight_plan",
    "autopilot",
    "black_box",
    "landed",
    "failed",
]
FlightAgent = Literal["frontend", "orchestrator", "rag", "github", "black_box"]


class TaskStatus(str, Enum):
    STARTED = "started"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadedSourceFile(BaseModel):
    name: str = Field(min_length=1)
    content_type: str
    size_bytes: int = Field(ge=0)

    @field_validator("name", "content_type", mode="before")
    @classmethod
    def trim_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class UploadedSourceFileContent(UploadedSourceFile):
    content: bytes = Field(default=b"", repr=False, exclude=True)


class RunAgentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str | None = None
    idea: str = Field(min_length=1)
    repo_visibility: RepoVisibility = Field(
        default="public",
        validation_alias=AliasChoices("repo_visibility", "visibility"),
    )
    repo_preference: RepoPreference = Field(
        default="create_new_repo",
        validation_alias=AliasChoices("repo_preference", "repoPreference"),
    )
    repo_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("repo_name", "repoName"),
    )
    repo_description: str | None = Field(
        default=None,
        validation_alias=AliasChoices("repo_description", "repoDescription"),
    )
    repo_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("repo_url", "repoUrl"),
    )
    branch: str = "main"
    demo_mode: bool = False
    source: str | None = None
    primary_rules_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("primary_rules_url", "rules_url", "rulesUrl"),
    )
    rules_url: str | None = None
    additional_urls: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("additional_urls", "reference_urls", "referenceUrls"),
    )
    source_urls: list[str] = Field(default_factory=list)
    additional_files: list[UploadedSourceFile] = Field(default_factory=list)
    uploaded_file_contents: list[UploadedSourceFileContent] = Field(
        default_factory=list,
        exclude=True,
    )
    github_connected: bool = False
    github_connection_id: str | None = None
    target_users: str | None = Field(
        default=None,
        validation_alias=AliasChoices("target_users", "targetUsers"),
    )
    tech_stack_preference: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tech_stack_preference", "techStackPreference"),
    )
    required_features: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("required_features", "requiredFeatures"),
    )
    project_depth: str = Field(
        default="Advanced Project",
        validation_alias=AliasChoices("project_depth", "projectDepth"),
    )
    target_platform: str = Field(
        default="web app",
        validation_alias=AliasChoices("target_platform", "targetPlatform"),
    )
    use_openclaw_orchestration: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "use_openclaw_orchestration",
            "useOpenClawOrchestration",
        ),
    )

    @field_validator(
        "title",
        "idea",
        "source",
        "primary_rules_url",
        "rules_url",
        "repo_name",
        "repo_description",
        "repo_url",
        "branch",
        "github_connection_id",
        "target_users",
        "tech_stack_preference",
        "project_depth",
        "target_platform",
        mode="before",
    )
    @classmethod
    def trim_text(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("required_features", mode="before")
    @classmethod
    def coerce_required_features(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [part.strip() for part in stripped.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("additional_urls", mode="before")
    @classmethod
    def coerce_additional_urls(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("additional_urls")
    @classmethod
    def trim_additional_urls(cls, value: list[str]) -> list[str]:
        urls = [url.strip() for url in value if url.strip()]
        if len(urls) > MAX_ADDITIONAL_URLS:
            raise ValueError(f"At most {MAX_ADDITIONAL_URLS} additional URLs are supported")
        return urls

    @field_validator("additional_files", "uploaded_file_contents")
    @classmethod
    def limit_uploaded_files(
        cls,
        value: list[UploadedSourceFile] | list[UploadedSourceFileContent],
    ) -> list[UploadedSourceFile] | list[UploadedSourceFileContent]:
        if len(value) > MAX_UPLOADED_FILES:
            raise ValueError(f"At most {MAX_UPLOADED_FILES} uploaded files are supported")
        return value

    @model_validator(mode="after")
    def merge_source_urls(self) -> RunAgentRequest:
        merged = collect_source_urls(
            source_urls=self.source_urls,
            primary_rules_url=self.primary_rules_url,
            rules_url=self.rules_url,
            additional_urls=self.additional_urls,
        )
        self.source_urls = merged
        if merged and not self.primary_rules_url:
            self.primary_rules_url = merged[0]
        return self


class RunAgentResponse(BaseModel):
    task_id: str
    status: Literal["started"]


class TaskRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    title: str | None = None
    idea: str
    repo_visibility: RepoVisibility
    repo_preference: RepoPreference = "create_new_repo"
    repo_name: str | None = None
    repo_description: str | None = None
    repo_url: str | None = None
    branch: str = "main"
    demo_mode: bool
    source: str | None = None
    primary_rules_url: str | None = None
    additional_urls: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    additional_files: list[UploadedSourceFile] = Field(default_factory=list)
    github_connected: bool = False
    github_connection_id: str | None = None
    target_users: str | None = None
    tech_stack_preference: str | None = None
    required_features: list[str] = Field(default_factory=list)
    status: TaskStatus
    created_at: datetime
    updated_at: datetime


class AgentStep(BaseModel):
    project_id: str | None = None
    flight_stage: FlightStage | None = None
    agent: FlightAgent | None = None
    node_name: str
    status: str
    message: str
    model: str | None = None
    prompt_purpose: str | None = None
    model_mode: Literal["mock", "live", "partial", "degraded"] | None = None
    decision_trace: list[str] = Field(default_factory=list)
    timestamp: datetime


class ApprovalRecord(BaseModel):
    approval_id: str
    task_id: str
    proposed_action: str
    risk_level: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    approved_by: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class TaskDetailResponse(BaseModel):
    task: TaskRecord
    runtime: str = "langgraph"
    registered_tools: list[str] = Field(default_factory=list)
    openclaw_trace: list[dict[str, Any]] = Field(default_factory=list)
    agent_steps: list[AgentStep] = Field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    build_context: dict[str, Any] | None = None
    memory_matches: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    generated_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    graph_trace: list[AgentStep] = Field(default_factory=list)
    mvp_plan: dict[str, Any] | None = None
    build_timeline: list[dict[str, Any]] = Field(default_factory=list)
    mvp_validation: dict[str, Any] | None = None
    mvp_delivery: dict[str, Any] | None = None
    final_report: dict[str, Any] | None = None


class ApprovalDecisionRequest(BaseModel):
    task_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    decision: Literal["approved", "rejected"]
    approved_by: str = Field(min_length=1)

    @field_validator("task_id", "approval_id", "approved_by", mode="before")
    @classmethod
    def trim_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ApprovalDecisionResponse(BaseModel):
    task_id: str
    approval_id: str
    status: Literal["approved", "rejected"]
    approved_by: str
