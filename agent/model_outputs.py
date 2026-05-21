from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ModelOutputMode = Literal["mock", "live", "degraded", "partial"]


class TracedModelOutput(BaseModel):
    decision_trace: list[str] = Field(min_length=1)


class UserFlowStep(BaseModel):
    step: str
    screen: str
    action: str
    api: str | None = None


class ProjectRequirementsOutput(TracedModelOutput):
    target_users: str
    user_personas: list[str] = Field(min_length=1)
    core_features: list[str] = Field(min_length=3)
    advanced_features: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)
    project_depth: str = "Advanced Project"
    target_platform: str = "web app"
    project_archetype: str = "workflow"
    primary_entity: str = ""
    auth_required: bool = True
    database_required: bool = True
    data_entities: list[str] = Field(default_factory=list)
    user_flows: list[UserFlowStep] = Field(default_factory=list)
    api_routes: list[str] = Field(default_factory=list)
    mode: ModelOutputMode


class RecommendedStackOutput(TracedModelOutput):
    frontend: str
    backend: str
    database: str
    authentication: str
    aiModels: list[str] = Field(min_length=1)
    orchestration: list[str] = Field(min_length=1)
    ragRetrieval: str
    vectorStorage: str
    deployment: str
    testing: str
    reasonForChoices: list[str] = Field(min_length=1)
    hackathonRuleAlignment: list[str] = Field(min_length=1)
    rejectedAlternatives: list[str] = Field(default_factory=list)
    ruleConflicts: list[str] = Field(default_factory=list)
    mode: ModelOutputMode


class ArchitecturePlanOutput(TracedModelOutput):
    files: list[str] = Field(min_length=1)
    file_tree: list[str] = Field(default_factory=list)
    selected_stack: list[str] = Field(default_factory=list)
    architecture_overview: list[str] = Field(default_factory=list)
    frontend_architecture: list[str] = Field(default_factory=list)
    backend_architecture: list[str] = Field(default_factory=list)
    data_model: list[str] = Field(default_factory=list)
    api_design: list[str] = Field(default_factory=list)
    auth_design: list[str] = Field(default_factory=list)
    database_schema: list[str] = Field(default_factory=list)
    state_management: list[str] = Field(default_factory=list)
    integration_points: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    agent_assignments: list[str] = Field(default_factory=list)
    github_actions_needed: list[str] = Field(default_factory=list)
    generated_artifacts: list[str] = Field(default_factory=list)
    security_constraints: list[str] = Field(default_factory=list)
    test_plan: list[str] = Field(min_length=1)
    deployment_plan: list[str] = Field(default_factory=list)
    documentation_plan: list[str] = Field(default_factory=list)
    mode: ModelOutputMode


class GeneratedArtifactOutput(BaseModel):
    name: str
    kind: str
    summary: str
    content: str | None = None


class FileManifestOutput(TracedModelOutput):
    artifacts: list[GeneratedArtifactOutput] = Field(min_length=1)
    mode: ModelOutputMode


class FilePlanOutput(TracedModelOutput):
    artifacts: list[GeneratedArtifactOutput] = Field(min_length=1)
    mode: ModelOutputMode


class GeneratedFileOutput(TracedModelOutput):
    name: str
    kind: str
    summary: str
    content: str = Field(min_length=1)
    mode: ModelOutputMode


class BlockerAnalysisOutput(TracedModelOutput):
    blocker_type: str
    severity: Literal["low", "medium", "high", "critical"]
    recoverable: bool
    root_cause: str
    recovery_plan: list[str] = Field(min_length=1)


class FinalReadmeOutput(TracedModelOutput):
    title: str
    content: str
    setup_steps: list[str] = Field(min_length=1)


class WalkthroughOutput(TracedModelOutput):
    title: str
    content: str
    beats: list[str] = Field(min_length=1)


class PitchOutput(TracedModelOutput):
    title: str
    tagline: str
    content: str
    proof_points: list[str] = Field(min_length=1)


# Compatibility aliases for older imports while active code uses project language.
DemoPathStep = UserFlowStep
MvpScopeOutput = ProjectRequirementsOutput
RepoPlanOutput = ArchitecturePlanOutput
DemoScriptOutput = WalkthroughOutput
