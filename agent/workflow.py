from __future__ import annotations

import operator
import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.config import Settings
from agent.model_client import (
    DeterministicModelClient,
    ModelCallResult,
    ModelClient,
    NemotronModelClient,
)
from agent.model_outputs import (
    BlockerAnalysisOutput,
    DemoScriptOutput,
    FileManifestOutput,
    FinalReadmeOutput,
    MvpScopeOutput,
    PitchOutput,
    RecommendedStackOutput,
    RepoPlanOutput,
)
from agent.prompts import (
    build_blocker_analysis_prompt,
    build_demo_script_prompt,
    build_file_manifest_prompt,
    build_final_readme_prompt,
    build_pitch_prompt,
    build_plan_repo_prompt,
    build_stack_recommendation_prompt,
    build_scope_mvp_prompt,
)
from agent.stack_recommendation import (
    align_architecture_plan_with_recommended_stack,
    apply_recommended_stack_to_build_context,
    recommended_stack_summary,
    stack_items_from_recommended,
)
from agent.schemas import AgentStep
from agent.adapters import AuditAdapter, RagMemoryAdapter, ToolAdapter, InMemoryAuditAdapter, InMemoryToolAdapter
from agent.frontend_intake import (
    FrontendIntake,
    build_optional_params_from_frontend_intake,
    build_source_context,
)
from agent.github_oauth import GitHubConnectionService, GitHubOAuthError
from agent.live_adapters import LiveRagMemoryAdapter
from agent.idea_context import (
    features_from_context,
    project_title_from_context,
    target_users_from_context,
    tech_stack_from_context,
)
from agent.mvp_depth import enrich_mvp_scope
from agent.mvp_validation import build_delivery_report, validate_mvp_output
from agent.openclaw_orchestrator import OpenClawOrchestrator, _planned_file_path
from agent.orchestration_pipeline import (
    API_DESIGN,
    AUTH_AUTHORIZATION_DESIGN,
    BACKEND_ARCHITECTURE,
    BUILD_TEST_VALIDATION,
    CODE_IMPLEMENTATION,
    DATA_MODEL_DESIGN,
    DATABASE_SCHEMA_PLANNING,
    DEPLOYMENT_INSTRUCTIONS,
    DOCUMENTATION_GENERATION,
    DOMAIN_RESEARCH,
    FEATURE_SYSTEM_DESIGN,
    FILE_TREE_GENERATION,
    FINAL_PROJECT_REPORT,
    FRONTEND_ARCHITECTURE,
    GITHUB_REPO_EXPORT,
    IDEA_INTAKE,
    REFERENCE_URL_ANALYSIS,
    REQUIREMENT_EXPANSION,
    TESTING_STRATEGY,
    TECH_STACK_RECOMMENDATION,
    USER_GOAL_INTERPRETATION,
    agent_for_node,
    log_node_activity,
    record_phases,
)
from agent.project_generation import (
    artifact_groups,
    hydrate_file_manifest,
)
from tools.build_checker import merge_repo_health_scaffold
from agent.schemas import UploadedSourceFileContent

ListReducer = Annotated[list[dict[str, Any]], operator.add]
StepReducer = Annotated[list[AgentStep], operator.add]

NODE_FLIGHT_STAGE: dict[str, str] = {
    "receive_idea": "preflight",
    "exchange_github_code": "preflight",
    "retrieve_context": "radar_scan",
    "scope_mvp": "flight_plan",
    "recommend_stack": "flight_plan",
    "plan_repo": "flight_plan",
    "create_repo": "autopilot",
    "generate_files": "autopilot",
    "validate_mvp": "autopilot",
    "debug_generated_files": "autopilot",
    "commit_progress": "autopilot",
    "verify_build": "autopilot",
    "handle_blocker": "autopilot",
    "generate_final_package": "black_box",
    "remember_outcome": "black_box",
    "report_result": "landed",
    "failed": "failed",
}

NODE_AGENT: dict[str, str] = {
    "receive_idea": "orchestrator",
    "exchange_github_code": "github",
    "retrieve_context": "rag",
    "scope_mvp": "orchestrator",
    "plan_repo": "orchestrator",
    "create_repo": "github",
    "generate_files": "orchestrator",
    "validate_mvp": "orchestrator",
    "debug_generated_files": "orchestrator",
    "commit_progress": "github",
    "verify_build": "github",
    "handle_blocker": "orchestrator",
    "generate_final_package": "orchestrator",
    "remember_outcome": "black_box",
    "report_result": "orchestrator",
    "failed": "orchestrator",
}


class WorkflowState(TypedDict):
    task_id: str
    idea: str
    repo_visibility: Literal["public", "private"]
    demo_mode: bool
    source_urls: NotRequired[list[str]]
    runtime: str
    registered_tools: NotRequired[list[str]]
    openclaw_trace: ListReducer
    status: str
    nemotron_model: str
    mock_mode: bool
    blocker_recovered: bool
    agent_steps: StepReducer
    graph_trace: StepReducer
    retrieved_docs: ListReducer
    build_context: NotRequired[dict[str, Any]]
    frontend_intake: NotRequired[dict[str, Any]]
    uploaded_file_contents: NotRequired[list[dict[str, Any]]]
    memory_matches: ListReducer
    tool_calls: ListReducer
    generated_artifacts: ListReducer
    last_tool_result: NotRequired[dict[str, Any]]
    repo: NotRequired[dict[str, Any]]
    github_connection: NotRequired[dict[str, Any]]
    mvp_scope: NotRequired[dict[str, Any]]
    recommended_stack: NotRequired[dict[str, Any]]
    repo_plan: NotRequired[dict[str, Any]]
    file_manifest: NotRequired[dict[str, Any]]
    blocker_analysis: NotRequired[dict[str, Any]]
    final_readme: NotRequired[dict[str, Any]]
    demo_script: NotRequired[dict[str, Any]]
    pitch: NotRequired[dict[str, Any]]
    final_report: NotRequired[dict[str, Any] | None]
    failure_reason: NotRequired[str | None]
    mvp_plan: NotRequired[dict[str, Any]]
    project_plan: NotRequired[dict[str, Any]]
    agent_logs: NotRequired[list[dict[str, Any]]]
    project_agents: NotRequired[list[dict[str, Any]]]
    build_timeline: NotRequired[list[dict[str, Any]]]
    openclaw_pipeline: NotRequired[dict[str, Any]]
    model_modes: ListReducer
    file_manifest_mode: NotRequired[str]
    mvp_validation: NotRequired[dict[str, Any]]
    mvp_delivery: NotRequired[dict[str, Any]]



def build_initial_state(
    *,
    task_id: str,
    idea: str,
    repo_visibility: Literal["public", "private"],
    demo_mode: bool,
    settings: Settings,
    source_urls: list[str] | None = None,
    frontend_intake: dict[str, Any] | None = None,
    uploaded_file_contents: list[dict[str, Any]] | None = None,
) -> WorkflowState:
    normalized_intake = frontend_intake or FrontendIntake(idea=idea).model_dump()
    orchestrator = OpenClawOrchestrator(settings)
    return {
        "task_id": task_id,
        "idea": idea,
        "repo_visibility": repo_visibility,
        "demo_mode": demo_mode,
        "source_urls": list(source_urls or []),
        **orchestrator.initial_state_extras(),
        "openclaw_trace": [],
        "status": "started",
        "nemotron_model": settings.nemotron_model,
        "mock_mode": settings.mock_mode,
        "blocker_recovered": False,
        "agent_steps": [],
        "graph_trace": [],
        "retrieved_docs": [],
        "build_context": {},
        "frontend_intake": normalized_intake,
        "uploaded_file_contents": uploaded_file_contents or [],
        "memory_matches": [],
        "tool_calls": [],
        "generated_artifacts": [],
        "final_report": None,
        "failure_reason": None,
        "model_modes": [],
    }


def build_workflow(
    settings: Settings,
    *,
    model_client: ModelClient | None = None,
    audit: AuditAdapter | None = None,
    retrieval: RagMemoryAdapter | None = None,
    tools: ToolAdapter | None = None,
    github_connections: GitHubConnectionService | None = None,
):
    active_audit = audit or InMemoryAuditAdapter(model_name=settings.nemotron_fast_model)
    active_model_client = model_client or _build_default_model_client(settings)
    enforce_live_nemotron = model_client is None and _live_nemotron_workflow(settings)
    active_retrieval = retrieval or LiveRagMemoryAdapter()
    active_tools = tools or InMemoryToolAdapter()
    orchestrator = OpenClawOrchestrator(settings)

    def append_step(
        *,
        state: WorkflowState | None = None,
        node_name: str,
        message: str,
        decision_trace: list[str],
        status: str = "completed",
    ) -> dict[str, list[AgentStep]]:
        project_id = state["task_id"] if state else None
        flight_stage = NODE_FLIGHT_STAGE.get(node_name)
        agent = NODE_AGENT.get(node_name)
        step = active_audit.write_audit_log(
            node_name=node_name,
            message=message,
            decision_trace=decision_trace,
            status=status,
            project_id=project_id,
            flight_stage=flight_stage,
            agent=agent,
        )
        return {"agent_steps": [step], "graph_trace": [step]}

    def append_model_step(
        *,
        state: WorkflowState | None = None,
        node_name: str,
        message: str,
        result: ModelCallResult,
        status: str = "completed",
        prompt_purpose: str | None = None,
        extra_trace: list[str] | None = None,
    ) -> dict[str, list[AgentStep]]:
        step = AgentStep(
            project_id=state["task_id"] if state else None,
            flight_stage=NODE_FLIGHT_STAGE.get(node_name),
            agent=NODE_AGENT.get(node_name),
            node_name=node_name,
            status=status,
            message=message,
            model=result.model,
            prompt_purpose=prompt_purpose or result.purpose,
            model_mode=result.mode,
            decision_trace=[
                *result.output.decision_trace,
                *(extra_trace or []),
            ],
            timestamp=datetime.now(UTC),
        )
        return {"agent_steps": [step], "graph_trace": [step]}

    def receive_idea(state: WorkflowState) -> dict[str, Any]:
        intake = state.get("frontend_intake") or {}
        return {
            **append_step(
                state=state,
                node_name="receive_idea",
                message=f"{agent_for_node('receive_idea')} captured the project idea and intake brief.",
                decision_trace=[
                    "Accepted the trimmed idea payload.",
                    f"Runtime: {state.get('runtime', 'langgraph')}.",
                    f"Target users: {intake.get('targetUsers') or 'not specified'}.",
                    f"Tech preference: {intake.get('techStackPreference') or 'auto-selected'}.",
                ],
            ),
            **orchestrator.record_phase(
                state,
                phase_id=IDEA_INTAKE,
                status="completed",
                detail="Parsed the startup idea, optional reference URL, and feature constraints.",
            ),
            **orchestrator.update_mvp_plan(
                state,
                idea=state["idea"],
                target_users=intake.get("targetUsers"),
                tech_stack_preference=intake.get("techStackPreference"),
                reference_url=intake.get("primaryRulesUrl"),
                project_depth=intake.get("projectDepth") or intake.get("project_depth"),
                target_platform=intake.get("targetPlatform") or intake.get("target_platform"),
            ),
            **log_node_activity(
                state=state,
                node_name="receive_idea",
                stage_id=IDEA_INTAKE,
                message="Project intake recorded.",
            ),
        }

    async def exchange_github_code(state: WorkflowState) -> dict[str, Any]:
        frontend_intake = FrontendIntake.model_validate(
            state.get("frontend_intake") or {"idea": state["idea"]}
        )
        connection_id = frontend_intake.githubConnectionId

        if settings.mock_mode:
            return {
                **append_step(
                    state=state,
                    node_name="exchange_github_code",
                    message="Test mode skipped live GitHub OAuth exchange.",
                    decision_trace=[
                        "Test mode keeps the workflow deterministic.",
                        "No GitHub code or token is required for in-memory repository actions.",
                    ],
                ),
                "github_connection": {
                    "status": "mock_skipped",
                    "connection_id": connection_id,
                    "login": None,
                },
            }

        if not connection_id:
            reason = (
                "GitHub is not connected. Use Connect GitHub on the website before launching "
                "so the backend receives a github_connection_id."
            )
            return {
                **append_step(
                    state=state,
                    node_name="exchange_github_code",
                    status="failed",
                    message=reason,
                    decision_trace=[
                        "Live mode requires a backend-issued github_connection_id.",
                        "Stopped before RAG planning because repo actions need an authenticated user.",
                    ],
                ),
                "status": "failed",
                "failure_reason": reason,
                "github_connection": {
                    "status": "missing",
                    "connection_id": None,
                    "login": None,
                },
            }

        if github_connections is None:
            reason = "GitHub connection service is not configured."
            return {
                **append_step(
                    state=state,
                    node_name="exchange_github_code",
                    status="failed",
                    message=reason,
                    decision_trace=[
                        "Live mode cannot exchange a GitHub connection without the service.",
                        "Stopped before repo creation.",
                    ],
                ),
                "status": "failed",
                "failure_reason": reason,
            }

        try:
            auth = await github_connections.exchange_for_workflow(
                connection_id,
                task_id=state["task_id"],
            )
            configure_github = getattr(active_tools, "set_github_config", None)
            if configure_github is None:
                raise GitHubOAuthError("GitHub tool adapter cannot accept per-task credentials.")
            configure_github(auth.config)
        except GitHubOAuthError as exc:
            reason = str(exc) or "GitHub OAuth exchange failed."
            return {
                **append_step(
                    state=state,
                    node_name="exchange_github_code",
                    status="failed",
                    message=reason,
                    decision_trace=[
                        "GitHub OAuth exchange did not complete.",
                        "No repository action was attempted.",
                    ],
                ),
                "status": "failed",
                "failure_reason": reason,
            }

        return {
            **append_step(
                state=state,
                node_name="exchange_github_code",
                message="Exchanged the backend GitHub connection for live repo credentials.",
                decision_trace=[
                    "Validated the backend-issued github_connection_id.",
                    f"Configured GitHub tools for connected user {auth.login}.",
                    "Kept the access token out of workflow state and API responses.",
                ],
            ),
            "github_connection": {
                "status": "exchanged",
                "connection_id": auth.connection_id,
                "login": auth.login,
                "scopes": auth.scopes,
            },
        }

    async def retrieve_context(state: WorkflowState) -> dict[str, Any]:
        frontend_intake = FrontendIntake.model_validate(
            state.get("frontend_intake") or {"idea": state["idea"]}
        )
        source_urls = list(state.get("source_urls") or [])
        index_summary: dict[str, Any] = {
            "documentsLoaded": 0,
            "chunksCreated": 0,
        }
        if source_urls:
            index_summary = await active_retrieval.index_source_urls(source_urls)

        optional_params = build_optional_params_from_frontend_intake(frontend_intake) or {}
        if source_urls:
            optional_params = {**optional_params, "sourceUrls": source_urls}
        optional_params = optional_params or None

        build_context = await active_retrieval.retrieve_build_context(
            state["task_id"],
            state["idea"],
            optional_params=optional_params,
            rules_url=frontend_intake.primaryRulesUrl,
            reference_urls=frontend_intake.additionalUrls,
            context_needed=[
                "required_deliverables",
                "allowed_tools_apis",
                "required_repository_format",
                "required_demo_format",
                "required_tech_stack_pieces",
                "hackathon_rules",
                "nvidia_model_usage",
                "security_constraints",
                "agent_boundaries",
                "scope_warnings",
            ],
            top_k=8,
        )
        uploaded_files = [
            UploadedSourceFileContent.model_validate(file)
            for file in state.get("uploaded_file_contents", [])
        ]
        source_context = await build_source_context(
            frontend_intake,
            uploaded_files=uploaded_files,
        )
        build_context = {
            **build_context,
            "frontendIntake": frontend_intake.model_dump(),
            "sourceContext": source_context,
        }
        docs = (
            await active_retrieval.retrieve_hackathon_context(state["idea"])
            + await active_retrieval.retrieve_nvidia_context(state["idea"])
        )
        memories = await active_retrieval.find_similar_builds(state["idea"])

        evidence_count = len(build_context.get("evidence", []))
        context_mode = build_context.get("mode", "unknown")
        critical_count = sum(
            1
            for field in (
                "requiredDeliverables",
                "allowedToolsAndAPIs",
                "requiredRepositoryFormat",
                "requiredDemoFormat",
                "requiredTechStackPieces",
            )
            for item in build_context.get(field, [])
            if item.get("priority") == "critical"
        )

        return {
            **append_step(
                state=state,
                node_name="retrieve_context",
                message="Retrieved structured RAG build context for the orchestrator.",
                decision_trace=[
                    f"Loaded build context via RAG adapter (mode={context_mode}).",
                    (
                        f"Indexed {index_summary.get('documentsLoaded', 0)} orchestrator URL document(s) "
                        f"({index_summary.get('chunksCreated', 0)} chunks) before retrieval."
                        if source_urls
                        else "No orchestrator source URLs to index (using ingested corpus and RAG_SCRAPE_URLS only)."
                    ),
                    f"Structured constraints: {len(build_context.get('requiredDeliverables', []))} deliverables, "
                    f"{len(build_context.get('requiredTechStackPieces', []))} stack items, "
                    f"{critical_count} critical priorities, {evidence_count} evidence chunks.",
                    f"Frontend intake title: {frontend_intake.title or 'not provided'}.",
                    f"Source context warnings: {source_context['sourceCounts']['warnings']}.",
                    f"Augmented with {len(docs)} retrieved doc matches and {len(memories)} memory matches.",
                ],
            ),
            "build_context": build_context,
            "retrieved_docs": docs,
            "memory_matches": memories,
            **record_phases(
                orchestrator,
                state,
                (
                    DOMAIN_RESEARCH,
                    "completed",
                    f"Retrieved {evidence_count} evidence chunks and {len(memories)} memory matches.",
                    [str(doc.get("source", "")) for doc in docs[:4] if doc.get("source")],
                ),
                (
                    REFERENCE_URL_ANALYSIS,
                    "completed",
                    f"Analyzed intake sources with {source_context['sourceCounts']['warnings']} warning(s).",
                    list(source_urls[:4]),
                ),
            ),
            **log_node_activity(
                state=state,
                node_name="retrieve_context",
                stage_id=DOMAIN_RESEARCH,
                message="Research/RAG Agent enriched build context.",
            ),
        }

    async def scope_mvp(state: WorkflowState) -> dict[str, Any]:
        result = await active_model_client.complete_structured(
            purpose="scope_mvp",
            model=settings.nemotron_model,
            prompt=build_scope_mvp_prompt(
                idea=state["idea"],
                build_context=state.get("build_context", {}),
                memory_matches=state["memory_matches"],
            ),
            response_model=MvpScopeOutput,
            max_tokens=settings.nemotron_planning_max_tokens,
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        _require_live_nemotron_result(
            result, "scope_mvp", enforced=enforce_live_nemotron, settings=settings
        )
        scope = enrich_mvp_scope(
            result.output.model_dump(),
            idea=state["idea"],
            intake=state.get("frontend_intake"),
        )
        project_plan = orchestrator.compose_project_plan(
            idea=state["idea"],
            intake=state.get("frontend_intake"),
            mvp_scope=scope,
            repo_plan=state.get("repo_plan"),
            build_context=state.get("build_context"),
            recommended_stack=state.get("recommended_stack"),
        )
        return {
            **append_model_step(
                state=state,
                node_name="scope_mvp",
                message="Expanded requirements for a full complex project.",
                result=result,
            ),
            "mvp_scope": scope,
            "mvp_plan": project_plan,
            "project_plan": project_plan,
            "model_modes": [result.mode],
            **record_phases(
                orchestrator,
                state,
                (
                    REQUIREMENT_EXPANSION,
                    "completed",
                    f"Defined {len(project_plan.get('features') or [])} core project features.",
                    [str(item) for item in (project_plan.get("features") or [])[:6]],
                ),
                (
                    USER_GOAL_INTERPRETATION,
                    "completed",
                    f"Target users: {project_plan.get('target_users') or 'inferred from idea'}.",
                    [],
                ),
                (
                    FEATURE_SYSTEM_DESIGN,
                    "completed",
                    f"Archetype: {scope.get('project_archetype') or scope.get('vertical_pack')}.",
                    list(scope.get("api_routes") or [])[:6],
                ),
            ),
            **log_node_activity(
                state=state,
                node_name="scope_mvp",
                stage_id=REQUIREMENT_EXPANSION,
                message="Product Strategist Agent expanded project requirements.",
            ),
        }

    async def recommend_stack(state: WorkflowState) -> dict[str, Any]:
        build_context = dict(state.get("build_context") or {})
        result = await active_model_client.complete_structured(
            purpose="recommend_stack",
            model=settings.nemotron_model,
            prompt=build_stack_recommendation_prompt(
                idea=state["idea"],
                project_requirements=state.get("mvp_scope", {}),
                build_context=build_context,
            ),
            response_model=RecommendedStackOutput,
            max_tokens=settings.nemotron_max_tokens_for("recommend_stack"),
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        _require_live_nemotron_result(
            result, "recommend_stack", enforced=enforce_live_nemotron, settings=settings
        )
        recommended = result.output.model_dump()
        build_context = apply_recommended_stack_to_build_context(build_context, recommended)
        project_plan = orchestrator.compose_project_plan(
            idea=state["idea"],
            intake=state.get("frontend_intake"),
            mvp_scope=state.get("mvp_scope"),
            repo_plan=state.get("repo_plan"),
            build_context=build_context,
            recommended_stack=recommended,
        )
        stack_summary = recommended_stack_summary(recommended)
        stack_artifacts = stack_items_from_recommended(recommended)[:8]
        return {
            **append_model_step(
                state=state,
                node_name="recommend_stack",
                message="Stack Selector Agent recommended a project-specific tech stack.",
                result=result,
            ),
            "recommended_stack": recommended,
            "build_context": build_context,
            "mvp_plan": project_plan,
            "project_plan": project_plan,
            "model_modes": [result.mode],
            **record_phases(
                orchestrator,
                state,
                (
                    TECH_STACK_RECOMMENDATION,
                    "completed",
                    stack_summary[:240],
                    stack_artifacts,
                ),
            ),
            **log_node_activity(
                state=state,
                node_name="recommend_stack",
                stage_id=TECH_STACK_RECOMMENDATION,
                message="Stack Selector Agent aligned stack with hackathon rules and project scope.",
            ),
        }

    async def plan_repo(state: WorkflowState) -> dict[str, Any]:
        result = await active_model_client.complete_structured(
            purpose="plan_repo",
            model=settings.nemotron_model,
            prompt=build_plan_repo_prompt(
                idea=state["idea"],
                project_requirements=state.get("mvp_scope", {}),
                build_context=state.get("build_context", {}),
            ),
            response_model=RepoPlanOutput,
            max_tokens=settings.nemotron_max_tokens_for("plan_repo"),
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        _require_live_nemotron_result(
            result, "plan_repo", enforced=enforce_live_nemotron, settings=settings
        )
        plan = align_architecture_plan_with_recommended_stack(
            result.output.model_dump(),
            state.get("recommended_stack"),
        )
        project_plan = orchestrator.compose_project_plan(
            idea=state["idea"],
            intake=state.get("frontend_intake"),
            mvp_scope=state.get("mvp_scope"),
            repo_plan=plan,
            build_context=state.get("build_context"),
            recommended_stack=state.get("recommended_stack"),
        )
        timeline_updates = record_phases(
            orchestrator,
            state,
            (
                DATA_MODEL_DESIGN,
                "completed",
                "Designed entities, relationships, and persistence boundaries.",
                list(plan.get("data_model") or [])[:6],
            ),
            (
                API_DESIGN,
                "completed",
                "Mapped REST routes and service contracts.",
                list(plan.get("api_design") or plan.get("api_routes") or [])[:6],
            ),
            (
                FRONTEND_ARCHITECTURE,
                "completed",
                "Planned pages, components, and client state.",
                list(plan.get("frontend_architecture") or [])[:6],
            ),
            (
                BACKEND_ARCHITECTURE,
                "completed",
                "Planned services, modules, and integrations.",
                list(plan.get("backend_architecture") or [])[:6],
            ),
            (
                AUTH_AUTHORIZATION_DESIGN,
                "completed",
                "Defined authentication and authorization approach.",
                list(plan.get("auth_design") or [])[:6],
            ),
            (
                DATABASE_SCHEMA_PLANNING,
                "completed",
                "Outlined SQL schema and migration strategy.",
                list(plan.get("database_schema") or [])[:6],
            ),
        )
        plan_message = "Planned full project architecture and repository layout."
        if result.mode != "live":
            reason = result.fallback_reason or "Nemotron unavailable"
            plan_message = (
                f"Planned architecture via {result.mode} fallback ({reason}). "
                "Continuing pipeline; file manifest still requires live Nemotron when configured."
            )
        return {
            **append_model_step(
                state=state,
                node_name="plan_repo",
                message=plan_message,
                result=result,
            ),
            "repo_plan": plan,
            "mvp_plan": project_plan,
            "project_plan": project_plan,
            "model_modes": [result.mode],
            **timeline_updates,
        }

    def create_repo(state: WorkflowState) -> dict[str, Any]:
        frontend_intake = FrontendIntake.model_validate(
            state.get("frontend_intake") or {"idea": state["idea"]}
        )
        repo_name = frontend_intake.repoName or _repo_name_from_url(frontend_intake.repoUrl)
        repo_description = (
            frontend_intake.repoDescription
            or frontend_intake.title
            or f"Generated MVPilot project for: {state['idea'][:120]}"
        )
        tool_call = active_tools.create_repo(
            task_id=state["task_id"],
            visibility=state["repo_visibility"],
            repo_preference=frontend_intake.repoPreference,
            repo_name=repo_name,
            repo_description=repo_description,
            repo_url=frontend_intake.repoUrl,
        )
        repo_step_status = "completed" if tool_call["status"] == "success" else "failed"
        update: dict[str, Any] = {
            **append_step(
                state=state,
                node_name="create_repo",
                status=repo_step_status,
                message=tool_call["summary"],
                decision_trace=[
                    (
                        "Used mock GitHub adapter instead of a live API call."
                        if state["mock_mode"]
                        else "Used the connected GitHub account for the requested repo action."
                    ),
                    f"Repository preference: {frontend_intake.repoPreference}.",
                    *(
                        [
                            f"Repository target: {tool_call.get('repo', {}).get('name')}.",
                            tool_call.get("summary", ""),
                        ]
                        if tool_call["status"] == "success"
                        else [
                            "Repository creation failed; workflow stopped before file generation.",
                        ]
                    ),
                ],
            ),
            "repo": tool_call["repo"],
            "tool_calls": [tool_call],
            "openclaw_trace": _openclaw_trace_from_tool_call(tool_call),
            "last_tool_result": tool_call,
        }
        if tool_call["status"] != "success":
            update["status"] = "failed"
            update["failure_reason"] = tool_call["summary"]
        else:
            update.update(
                orchestrator.record_phase(
                    state,
                    phase_id=GITHUB_REPO_EXPORT,
                    status="completed",
                    detail=tool_call["summary"],
                    artifacts=[str(tool_call.get("repo", {}).get("name") or repo_name)],
                )
            )
        return update

    async def generate_files(state: WorkflowState) -> dict[str, Any]:
        mvp_scope = state.get("mvp_scope") or {}
        if not settings.mock_mode and not settings.nvidia_configured:
            raise RuntimeError(
                "Live Nemotron file_manifest requires NVIDIA_API_KEY. "
                "Set the key or use ADAPTER_MODE=mock for deterministic runs."
            )
        file_model = settings.nemotron_model
        result = await active_model_client.complete_structured(
            purpose="file_manifest",
            model=file_model,
            prompt=build_file_manifest_prompt(
                idea=state["idea"],
                project_requirements=mvp_scope,
                architecture_plan=state.get("repo_plan", {}),
                build_context=state.get("build_context", {}),
            ),
            response_model=FileManifestOutput,
            max_tokens=settings.nemotron_file_manifest_max_tokens,
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        _require_live_nemotron_result(
            result, "file_manifest", enforced=enforce_live_nemotron, settings=settings
        )
        manifest = result.output.model_dump()
        build_context = state.get("build_context", {})
        intake = state.get("frontend_intake") or build_context.get("frontendIntake", {})
        manifest["artifacts"] = hydrate_file_manifest(
            manifest.get("artifacts") or [],
            idea=state["idea"],
            title=project_title_from_context(idea=state["idea"], intake=intake),
            resolved_stack=_resolved_stack_summary(build_context),
            architecture_plan=state.get("repo_plan"),
            source_warnings=_source_warnings(build_context),
            target_users=target_users_from_context(intake),
            required_features=features_from_context(
                idea=state["idea"],
                intake=intake,
                mvp_scope=mvp_scope,
                repo_plan=state.get("repo_plan"),
            ),
            tech_stack_preference=tech_stack_from_context(
                intake,
                state.get("repo_plan"),
            ),
            project_requirements=mvp_scope,
        )
        if not manifest["artifacts"]:
            raise RuntimeError(
                "Live Nemotron file_manifest returned no usable artifacts."
            )
        groups = artifact_groups(manifest["artifacts"])
        artifacts = [
            {
                **artifact,
                "mock_mode": result.mode != "live",
                "summary": (
                    artifact["summary"]
                    if result.mode == "live"
                    else f"{result.mode.title()} mode: {artifact['summary']}"
                ),
            }
            for artifact in manifest["artifacts"]
        ]
        timeline_updates = orchestrator.record_phase(
            state,
            phase_id=FILE_TREE_GENERATION,
            status="completed",
            detail=f"Planned {len(manifest['artifacts'])} repository paths for the project.",
            artifacts=[artifact["name"] for artifact in manifest["artifacts"][:8]],
        )
        timeline_updates["build_timeline"] = record_phases(
            orchestrator,
            {**state, "build_timeline": timeline_updates["build_timeline"]},
            (
                CODE_IMPLEMENTATION,
                "completed",
                f"Generated {len(groups['frontend'])} frontend and {len(groups['backend'])} backend file(s).",
                (groups["frontend"] + groups["backend"])[:8],
            ),
            (
                DOCUMENTATION_GENERATION,
                "completed",
                f"Added {len(groups['docs']) + len(groups['tests'])} documentation and test artifacts.",
                (groups["docs"] + groups["tests"])[:8],
            ),
        )["build_timeline"]
        return {
            **append_model_step(
                state=state,
                node_name="generate_files",
                message="Generated runnable frontend, backend, database, test, docs, and idea-specific walkthrough files.",
                result=result,
            ),
            "generated_artifacts": artifacts,
            "file_manifest": manifest,
            "file_manifest_mode": result.mode,
            "model_modes": [result.mode],
            **timeline_updates,
        }

    def validate_mvp(state: WorkflowState) -> dict[str, Any]:
        frontend_intake = state.get("frontend_intake") or {}
        original_modes = list(state.get("model_modes") or [])
        modes = original_modes[:]
        enriched_scope = enrich_mvp_scope(
            state.get("mvp_scope") or {},
            idea=state["idea"],
            intake=frontend_intake,
        )
        artifacts = list(state.get("generated_artifacts", []))
        validation = validate_mvp_output(
            idea=state["idea"],
            intake=frontend_intake,
            mvp_scope=enriched_scope,
            repo_plan=state.get("repo_plan"),
            generated_artifacts=artifacts,
            model_modes=modes,
            require_live_manifest=enforce_live_nemotron,
            manifest_model_mode=state.get("file_manifest_mode"),
        )

        delivery = build_delivery_report(
            idea=state["idea"],
            intake=frontend_intake,
            mvp_scope=state.get("mvp_scope"),
            validation=validation,
            model_modes=modes,
            generated_artifacts=artifacts,
        )
        status = "completed" if validation["passed"] else "failed"
        if validation["passed"]:
            message = "Project output validated against requirements and architecture."
        else:
            failed_checks = [
                check["detail"]
                for check in validation.get("checks", [])
                if not check.get("passed")
            ]
            message = failed_checks[0] if failed_checks else (
                "Project validation failed before GitHub export."
            )
        update = {
            **append_step(
                state=state,
                node_name="validate_mvp",
                status=status,
                message=message,
                decision_trace=[
                    f"Validation passed: {validation['passed']}",
                    f"Model modes used: {', '.join(modes) or 'unknown'}",
                    "Validated Nemotron manifest only (no scaffold merge or repair).",
                    *(validation.get("warnings") or [])[:3],
                ],
            ),
            "mvp_validation": validation,
            "mvp_delivery": delivery,
            "model_modes": [mode for mode in modes if mode not in original_modes],
            **orchestrator.record_phase(
                state,
                phase_id=TESTING_STRATEGY,
                status=status,
                detail=message,
                artifacts=[check["name"] for check in validation.get("checks", []) if check.get("passed")][:8],
            ),
        }
        if not validation["passed"]:
            update["status"] = "failed"
            update["failure_reason"] = message
        return update

    def debug_generated_files(state: WorkflowState) -> dict[str, Any]:
        debug_report = _debug_generated_artifacts(state["generated_artifacts"])
        debug_artifact = {
            "name": "docs/DEBUG_REPORT.md",
            "kind": "markdown",
            "mock_mode": False,
            "summary": "Pre-commit debug report for generated files.",
            "content": debug_report["content"],
        }
        issue_count = len(debug_report["issues"])
        warning_count = len(debug_report["warnings"])
        return {
            **append_step(
                state=state,
                node_name="debug_generated_files",
                status="completed",
                message=(
                    "Ran pre-commit generated-file debug checks."
                    if issue_count == 0
                    else "Ran pre-commit generated-file debug checks with issues."
                ),
                decision_trace=[
                    f"Checked {len(state['generated_artifacts'])} generated artifact(s).",
                    f"Found {issue_count} issue(s) and {warning_count} warning(s).",
                    "Added docs/DEBUG_REPORT.md to make the debug pass visible in the repo.",
                ],
            ),
            "generated_artifacts": [*state["generated_artifacts"], debug_artifact],
        }

    def commit_progress(state: WorkflowState) -> dict[str, Any]:
        repo_name = state.get("repo", {}).get("name") or f"mvpilot-generated-{state['task_id'][:8]}"
        files = _files_from_generated_artifacts(state["generated_artifacts"])
        frontend_intake = state.get("frontend_intake") or {}
        files = merge_repo_health_scaffold(
            files,
            idea=state["idea"],
            title=project_title_from_context(idea=state["idea"], intake=frontend_intake),
        )
        tool_call = active_tools.commit_files(
            repo_name=repo_name,
            files=files,
            message="Add generated MVPilot project package",
        )
        update: dict[str, Any] = {
            **append_step(
                state=state,
                node_name="commit_progress",
                message=tool_call["summary"],
                decision_trace=[
                    f"Committed {len(files)} generated package file(s).",
                    (
                        "Recorded a deterministic commit SHA for dashboard traceability."
                        if state["mock_mode"]
                        else "Committed files using the connected GitHub account."
                    ),
                ],
            ),
            "tool_calls": [tool_call],
            "openclaw_trace": _openclaw_trace_from_tool_call(tool_call),
            "last_tool_result": tool_call,
        }
        if tool_call["status"] != "success":
            update["status"] = "failed"
            update["failure_reason"] = tool_call["summary"]
        else:
            update.update(
                orchestrator.record_phase(
                state,
                phase_id=GITHUB_REPO_EXPORT,
                status="running",
                detail=tool_call["summary"],
                artifacts=[_planned_file_path(item) for item in files[:8] if item],
                )
            )
        return update

    def verify_build(state: WorkflowState) -> dict[str, Any]:
        repo_name = state.get("repo", {}).get("name")
        tool_call = active_tools.verify_build(
            recovered=state["blocker_recovered"],
            repo_name=repo_name,
        )
        status = "completed" if tool_call["status"] == "success" else "blocked"
        update: dict[str, Any] = {
            **append_step(
                state=state,
                node_name="verify_build",
                status=status,
                message=tool_call["summary"],
                decision_trace=[
                    (
                        "Ran deterministic mock build verification."
                        if state["mock_mode"]
                        else "Ran repository health verification against committed files."
                    ),
                    "Routed recoverable failures through the blocker handler.",
                ],
            ),
            "tool_calls": [tool_call],
            "openclaw_trace": _openclaw_trace_from_tool_call(tool_call),
            "last_tool_result": tool_call,
        }
        phase_status = "completed" if tool_call["status"] == "success" else "failed"
        phase_detail = (
            tool_call["summary"]
            if tool_call["status"] == "success"
            else (
                tool_call.get("error")
                or tool_call.get("summary")
                or "Generated repository health check failed."
            )
        )
        if tool_call["status"] != "success":
            update["failure_reason"] = phase_detail
        update.update(
            orchestrator.record_phase(
                state,
                phase_id=BUILD_TEST_VALIDATION,
                status=phase_status,
                detail=phase_detail,
            )
        )
        return update

    async def handle_blocker(state: WorkflowState) -> dict[str, Any]:
        result = await active_model_client.complete_structured(
            purpose="blocker_analysis",
            model=settings.nemotron_model,
            prompt=build_blocker_analysis_prompt(
                idea=state["idea"],
                tool_result=state.get("last_tool_result", {}),
            ),
            response_model=BlockerAnalysisOutput,
            max_tokens=900,
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        _require_live_nemotron_result(
            result, "blocker_analysis", enforced=enforce_live_nemotron, settings=settings
        )
        blocker_analysis = result.output.model_dump()
        tool_call = active_tools.recover_build()
        return {
            **append_model_step(
                state=state,
                node_name="handle_blocker",
                message="Applied an idea-specific recovery for the repository health blocker.",
                result=result,
            ),
            "blocker_recovered": True,
            "blocker_analysis": blocker_analysis,
            "tool_calls": [tool_call],
            "openclaw_trace": _openclaw_trace_from_tool_call(tool_call),
            "last_tool_result": tool_call,
        }

    async def generate_final_package(state: WorkflowState) -> dict[str, Any]:
        readme_result = await active_model_client.complete_structured(
            purpose="final_readme",
            model=settings.nemotron_model,
            prompt=build_final_readme_prompt(
                idea=state["idea"],
                project_requirements=state.get("mvp_scope", {}),
                architecture_plan=state.get("repo_plan", {}),
                generated_artifacts=state["generated_artifacts"],
                build_context=state.get("build_context", {}),
            ),
            response_model=FinalReadmeOutput,
            max_tokens=1400,
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        demo_result = await active_model_client.complete_structured(
            purpose="demo_script",
            model=settings.nemotron_model,
            prompt=build_demo_script_prompt(
                idea=state["idea"],
                blocker_analysis=state.get("blocker_analysis"),
                build_context=state.get("build_context", {}),
            ),
            response_model=DemoScriptOutput,
            max_tokens=1100,
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        pitch_result = await active_model_client.complete_structured(
            purpose="pitch",
            model=settings.nemotron_model,
            prompt=build_pitch_prompt(
                idea=state["idea"],
                final_readme=readme_result.output.model_dump(),
                walkthrough=demo_result.output.model_dump(),
                build_context=state.get("build_context", {}),
            ),
            response_model=PitchOutput,
            max_tokens=1100,
            reasoning_effort=settings.nemotron_reasoning_effort,
        )
        for label, model_result in (
            ("final_readme", readme_result),
            ("demo_script", demo_result),
            ("pitch", pitch_result),
        ):
            _require_live_nemotron_result(
                model_result, label, enforced=enforce_live_nemotron, settings=settings
            )
        package_mode = _package_mode([readme_result, demo_result, pitch_result])
        readme = readme_result.output.model_dump()
        demo_script = demo_result.output.model_dump()
        pitch = pitch_result.output.model_dump()
        source_warnings = _source_warnings(state.get("build_context", {}))
        last_commit = _last_tool_call(state, "github.commit_files")
        repo = state.get("repo") or {}
        commit_url = last_commit.get("commit_url")
        links = {
            "repoUrl": repo.get("url"),
            "commitUrl": commit_url,
            "branch": "main",
            "buildLogPath": "docs/BUILD_LOG.md",
            "architectureDocPath": "docs/ARCHITECTURE.md",
            "demoScriptPath": "demo/demo_script.md",
        }
        final_report = {
            "status": "completed",
            "mode": package_mode,
            "model": settings.nemotron_model,
            "repo": repo,
            "links": links,
            "github_result": last_commit or None,
            "summary": (
                f"{package_mode.title()} orchestration produced a scoped MVP package with "
                "retrieval context, generated artifacts, validation results, and GitHub delivery proof."
            ),
            "readme": readme,
            "demo_script": demo_script,
            "pitch": pitch,
            "blocker_analysis": state.get("blocker_analysis"),
            "source_warnings": source_warnings,
            "artifact_count": len(state["generated_artifacts"]) + 1,
        }
        final_artifact = {
            "name": "final_report.json",
            "kind": "json",
            "mock_mode": package_mode != "live",
            "summary": (
                f"{package_mode.title()} mode: generated final package report"
                f" with {len(source_warnings)} source warnings."
            ),
            "content": json_dumps_safe(final_report),
        }
        combined_result = ModelCallResult(
            output=pitch_result.output,
            model=settings.nemotron_model,
            purpose="final_package",
            mode=package_mode,
            latency_ms=(
                readme_result.latency_ms
                + demo_result.latency_ms
                + pitch_result.latency_ms
            ),
            fallback_reason=(
                readme_result.fallback_reason
                or demo_result.fallback_reason
                or pitch_result.fallback_reason
            ),
        )
        timeline_updates = orchestrator.record_phase(
            state,
            phase_id=DEPLOYMENT_INSTRUCTIONS,
            status="completed",
            detail="Added docs/DEPLOY.md with deployment instructions for the generated repo.",
            artifacts=["docs/DEPLOY.md"],
        )
        timeline_updates["build_timeline"] = record_phases(
            orchestrator,
            {**state, "build_timeline": timeline_updates["build_timeline"]},
            (
                DOCUMENTATION_GENERATION,
                "completed",
                "Packaged README, walkthrough, and pitch documentation.",
                ["README.md", "demo/demo_script.md"],
            ),
            (
                FINAL_PROJECT_REPORT,
                "completed",
                "Published final project report and landing-zone links.",
                ["final_report.json", links.get("buildLogPath") or "docs/BUILD_LOG.md"],
            ),
        )["build_timeline"]
        return {
            **append_model_step(
                state=state,
                node_name="generate_final_package",
                message="Generated the final package and report.",
                result=combined_result,
                prompt_purpose="final_package",
                extra_trace=[
                    "README synthesis completed.",
                    "Demo script synthesis completed.",
                    "Pitch synthesis completed.",
                ],
            ),
            "generated_artifacts": [final_artifact],
            "final_readme": readme,
            "demo_script": demo_script,
            "pitch": pitch,
            "final_report": {
                **final_report,
                "mvp_plan": state.get("mvp_plan"),
                "build_timeline": state.get("build_timeline"),
                "mvp_delivery": state.get("mvp_delivery"),
                "mvp_validation": state.get("mvp_validation"),
            },
            **timeline_updates,
        }

    async def remember_outcome(state: WorkflowState) -> dict[str, Any]:
        final_report = state.get("final_report", {})
        summary = final_report.get("summary", "Workflow completed.")

        build_decisions = [
            {
                "node_name": step.node_name,
                "status": step.status,
                "message": step.message,
            }
            for step in state.get("graph_trace", [])
        ]
        payload = {
            "task_id": state["task_id"],
            "idea": state["idea"],
            "summary": summary,
            "outcome": {
                "mvp_scope": state.get("mvp_scope"),
                "repo_plan": state.get("repo_plan"),
                "blocker_analysis": state.get("blocker_analysis"),
                "file_tree": [artifact.get("name") for artifact in state.get("generated_artifacts", [])],
                "generated_artifacts": state.get("generated_artifacts"),
                "github_result": _last_tool_call(state, "github.commit_files") or _last_tool_call(state, "github.create_repo"),
                "build_decisions": build_decisions,
                "errors_and_fixes": state.get("blocker_analysis"),
                "final_report": final_report,
            },
            "tags": ["workflow_outcome"],
        }
        memory_trace = [
            "Captured plan, file tree, GitHub result summary, decisions, and final landing summary.",
        ]
        try:
            await active_retrieval.write_memory(payload)
            memory_trace.append("Wrote memory to storage.")
        except Exception as exc:
            memory_trace.append(f"Memory write skipped: {exc}")

        return append_step(
            state=state,
            node_name="remember_outcome",
            message="Stored the outcome for future retrieval.",
            decision_trace=memory_trace,
        )

    def report_result(state: WorkflowState) -> dict[str, Any]:
        return {
            **append_step(
                state=state,
                node_name="report_result",
                message="Reported the completed Person 1 workflow result.",
                decision_trace=[
                    "Marked the task completed after final package generation.",
                    "Exposed populated dashboard state through the existing API.",
                ],
            ),
            "status": "completed",
        }

    def failed(state: WorkflowState) -> dict[str, Any]:
        last_tool = state.get("last_tool_result") or {}
        reason = (
            state.get("failure_reason")
            or last_tool.get("error")
            or last_tool.get("summary")
            or (
                "Unrecoverable mock tool failure."
                if state["mock_mode"]
                else "Workflow stopped after an unrecoverable tool or validation failure."
            )
        )
        final_report = {
            "status": "failed",
            "mode": "mock" if state["mock_mode"] else "live",
            "model": state["nemotron_model"],
            "summary": reason,
        }
        return {
            **append_step(
                state=state,
                node_name="failed",
                status="failed",
                message=reason,
                decision_trace=[
                    "Detected an unrecoverable tool result.",
                    "Stopped workflow execution and surfaced failure state.",
                ],
            ),
            "status": "failed",
            "final_report": final_report,
            "failure_reason": reason,
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("receive_idea", receive_idea)
    graph.add_node("exchange_github_code", exchange_github_code)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("scope_mvp", scope_mvp)
    graph.add_node("recommend_stack", recommend_stack)
    graph.add_node("plan_repo", plan_repo)
    graph.add_node("create_repo", create_repo)
    graph.add_node("generate_files", generate_files)
    graph.add_node("validate_mvp", validate_mvp)
    graph.add_node("debug_generated_files", debug_generated_files)
    graph.add_node("commit_progress", commit_progress)
    graph.add_node("verify_build", verify_build)
    graph.add_node("handle_blocker", handle_blocker)
    graph.add_node("generate_final_package", generate_final_package)
    graph.add_node("remember_outcome", remember_outcome)
    graph.add_node("report_result", report_result)
    graph.add_node("failed", failed)
    graph.add_edge(START, "receive_idea")
    graph.add_edge("receive_idea", "exchange_github_code")
    graph.add_conditional_edges(
        "exchange_github_code",
        route_after_github_exchange,
        {
            "retrieve_context": "retrieve_context",
            "failed": "failed",
        },
    )
    graph.add_edge("retrieve_context", "scope_mvp")
    graph.add_edge("scope_mvp", "recommend_stack")
    graph.add_edge("recommend_stack", "plan_repo")
    graph.add_edge("plan_repo", "create_repo")
    graph.add_conditional_edges(
        "create_repo",
        route_after_create_repo,
        {
            "generate_files": "generate_files",
            "failed": "failed",
        },
    )
    graph.add_edge("generate_files", "validate_mvp")
    graph.add_conditional_edges(
        "validate_mvp",
        route_after_validate_mvp,
        {
            "debug_generated_files": "debug_generated_files",
            "failed": "failed",
        },
    )
    graph.add_edge("debug_generated_files", "commit_progress")
    graph.add_conditional_edges(
        "commit_progress",
        route_after_commit_progress,
        {
            "verify_build": "verify_build",
            "failed": "failed",
        },
    )
    graph.add_conditional_edges(
        "verify_build",
        route_after_tool_result,
        {
            "generate_final_package": "generate_final_package",
            "handle_blocker": "handle_blocker",
            "failed": "failed",
        },
    )
    graph.add_edge("handle_blocker", "verify_build")
    graph.add_edge("generate_final_package", "remember_outcome")
    graph.add_edge("remember_outcome", "report_result")
    graph.add_edge("report_result", END)
    graph.add_edge("failed", END)
    return graph.compile()


def _build_default_model_client(settings: Settings) -> ModelClient:
    if settings.mock_mode:
        return DeterministicModelClient(mode="mock")
    return NemotronModelClient(settings)


def _live_nemotron_workflow(settings: Settings) -> bool:
    return not settings.mock_mode


def _nemotron_purpose_requires_live(purpose: str, settings: Settings) -> bool:
    if settings.nemotron_strict_live_active:
        return True
    if purpose == "file_manifest" and settings.require_live_file_manifest:
        return True
    return False


def _require_live_nemotron_result(
    result: ModelCallResult[Any],
    purpose: str,
    *,
    enforced: bool,
    settings: Settings,
) -> None:
    if not enforced:
        return
    if result.mode == "live":
        return
    if not _nemotron_purpose_requires_live(purpose, settings):
        return
    detail = result.fallback_reason or "non-live model output"
    raise RuntimeError(
        f"Live Nemotron is required for {purpose} (got mode={result.mode}). {detail}"
    )


def _package_mode(results: list[ModelCallResult]) -> Literal["mock", "live", "partial"]:
    modes = {result.mode for result in results}
    if "live" in modes:
        return "live"
    if "partial" in modes:
        return "partial"
    return "mock"


def _source_warnings(build_context: dict[str, Any]) -> list[dict[str, str]]:
    source_context = build_context.get("sourceContext", {})
    warnings = source_context.get("warnings") if isinstance(source_context, dict) else []
    return [warning for warning in warnings if isinstance(warning, dict)]


def _resolved_stack_summary(build_context: dict[str, Any]) -> str:
    resolved_stack = build_context.get("resolvedTechStack", {})
    items = resolved_stack.get("items") if isinstance(resolved_stack, dict) else None
    if isinstance(items, list):
        stack_items = [str(item).strip() for item in items if str(item).strip()]
        if stack_items:
            return ", ".join(stack_items)
    return "React, FastAPI, Supabase Postgres, Pytest"


def _openclaw_trace_from_tool_call(tool_call: dict[str, Any]) -> list[dict[str, Any]]:
    trace = tool_call.get("openclaw_trace", [])
    return [entry for entry in trace if isinstance(entry, dict)]


def _files_from_generated_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for artifact in artifacts:
        path = str(artifact.get("name") or "").strip()
        if not path or path.endswith("/"):
            continue
        if path.split("/")[-1] == ".env":
            continue

        content = artifact.get("content")
        if content is None:
            content = artifact.get("summary", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, sort_keys=True)
        files.append({"path": path, "content": content})

    return files


def _debug_generated_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    files = _files_from_generated_artifacts(artifacts)
    paths = {file["path"] for file in files}
    content_by_path = {file["path"]: file["content"] for file in files}
    issues: list[str] = []
    warnings: list[str] = []

    required_paths = ["README.md", "docs/ARCHITECTURE.md"]
    for path in required_paths:
        if path not in paths:
            issues.append(f"Missing required file: `{path}`.")
    if not any(path in paths for path in ("demo/demo_script.md", "docs/WALKTHROUGH.md")):
        issues.append(
            "Missing walkthrough file: `demo/demo_script.md` or `docs/WALKTHROUGH.md`."
        )

    if not any(path in paths for path in ("package.json", "requirements.txt")):
        issues.append("Missing a runnable dependency manifest: `package.json` or `requirements.txt`.")
    if not any(path.startswith(("src/", "backend/")) for path in paths):
        issues.append("Missing application source under `src/` or `backend/`.")
    if "package.json" in paths and not any(path.startswith("src/") for path in paths):
        warnings.append("`package.json` exists, but no frontend `src/` files were generated.")
    if "requirements.txt" in paths and not any(path.startswith("backend/") for path in paths):
        warnings.append("`requirements.txt` exists, but no `backend/` files were generated.")

    for path, content in content_by_path.items():
        if not content.strip():
            issues.append(f"Empty generated file: `{path}`.")
        if path.split("/")[-1] == ".env":
            issues.append(f"Unsafe real env file was generated: `{path}`.")
        if "TODO: implement" in content or "placeholder" in content.lower():
            warnings.append(f"Possible placeholder content in `{path}`.")
        if path == "package.json":
            try:
                package_json = json.loads(content)
            except json.JSONDecodeError:
                issues.append("`package.json` is not valid JSON.")
            else:
                package_issues, package_warnings = _debug_frontend_dependencies(
                    package_json
                )
                issues.extend(package_issues)
                warnings.extend(package_warnings)

    status = "passed" if not issues else "needs attention"
    lines = [
        "# Debug Report",
        "",
        f"Status: {status}",
        f"Files checked: {len(files)}",
        f"Issues: {len(issues)}",
        f"Warnings: {len(warnings)}",
        "",
        "## Issues",
        "",
        *(f"- {issue}" for issue in issues),
        *(["- None"] if not issues else []),
        "",
        "## Warnings",
        "",
        *(f"- {warning}" for warning in warnings),
        *(["- None"] if not warnings else []),
        "",
        "## Checked Files",
        "",
        *(f"- `{path}`" for path in sorted(paths)),
        "",
    ]
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "content": "\n".join(lines),
    }


def _debug_frontend_dependencies(package_json: Any) -> tuple[list[str], list[str]]:
    if not isinstance(package_json, dict):
        return ["`package.json` must be a JSON object."], []

    deps = package_json.get("dependencies")
    dev_deps = package_json.get("devDependencies")
    all_deps: dict[str, str] = {}
    for section in (deps, dev_deps):
        if isinstance(section, dict):
            all_deps.update(
                {
                    str(name): str(version)
                    for name, version in section.items()
                    if isinstance(name, str)
                }
            )

    issues: list[str] = []
    warnings: list[str] = []
    minimum_majors = {
        "vite": 8,
        "@vitejs/plugin-react": 6,
        "react": 19,
        "react-dom": 19,
    }
    for package_name, minimum_major in minimum_majors.items():
        version = all_deps.get(package_name)
        if version is None:
            if package_name in {"vite", "react", "react-dom"}:
                issues.append(f"Missing frontend dependency `{package_name}`.")
            else:
                warnings.append(f"Missing frontend helper dependency `{package_name}`.")
            continue
        major = _major_version(version)
        if major is None:
            warnings.append(
                f"Could not verify `{package_name}` version `{version}`."
            )
            continue
        if major < minimum_major:
            issues.append(
                f"`{package_name}` is pinned to `{version}`; expected major {minimum_major} or newer."
            )

    react_version = all_deps.get("react")
    react_dom_version = all_deps.get("react-dom")
    if react_version and react_dom_version and react_version != react_dom_version:
        warnings.append(
            f"`react` ({react_version}) and `react-dom` ({react_dom_version}) should match."
        )

    if "vite" in all_deps and "@vitejs/plugin-react" in all_deps:
        vite_major = _major_version(all_deps["vite"])
        plugin_major = _major_version(all_deps["@vitejs/plugin-react"])
        if vite_major is not None and plugin_major is not None:
            if vite_major >= 8 and plugin_major < 6:
                issues.append(
                    "Vite 8 generated apps should use `@vitejs/plugin-react` major 6 or newer."
                )

    return issues, warnings


def _major_version(version: str) -> int | None:
    cleaned = version.strip()
    for prefix in ("^", "~", ">=", "<=", ">", "<", "="):
        cleaned = cleaned.removeprefix(prefix).strip()
    first = cleaned.split(".", 1)[0]
    return int(first) if first.isdigit() else None


def _last_tool_call(state: WorkflowState, tool_name: str) -> dict[str, Any]:
    for call in reversed(state.get("tool_calls", [])):
        raw_result = call.get("raw_result") if isinstance(call, dict) else None
        raw_tool_name = raw_result.get("tool_name") if isinstance(raw_result, dict) else None
        if call.get("tool") == tool_name or raw_tool_name == tool_name:
            return call
    return {}


def _repo_name_from_url(repo_url: str | None) -> str | None:
    if not repo_url:
        return None
    name = repo_url.rstrip("/").split("/")[-1].strip()
    return name or None


def json_dumps_safe(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def route_after_tool_result(state: dict[str, Any]) -> str:
    result = state.get("last_tool_result") or {}
    if result.get("status") == "success":
        return "generate_final_package"
    if result.get("recoverable") is True:
        return "handle_blocker"
    return "failed"


def route_after_github_exchange(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failed"
    return "retrieve_context"


def route_after_create_repo(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failed"
    return "generate_files"


def route_after_validate_mvp(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failed"
    validation = state.get("mvp_validation") or {}
    if validation.get("passed") is False:
        return "failed"
    return "debug_generated_files"


def route_after_commit_progress(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failed"
    return "verify_build"
