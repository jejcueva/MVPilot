from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.config import Settings
from agent.dependencies import build_settings
from agent.github_oauth import (
    GitHubConnectionService,
    InMemoryGitHubConnectionStore,
    SupabaseGitHubConnectionStore,
)
from agent.rag.routes import router as rag_router
from agent.routers.agent import orchestrator_router, router as agent_router
from agent.routers.github import auth_router as github_auth_router
from agent.routers.github import router as github_router
from agent.routers.github import upload_router as github_upload_router
from agent.routers.health import router as health_router
from agent.service import AgentService
from agent.project_session_store import SupabasePersistingTaskStore, build_task_store
from agent.task_store import InMemoryTaskStore

load_dotenv()


def create_app(
    *,
    settings: Settings | None = None,
    task_store: InMemoryTaskStore | None = None,
) -> FastAPI:
    active_settings = settings or build_settings()
    active_task_store = task_store or build_task_store(active_settings)
    if active_settings.adapter_mode == "live" and active_settings.supabase_configured:
        github_connection_store = SupabaseGitHubConnectionStore(active_settings)
    else:
        github_connection_store = InMemoryGitHubConnectionStore()
    github_connection_service = GitHubConnectionService(
        settings=active_settings,
        store=github_connection_store,
    )
    active_service = AgentService(
        active_task_store,
        active_settings,
        github_connections=github_connection_service,
    )

    app = FastAPI(title="MVPilot Agent Backend")
    app.state.settings = active_settings
    app.state.task_store = active_task_store
    app.state.github_connection_store = github_connection_store
    app.state.github_connection_service = github_connection_service
    app.state.agent_service = active_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(github_router)
    app.include_router(github_auth_router)
    app.include_router(github_upload_router)
    app.include_router(agent_router)
    app.include_router(orchestrator_router)
    app.include_router(rag_router, prefix="/rag", tags=["rag"])
    return app


app = create_app()
