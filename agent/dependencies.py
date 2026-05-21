from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Request

from agent.config import Settings
from agent.github_oauth import (
    GitHubConnectionService,
    InMemoryGitHubConnectionStore,
    SupabaseGitHubConnectionStore,
)
from agent.service import AgentService
from agent.project_session_store import SupabasePersistingTaskStore, build_task_store
from agent.task_store import InMemoryTaskStore


@lru_cache
def build_settings() -> Settings:
    return Settings()


def reload_runtime_settings(request: Request) -> Settings:
    """Re-read .env and refresh app-scoped services (e.g. after GITHUB_TOKEN changes)."""

    build_settings.cache_clear()
    load_dotenv(override=True)
    fresh_settings = Settings()
    store: InMemoryGitHubConnectionStore | SupabaseGitHubConnectionStore
    if fresh_settings.adapter_mode == "live" and fresh_settings.supabase_configured:
        store = SupabaseGitHubConnectionStore(fresh_settings)
    else:
        store = InMemoryGitHubConnectionStore()
    github_service = GitHubConnectionService(settings=fresh_settings, store=store)
    request.app.state.settings = fresh_settings
    request.app.state.github_connection_store = store
    request.app.state.github_connection_service = github_service
    request.app.state.task_store = build_task_store(fresh_settings)
    request.app.state.agent_service = AgentService(
        request.app.state.task_store,
        fresh_settings,
        github_connections=github_service,
    )
    return fresh_settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_task_store(request: Request) -> InMemoryTaskStore | SupabasePersistingTaskStore:
    return request.app.state.task_store


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_github_connection_service(request: Request) -> GitHubConnectionService:
    return request.app.state.github_connection_service
