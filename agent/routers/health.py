from __future__ import annotations

from fastapi import APIRouter, Depends

from agent.config import Settings
from agent.dependencies import get_settings
from agent.openclaw_runtime import openclaw_runtime_status
from agent.rag.env_status import rag_env_status

router = APIRouter()


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    rag_status = rag_env_status()
    openclaw_status = openclaw_runtime_status(settings)
    return {
        "status": settings.health_status,
        "adapter_mode": settings.adapter_mode,
        "mock_mode": settings.mock_mode,
        "nemotron_model": settings.nemotron_model,
        "nemotron_fast_model": settings.nemotron_fast_model,
        "nvidia_configured": settings.nvidia_configured,
        "allow_idea_aware_partial": settings.allow_idea_aware_partial,
        "nemotron_strict_live_active": settings.nemotron_strict_live_active,
        "workflow_live_manifest_only": settings.workflow_live_manifest_only,
        "nemotron_fast_fallback_active": settings.nemotron_fast_fallback_active,
        "nemotron_effective_timeout_seconds": settings.nemotron_effective_timeout_seconds,
        "nemotron_file_manifest_timeout_seconds": settings.nemotron_file_manifest_timeout_seconds,
        "nemotron_poll_max_seconds": settings.nemotron_poll_max_seconds,
        "require_live_file_manifest": settings.require_live_file_manifest,
        "openclaw_configured": settings.openclaw_configured,
        "openclaw_env": settings.openclaw_env,
        **openclaw_status,
        "supabase_configured": settings.supabase_configured,
        "rag_configured": rag_status["configured"],
        "rag_missing_env": rag_status["missing_required"],
        "rag_live_ready": settings.rag_live_ready,
        "github_oauth_configured": settings.github_oauth_configured,
        "github_pat_configured": settings.github_pat_configured,
        "github_oauth_redirect_uri": settings.github_oauth_redirect_uri,
        "service": "mvpilot-agent",
    }
