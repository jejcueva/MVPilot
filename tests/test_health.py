from agent.config import Settings
from agent.main import create_app
from fastapi.testclient import TestClient


def test_health_returns_mock_defaults_without_secret_values(monkeypatch):
    for key in (
        "NVIDIA_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "OPENCLAW_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_OWNER",
        "GITHUB_OAUTH_CLIENT_ID",
        "GITHUB_OAUTH_CLIENT_SECRET",
        "GITHUB_OAUTH_REDIRECT_URI",
        "GITHUB_CLIENT_ID",
        "GITHUB_CLIENT_SECRET",
        "GITHUB_REDIRECT_URI",
        "GITHUB_TOKEN_ENCRYPTION_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    app = create_app(settings=Settings(_env_file=None, adapter_mode="mock"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "status": "ok",
        "adapter_mode": "mock",
        "mock_mode": True,
        "nemotron_model": "nvidia/nemotron-3-super-120b-a12b",
        "nemotron_fast_model": "nvidia/nvidia-nemotron-nano-9b-v2",
        "nvidia_configured": False,
        "openclaw_configured": False,
        "openclaw_env": None,
        "openclaw_runtime_ready": False,
        "openclaw_registered_tools": [],
        "supabase_configured": False,
        "rag_configured": False,
        "rag_missing_env": [
            "NVIDIA_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
        ],
        "rag_live_ready": False,
        "github_oauth_configured": False,
        "github_pat_configured": False,
        "github_oauth_redirect_uri": "http://127.0.0.1:3001/api/auth/github/callback",
        "service": "mvpilot-agent",
    }
    assert "fake-nvidia" not in response.text
    assert "fake-openclaw" not in response.text


def test_health_is_degraded_when_live_mode_lacks_nvidia_config(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    app = create_app(settings=Settings(_env_file=None, adapter_mode="live"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_settings_load_from_environment_without_requiring_real_keys(monkeypatch):
    monkeypatch.setenv("ADAPTER_MODE", "live")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-nvidia-key")
    monkeypatch.setenv("OPENCLAW_API_KEY", "fake-openclaw-key")
    monkeypatch.setenv("NEMOTRON_MODEL", "nvidia/custom-model")
    monkeypatch.setenv("NEMOTRON_FAST_MODEL", "nvidia/custom-fast-model")
    monkeypatch.setenv("NEMOTRON_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("NEMOTRON_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("NEMOTRON_MAX_RETRIES", "2")
    monkeypatch.setenv("NEMOTRON_POLL_ATTEMPTS", "4")
    monkeypatch.setenv("NEMOTRON_POLL_INTERVAL_SECONDS", "0.5")
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )

    settings = Settings(_env_file=None)

    assert settings.adapter_mode == "live"
    assert settings.nemotron_model == "nvidia/custom-model"
    assert settings.nemotron_fast_model == "nvidia/custom-fast-model"
    assert str(settings.nemotron_base_url) == "https://example.test/v1"
    assert settings.nemotron_timeout_seconds == 12
    assert settings.nemotron_max_retries == 2
    assert settings.nemotron_poll_attempts == 4
    assert settings.nemotron_poll_interval_seconds == 0.5
    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert settings.nvidia_configured is True
    assert settings.openclaw_configured is True
    assert "fake-nvidia-key" not in settings.model_dump_json()


def test_health_reports_rag_configured_when_supabase_and_nvidia_set(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-nvidia-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-service-role")
    monkeypatch.setenv("OPENCLAW_API_KEY", "fake-openclaw-key")
    monkeypatch.setenv("OPENCLAW_ENV", "development")

    app = create_app(settings=Settings(_env_file=None, adapter_mode="mock"))

    with TestClient(app) as client:
        response = client.get("/health")

    data = response.json()
    assert data["rag_configured"] is True
    assert data["rag_missing_env"] == []
    assert data["rag_live_ready"] is True
    assert data["supabase_configured"] is True
    assert data["openclaw_env"] == "development"
    assert data["openclaw_runtime_ready"] is True
    assert "github.create_repo" in data["openclaw_registered_tools"]
