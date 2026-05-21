from pathlib import Path


FRONTEND_PAGE = Path(__file__).resolve().parents[1] / "frontend" / "app" / "page.tsx"


def test_frontend_uses_backend_owned_github_oauth_flow():
    source = FRONTEND_PAGE.read_text(encoding="utf-8")

    assert "/api/auth/github/login" in source
    assert "/api/auth/github/config" in source
    assert "/api/auth/github/status" in source
    assert "/api/orchestrator/start-project" in source
    assert "rulesUrl" in source
    assert "referenceUrls" in source
    assert "repoPreference" in source
    assert "repoDescription" in source
    assert "github_connection_id" in source
    assert "Activity log" in source
    assert "githubToken" not in source
    assert "github_auth_code" not in source
    assert "GITHUB_TOKEN" not in source
    assert "Use backend token" not in source
    assert "NEXT_PUBLIC_GITHUB_CLIENT_ID" not in source
    assert "https://github.com/login/oauth/authorize" not in source
