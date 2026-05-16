from __future__ import annotations

import json
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        enable_decoding=False,
        extra="ignore",
        populate_by_name=True,
    )

    adapter_mode: Literal["mock", "live"] = Field(
        default="live",
        validation_alias="ADAPTER_MODE",
    )
    allow_idea_aware_partial: bool = Field(
        default=True,
        validation_alias="ALLOW_IDEA_AWARE_PARTIAL",
    )
    mock_mode_override: bool | None = Field(
        default=None,
        validation_alias="MOCK_MODE",
    )
    nvidia_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="NVIDIA_API_KEY",
    )
    nemotron_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        validation_alias="NEMOTRON_MODEL",
    )
    nemotron_fast_model: str = Field(
        default="nvidia/nvidia-nemotron-nano-9b-v2",
        validation_alias="NEMOTRON_FAST_MODEL",
    )
    nemotron_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias="NEMOTRON_BASE_URL",
    )
    nemotron_timeout_seconds: float = Field(
        default=30,
        gt=0,
        validation_alias="NEMOTRON_TIMEOUT_SECONDS",
    )
    nemotron_max_retries: int = Field(
        default=1,
        ge=0,
        validation_alias="NEMOTRON_MAX_RETRIES",
    )
    nemotron_poll_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias="NEMOTRON_POLL_ATTEMPTS",
    )
    nemotron_poll_interval_seconds: float = Field(
        default=1,
        ge=0,
        validation_alias="NEMOTRON_POLL_INTERVAL_SECONDS",
    )
    openclaw_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENCLAW_API_KEY",
    )
    openclaw_gateway_token: SecretStr | None = Field(
        default=None,
        validation_alias="OPENCLAW_GATEWAY_TOKEN",
    )
    openclaw_env: str | None = Field(
        default=None,
        validation_alias="OPENCLAW_ENV",
    )
    openclaw_base_url: str | None = Field(
        default=None,
        validation_alias="OPENCLAW_BASE_URL",
    )
    supabase_url: str | None = Field(
        default=None,
        validation_alias="SUPABASE_URL",
    )
    supabase_service_role_key: SecretStr | None = Field(
        default=None,
        validation_alias="SUPABASE_SERVICE_ROLE_KEY",
    )
    supabase_anon_key: SecretStr | None = Field(
        default=None,
        validation_alias="SUPABASE_ANON_KEY",
    )
    github_oauth_client_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GITHUB_OAUTH_CLIENT_ID", "GITHUB_CLIENT_ID"),
    )
    github_oauth_client_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GITHUB_OAUTH_CLIENT_SECRET", "GITHUB_CLIENT_SECRET"),
    )
    github_oauth_redirect_uri: str | None = Field(
        default="http://127.0.0.1:3001/api/auth/github/callback",
        validation_alias=AliasChoices("GITHUB_OAUTH_REDIRECT_URI", "GITHUB_REDIRECT_URI"),
    )
    github_personal_access_token: SecretStr | None = Field(
        default=None,
        validation_alias="GITHUB_TOKEN",
    )
    github_env_token_fallback_enabled: bool = Field(
        default=False,
        validation_alias="GITHUB_ENV_TOKEN_FALLBACK_ENABLED",
    )
    github_owner: str | None = Field(
        default=None,
        validation_alias="GITHUB_OWNER",
    )
    github_token_encryption_key: SecretStr | None = Field(
        default=None,
        validation_alias="GITHUB_TOKEN_ENCRYPTION_KEY",
    )
    frontend_base_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("FRONTEND_BASE_URL", "APP_BASE_URL"),
    )
    rag_scrape_urls: str = Field(
        default="",
        validation_alias="RAG_SCRAPE_URLS",
    )

    @model_validator(mode="after")
    def apply_mock_mode_override(self) -> "Settings":
        if self.mock_mode_override is not None:
            self.adapter_mode = "mock" if self.mock_mode_override else "live"
        return self
    cors_origins: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS),
        validation_alias="CORS_ORIGINS",
    )

    @field_validator(
        "nvidia_api_key",
        "openclaw_api_key",
        "openclaw_gateway_token",
        "supabase_service_role_key",
        "supabase_anon_key",
        "github_oauth_client_secret",
        "github_token_encryption_key",
        "github_personal_access_token",
        mode="before",
    )
    @classmethod
    def empty_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if value is None:
            return list(DEFAULT_CORS_ORIGINS)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return list(DEFAULT_CORS_ORIGINS)
            if stripped.startswith("["):
                return json.loads(stripped)
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return value

    @property
    def mock_mode(self) -> bool:
        return self.adapter_mode == "mock"

    @property
    def nvidia_configured(self) -> bool:
        return self._secret_has_value(self.nvidia_api_key)

    @property
    def openclaw_configured(self) -> bool:
        return self._secret_has_value(self.openclaw_api_key)

    @property
    def supabase_configured(self) -> bool:
        return bool((self.supabase_url or "").strip()) and self._secret_has_value(
            self.supabase_service_role_key
        )

    @property
    def github_oauth_configured(self) -> bool:
        return (
            bool((self.github_oauth_client_id or "").strip())
            and bool((self.github_oauth_redirect_uri or "").strip())
            and self._secret_has_value(self.github_oauth_client_secret)
            and self._secret_has_value(self.github_token_encryption_key)
        )

    @property
    def github_pat_configured(self) -> bool:
        return self._secret_has_value(self.github_personal_access_token) and bool(
            (self.github_owner or "").strip()
        )

    @property
    def github_pat_token_type(self) -> str | None:
        if not self._secret_has_value(self.github_personal_access_token):
            return None
        token = self.github_personal_access_token.get_secret_value().strip()
        if token.startswith("github_pat_"):
            return "fine_grained"
        if token.startswith("ghp_"):
            return "classic"
        return "unknown"

    @property
    def github_pat_can_create_repositories(self) -> bool:
        token_type = self.github_pat_token_type
        if token_type == "fine_grained":
            return False
        return self.github_pat_configured and token_type in {"classic", "unknown", None}

    @property
    def rag_live_ready(self) -> bool:
        return self.nvidia_configured and self.supabase_configured

    @staticmethod
    def _secret_has_value(secret: SecretStr | None) -> bool:
        if secret is None:
            return False
        return bool(secret.get_secret_value().strip())

    @property
    def health_status(self) -> Literal["ok", "degraded"]:
        if self.adapter_mode == "live" and not self.nvidia_configured:
            return "degraded"
        return "ok"
