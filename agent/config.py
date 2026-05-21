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
        default=False,
        validation_alias=AliasChoices("ALLOW_DEGRADED_MODE", "ALLOW_IDEA_AWARE_PARTIAL"),
    )
    nemotron_fast_fallback: bool = Field(
        default=False,
        validation_alias="NEMOTRON_FAST_FALLBACK",
    )
    nemotron_live_attempt_timeout_seconds: float = Field(
        default=75,
        gt=0,
        validation_alias="NEMOTRON_LIVE_ATTEMPT_TIMEOUT_SECONDS",
    )
    nemotron_fast_fallback_max_retries: int = Field(
        default=0,
        ge=0,
        validation_alias="NEMOTRON_FAST_FALLBACK_MAX_RETRIES",
    )
    nemotron_fast_fallback_poll_max_seconds: float = Field(
        default=90,
        gt=0,
        validation_alias="NEMOTRON_FAST_FALLBACK_POLL_MAX_SECONDS",
    )
    mock_mode_override: bool | None = Field(
        default=None,
        validation_alias="MOCK_MODE",
    )
    nvidia_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NEMOTRON_API_KEY", "NVIDIA_API_KEY"),
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
        default=900,
        gt=0,
        validation_alias="NEMOTRON_TIMEOUT_SECONDS",
    )
    nemotron_strict_timeout_seconds: float = Field(
        default=900,
        gt=0,
        validation_alias="NEMOTRON_STRICT_TIMEOUT_SECONDS",
    )
    nemotron_file_manifest_timeout_seconds: float = Field(
        default=1200,
        gt=0,
        validation_alias="NEMOTRON_FILE_MANIFEST_TIMEOUT_SECONDS",
    )
    nemotron_repo_plan_timeout_seconds: float = Field(
        default=1200,
        gt=0,
        validation_alias="NEMOTRON_REPO_PLAN_TIMEOUT_SECONDS",
    )
    nemotron_max_retries: int = Field(
        default=3,
        ge=0,
        validation_alias="NEMOTRON_MAX_RETRIES",
    )
    nemotron_poll_attempts: int = Field(
        default=90,
        ge=1,
        validation_alias="NEMOTRON_POLL_ATTEMPTS",
    )
    nemotron_poll_interval_seconds: float = Field(
        default=2,
        ge=0,
        validation_alias="NEMOTRON_POLL_INTERVAL_SECONDS",
    )
    nemotron_poll_max_seconds: float = Field(
        default=3600,
        gt=0,
        validation_alias="NEMOTRON_POLL_MAX_SECONDS",
    )
    nemotron_file_manifest_poll_max_seconds: float = Field(
        default=3600,
        gt=0,
        validation_alias="NEMOTRON_FILE_MANIFEST_POLL_MAX_SECONDS",
    )
    nemotron_repo_plan_poll_max_seconds: float = Field(
        default=3600,
        gt=0,
        validation_alias="NEMOTRON_REPO_PLAN_POLL_MAX_SECONDS",
    )
    nemotron_repo_plan_max_retries: int = Field(
        default=5,
        ge=0,
        validation_alias="NEMOTRON_REPO_PLAN_MAX_RETRIES",
    )
    nemotron_reasoning_effort: str = Field(
        default="none",
        validation_alias="NEMOTRON_REASONING_EFFORT",
    )
    nemotron_planning_max_tokens: int = Field(
        default=2500,
        ge=900,
        validation_alias="NEMOTRON_PLANNING_MAX_TOKENS",
    )
    nemotron_repo_plan_max_tokens: int = Field(
        default=6000,
        ge=2000,
        validation_alias="NEMOTRON_REPO_PLAN_MAX_TOKENS",
    )
    nemotron_stack_recommendation_max_tokens: int = Field(
        default=4000,
        ge=1500,
        validation_alias="NEMOTRON_STACK_RECOMMENDATION_MAX_TOKENS",
    )
    nemotron_file_manifest_max_tokens: int = Field(
        default=3500,
        ge=1200,
        validation_alias="NEMOTRON_FILE_MANIFEST_MAX_TOKENS",
    )
    nemotron_file_manifest_max_retries: int = Field(
        default=5,
        ge=0,
        validation_alias="NEMOTRON_FILE_MANIFEST_MAX_RETRIES",
    )
    require_live_file_manifest: bool = Field(
        default=True,
        validation_alias="REQUIRE_LIVE_FILE_MANIFEST",
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
        validation_alias=AliasChoices("OPENCLAW_ENDPOINT", "OPENCLAW_BASE_URL"),
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
    def nemotron_fast_fallback_active(self) -> bool:
        return self.allow_idea_aware_partial and self.nemotron_fast_fallback

    @property
    def nemotron_strict_live_active(self) -> bool:
        return not self.allow_idea_aware_partial

    @property
    def workflow_live_manifest_only(self) -> bool:
        """Non-mock workflow runs commit only Nemotron live artifacts."""
        return not self.mock_mode and self.require_live_file_manifest

    def nemotron_read_timeout_seconds(self, purpose: str) -> float:
        if self.nemotron_fast_fallback_active:
            return self.nemotron_live_attempt_timeout_seconds
        if purpose == "file_manifest":
            return self.nemotron_file_manifest_timeout_seconds
        if purpose == "plan_repo":
            return self.nemotron_repo_plan_timeout_seconds
        if self.nemotron_strict_live_active:
            return max(self.nemotron_strict_timeout_seconds, self.nemotron_timeout_seconds)
        return self.nemotron_timeout_seconds

    @property
    def nemotron_effective_timeout_seconds(self) -> float:
        return self.nemotron_read_timeout_seconds("scope_mvp")

    @property
    def nemotron_effective_max_retries(self) -> int:
        if self.nemotron_fast_fallback_active:
            return self.nemotron_fast_fallback_max_retries
        return self.nemotron_max_retries

    def nemotron_max_tokens_for(self, purpose: str) -> int:
        if purpose == "plan_repo":
            return self.nemotron_repo_plan_max_tokens
        if purpose == "file_manifest":
            return self.nemotron_file_manifest_max_tokens
        if purpose == "recommend_stack":
            return self.nemotron_stack_recommendation_max_tokens
        return self.nemotron_planning_max_tokens

    def nemotron_max_retries_for(self, purpose: str) -> int:
        if purpose == "file_manifest":
            return max(self.nemotron_effective_max_retries, self.nemotron_file_manifest_max_retries)
        if purpose == "plan_repo":
            return max(self.nemotron_effective_max_retries, self.nemotron_repo_plan_max_retries)
        return self.nemotron_effective_max_retries

    def nemotron_poll_max_seconds_for(self, purpose: str) -> float:
        if self.nemotron_fast_fallback_active:
            return self.nemotron_fast_fallback_poll_max_seconds
        if purpose == "file_manifest":
            return self.nemotron_file_manifest_poll_max_seconds
        if purpose == "plan_repo":
            return self.nemotron_repo_plan_poll_max_seconds
        if self.nemotron_strict_live_active:
            return self.nemotron_poll_max_seconds
        return self.nemotron_poll_max_seconds

    @property
    def nemotron_effective_poll_max_seconds(self) -> float:
        return self.nemotron_poll_max_seconds_for("scope_mvp")

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
