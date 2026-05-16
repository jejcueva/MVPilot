from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

from agent.schemas import (
    RunAgentRequest,
    TaskRecord,
    UploadedSourceFile,
    UploadedSourceFileContent,
)

MAX_TEXT_FILE_BYTES = 1_000_000
MAX_SOURCE_PROMPT_CHARS = 20_000
MAX_SOURCE_CHARS_PER_ITEM = 4_000
URL_FETCH_TIMEOUT_SECONDS = 5.0

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
UNSUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}
SUPPORTED_URL_CONTENT_TYPES = {
    "application/csv",
    "application/json",
    "application/markdown",
    "application/x-markdown",
    "application/xhtml+xml",
    "application/xml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/x-markdown",
    "text/xml",
}


class FrontendIntake(BaseModel):
    title: str | None = None
    idea: str
    source: str | None = None
    primaryRulesUrl: str | None = None
    additionalUrls: list[str] = Field(default_factory=list)
    repoPreference: str = "create_new_repo"
    repoName: str | None = None
    repoDescription: str | None = None
    repoUrl: str | None = None
    visibility: str = "public"
    branch: str = "main"
    uploadedFiles: list[UploadedSourceFile] = Field(default_factory=list)
    githubConnected: bool = False
    githubConnectionId: str | None = None
    targetUsers: str | None = None
    techStackPreference: str | None = None
    requiredFeatures: list[str] = Field(default_factory=list)

    @field_validator(
        "title",
        "idea",
        "source",
        "primaryRulesUrl",
        "repoPreference",
        "repoName",
        "repoDescription",
        "repoUrl",
        "visibility",
        "branch",
        "githubConnectionId",
        "targetUsers",
        "techStackPreference",
        mode="before",
    )
    @classmethod
    def trim_text(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("requiredFeatures", mode="before")
    @classmethod
    def coerce_required_features(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("additionalUrls")
    @classmethod
    def trim_urls(cls, value: list[str]) -> list[str]:
        return [url.strip() for url in value if url.strip()]


@dataclass(frozen=True)
class FetchedUrl:
    url: str
    title: str | None
    content_type: str
    text: str


class SourceFetchError(Exception):
    def __init__(self, source: str, reason: str) -> None:
        super().__init__(reason)
        self.source = source
        self.reason = reason


FetchUrl = Callable[[str], Awaitable[FetchedUrl]]


def build_frontend_intake_from_request(request: RunAgentRequest) -> FrontendIntake:
    return FrontendIntake(
        title=request.title,
        idea=request.idea,
        source=request.source,
        primaryRulesUrl=request.primary_rules_url,
        additionalUrls=request.additional_urls,
        repoPreference=request.repo_preference,
        repoName=request.repo_name,
        repoDescription=request.repo_description,
        repoUrl=request.repo_url,
        visibility=request.repo_visibility,
        branch=request.branch,
        uploadedFiles=_safe_uploaded_files(
            request.additional_files or request.uploaded_file_contents
        ),
        githubConnected=request.github_connected,
        githubConnectionId=request.github_connection_id,
        targetUsers=request.target_users,
        techStackPreference=request.tech_stack_preference,
        requiredFeatures=list(request.required_features),
    )


def build_frontend_intake_from_task(task: TaskRecord) -> FrontendIntake:
    return FrontendIntake(
        title=task.title,
        idea=task.idea,
        source=task.source,
        primaryRulesUrl=task.primary_rules_url,
        additionalUrls=task.additional_urls,
        repoPreference=task.repo_preference,
        repoName=task.repo_name,
        repoDescription=task.repo_description,
        repoUrl=task.repo_url,
        visibility=task.repo_visibility,
        branch=task.branch,
        uploadedFiles=task.additional_files,
        githubConnected=task.github_connected,
        githubConnectionId=task.github_connection_id,
        targetUsers=task.target_users,
        techStackPreference=task.tech_stack_preference,
        requiredFeatures=list(task.required_features),
    )


def build_optional_params_from_frontend_intake(
    intake: FrontendIntake,
) -> dict[str, Any] | None:
    features = [item for item in [intake.title, intake.source] if item]
    optional_params: dict[str, Any] = {}
    if features:
        optional_params["features"] = features
    if intake.targetUsers:
        optional_params["targetUsers"] = intake.targetUsers
    if intake.techStackPreference:
        optional_params["techStackPreference"] = intake.techStackPreference
    if intake.requiredFeatures:
        optional_params["requiredFeatures"] = intake.requiredFeatures
    if intake.githubConnected:
        optional_params["repoPreference"] = "GitHub-connected repository handoff"
    optional_params["repoPreference"] = intake.repoPreference
    optional_params["repoName"] = intake.repoName
    optional_params["repoDescription"] = intake.repoDescription
    optional_params["repoUrl"] = intake.repoUrl
    optional_params["visibility"] = intake.visibility
    return optional_params or None


async def build_source_context(
    intake: FrontendIntake,
    *,
    uploaded_files: list[UploadedSourceFileContent] | None = None,
    fetch_url: FetchUrl | None = None,
) -> dict[str, Any]:
    active_fetch_url = fetch_url or fetch_submitted_url
    budget = _SourceBudget()
    warnings: list[dict[str, str]] = []
    fetched_url_count = 0

    primary_summary = None
    if intake.primaryRulesUrl:
        primary_summary = await _summarize_url(
            intake.primaryRulesUrl,
            active_fetch_url,
            budget,
            warnings,
        )
        if primary_summary:
            fetched_url_count += 1

    additional_summaries = []
    for url in intake.additionalUrls:
        summary = await _summarize_url(url, active_fetch_url, budget, warnings)
        if summary:
            fetched_url_count += 1
            additional_summaries.append(summary)

    uploaded_summaries = []
    for file in uploaded_files or []:
        summary = _summarize_uploaded_file(file, budget, warnings)
        if summary:
            uploaded_summaries.append(summary)

    return {
        "primaryRulesUrl": primary_summary,
        "additionalUrls": additional_summaries,
        "uploadedFiles": uploaded_summaries,
        "warnings": warnings,
        "sourceCounts": {
            "submittedUrls": int(bool(intake.primaryRulesUrl)) + len(intake.additionalUrls),
            "submittedFiles": len(intake.uploadedFiles),
            "fetchedUrls": fetched_url_count,
            "summarizedFiles": len(uploaded_summaries),
            "warnings": len(warnings),
            "promptCharacters": budget.used,
            "promptCharacterLimit": MAX_SOURCE_PROMPT_CHARS,
        },
    }


async def fetch_submitted_url(url: str) -> FetchedUrl:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(URL_FETCH_TIMEOUT_SECONDS),
            follow_redirects=True,
        ) as client:
            response = await client.get(url, headers={"User-Agent": "MVPilot/0.1"})
    except httpx.TimeoutException as exc:
        raise SourceFetchError(url, "request timed out") from exc
    except httpx.HTTPError as exc:
        raise SourceFetchError(url, exc.__class__.__name__) from exc

    if response.status_code >= 400:
        raise SourceFetchError(url, f"HTTP {response.status_code}")

    content_type = _content_type(response.headers.get("content-type"))
    if not _is_supported_url_content_type(content_type):
        reason = f"unsupported content type {content_type or 'unknown'}"
        raise SourceFetchError(url, reason)

    text, title = _extract_response_text(response.text, content_type)
    return FetchedUrl(
        url=url,
        title=title,
        content_type=content_type,
        text=text,
    )


async def _summarize_url(
    url: str,
    fetch_url: FetchUrl,
    budget: "_SourceBudget",
    warnings: list[dict[str, str]],
) -> dict[str, Any] | None:
    try:
        fetched = await fetch_url(url)
    except SourceFetchError as exc:
        warnings.append(
            {
                "source": exc.source,
                "message": f"Could not read submitted URL: {exc.reason}",
            }
        )
        return None
    except Exception as exc:
        warnings.append(
            {
                "source": url,
                "message": f"Could not read submitted URL: {exc.__class__.__name__}",
            }
        )
        return None

    summary = budget.take(fetched.text, fetched.url, warnings)
    if not summary:
        warnings.append(
            {
                "source": fetched.url,
                "message": "Submitted URL did not contain readable text.",
            }
        )
        return None

    return {
        "url": fetched.url,
        "title": fetched.title,
        "contentType": fetched.content_type,
        "summary": summary,
        "characterCount": len(summary),
    }


def _summarize_uploaded_file(
    file: UploadedSourceFileContent,
    budget: "_SourceBudget",
    warnings: list[dict[str, str]],
) -> dict[str, Any] | None:
    extension = Path(file.name).suffix.lower()
    if extension in UNSUPPORTED_DOCUMENT_EXTENSIONS:
        warnings.append(
            {
                "source": file.name,
                "message": f"Text extraction is not supported for {extension} files yet.",
            }
        )
        return None

    if extension not in SUPPORTED_TEXT_EXTENSIONS:
        warnings.append(
            {
                "source": file.name,
                "message": f"Text extraction is not supported for {extension or 'unknown'} files.",
            }
        )
        return None

    if file.size_bytes > MAX_TEXT_FILE_BYTES:
        warnings.append(
            {
                "source": file.name,
                "message": "Supported text files are limited to 1 MB and this file was skipped.",
            }
        )
        return None

    text = file.content.decode("utf-8", errors="replace")
    summary = budget.take(text, file.name, warnings)
    if not summary:
        warnings.append(
            {
                "source": file.name,
                "message": "Uploaded file did not contain readable text.",
            }
        )
        return None

    return {
        "name": file.name,
        "contentType": file.content_type,
        "sizeBytes": file.size_bytes,
        "summary": summary,
        "characterCount": len(summary),
    }


class _SourceBudget:
    def __init__(self) -> None:
        self.used = 0

    def take(
        self,
        text: str,
        source: str,
        warnings: list[dict[str, str]],
    ) -> str:
        cleaned = _clean_text(text)
        if not cleaned or self.used >= MAX_SOURCE_PROMPT_CHARS:
            if cleaned:
                warnings.append(
                    {
                        "source": source,
                        "message": "Source prompt character limit reached before this source.",
                    }
                )
            return ""

        remaining = MAX_SOURCE_PROMPT_CHARS - self.used
        limit = min(MAX_SOURCE_CHARS_PER_ITEM, remaining)
        summary = cleaned[:limit].rstrip()
        self.used += len(summary)
        if len(cleaned) > limit:
            warnings.append(
                {
                    "source": source,
                    "message": "Source text was truncated to fit prompt limits.",
                }
            )
        return summary


def _safe_uploaded_files(
    files: list[UploadedSourceFile] | list[UploadedSourceFileContent],
) -> list[UploadedSourceFile]:
    return [UploadedSourceFile.model_validate(file.model_dump()) for file in files]


def _content_type(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def _is_supported_url_content_type(content_type: str) -> bool:
    return content_type.startswith("text/") or content_type in SUPPORTED_URL_CONTENT_TYPES


def _extract_response_text(raw_text: str, content_type: str) -> tuple[str, str | None]:
    if content_type == "text/html" or content_type == "application/xhtml+xml":
        soup = BeautifulSoup(raw_text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else None
        return soup.get_text(" ", strip=True), title
    return raw_text, None


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
