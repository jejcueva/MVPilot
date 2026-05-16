from __future__ import annotations

from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.dependencies import get_github_connection_service, reload_runtime_settings
from agent.github_oauth import GitHubConnectionService, GitHubOAuthError
from tools.github_tool import GitHubConfig, commit_files, create_repo

router = APIRouter(prefix="/github", tags=["github"])
auth_router = APIRouter(prefix="/api/auth/github", tags=["github-auth"])
upload_router = APIRouter(prefix="/api/github", tags=["github"])


class GitHubUploadFile(BaseModel):
    path: str = Field(min_length=1)
    content: str


class GitHubUploadProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projectId: str = Field(min_length=1)
    repoPreference: Literal["create_new_repo", "use_existing_repo"] = "create_new_repo"
    repoName: str | None = None
    repoDescription: str | None = None
    repoUrl: str | None = None
    visibility: Literal["private", "public"] = "private"
    branch: str = "main"
    files: list[GitHubUploadFile] = Field(min_length=1)
    commitMessage: str = "Add generated MVPilot project"
    githubConnectionId: str | None = None


@auth_router.get("/config")
async def github_oauth_config(
    service: GitHubConnectionService = Depends(get_github_connection_service),
) -> dict[str, object]:
    return service.oauth_public_config()


@auth_router.post("/use-env-token")
async def github_use_env_token(
    request: Request,
    return_to: str | None = Query(default=None),
) -> dict[str, object]:
    settings = reload_runtime_settings(request)
    service: GitHubConnectionService = request.app.state.github_connection_service
    try:
        record = service.create_env_token_connection(return_to)
    except GitHubOAuthError as exc:
        raise _http_error(exc) from exc
    return {
        "connected": True,
        "githubConnectionId": record.id,
        "username": record.github_login,
        "status": record.status,
        "mode": "env_token",
        "patTokenType": settings.github_pat_token_type,
        "canCreateRepositories": settings.github_pat_can_create_repositories,
    }


@router.get("/connect")
async def connect_github(
    return_to: str | None = Query(default=None),
    service: GitHubConnectionService = Depends(get_github_connection_service),
) -> RedirectResponse:
    return await _start_github_oauth(return_to=return_to, service=service)


@auth_router.get("/login")
async def login_github(
    return_to: str | None = Query(default=None),
    service: GitHubConnectionService = Depends(get_github_connection_service),
) -> RedirectResponse:
    return await _start_github_oauth(return_to=return_to, service=service)


@router.get("/callback")
async def github_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    service: GitHubConnectionService = Depends(get_github_connection_service),
    return_to: str | None = Query(default=None),
) -> RedirectResponse:
    return await _complete_github_oauth(
        code=code,
        state=state,
        return_to=return_to,
        service=service,
    )


@auth_router.get("/callback")
async def github_auth_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    service: GitHubConnectionService = Depends(get_github_connection_service),
    return_to: str | None = Query(default=None),
) -> RedirectResponse:
    return await _complete_github_oauth(
        code=code,
        state=state,
        return_to=return_to,
        service=service,
    )


@auth_router.get("/status")
async def github_status(
    github_connection_id: str | None = Query(default=None),
    service: GitHubConnectionService = Depends(get_github_connection_service),
) -> dict[str, object]:
    return service.connection_status(github_connection_id)


@auth_router.post("/disconnect")
async def github_disconnect(
    github_connection_id: str | None = Query(default=None),
    service: GitHubConnectionService = Depends(get_github_connection_service),
) -> dict[str, object]:
    return service.disconnect(github_connection_id)


@upload_router.post("/upload-project")
async def upload_project(
    request: GitHubUploadProjectRequest,
    service: GitHubConnectionService = Depends(get_github_connection_service),
) -> dict[str, Any]:
    if not request.githubConnectionId:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="GitHub connection is required.")
    try:
        auth = await service.exchange_for_workflow(
            request.githubConnectionId,
            task_id=request.projectId,
        )
    except GitHubOAuthError as exc:
        raise _http_error(exc) from exc

    repo_name = request.repoName or _repo_name_from_url(request.repoUrl) or f"mvpilot-generated-{request.projectId[:8]}"
    config: GitHubConfig = auth.config
    repo_result: dict[str, Any] | None = None
    if request.repoPreference == "create_new_repo":
        repo_result = create_repo(
            repo_name=repo_name,
            description=request.repoDescription or "Generated by MVPilot",
            visibility=request.visibility,
            config=config,
        )
        if repo_result.get("status") not in {"success", "mock"}:
            return _upload_error_response(repo_result)
    elif not request.repoUrl and not request.repoName:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Existing repo updates require repoName or repoUrl.",
        )

    commit_result = commit_files(
        repo_name=repo_name,
        files=[file.model_dump() for file in request.files],
        message=request.commitMessage,
        config=config,
        allow_existing_repo=request.repoPreference == "use_existing_repo",
    )
    if commit_result.get("status") not in {"success", "mock"}:
        return _upload_error_response(commit_result, repo_result=repo_result)

    output = commit_result.get("output", {})
    repo_output = (repo_result or {}).get("output", {})
    repo_url = repo_output.get("repo_url") or request.repoUrl or f"https://github.com/{auth.login}/{repo_name}"
    commit_sha = output.get("commit_sha")
    commit_url = f"{repo_url}/commit/{commit_sha}" if commit_sha else None
    return {
        "success": True,
        "repoUrl": repo_url,
        "commitUrl": commit_url,
        "branch": output.get("branch") or request.branch,
        "filesUploaded": output.get("changed_files") or [file.path for file in request.files],
        "errors": [],
    }


async def _start_github_oauth(
    *,
    return_to: str | None,
    service: GitHubConnectionService,
) -> RedirectResponse:
    try:
        pending = service.create_pending_connection(return_to)
    except GitHubOAuthError as exc:
        return RedirectResponse(service.redirect_url_for_error(return_to, str(exc)))
    return RedirectResponse(pending.authorization_url)


async def _complete_github_oauth(
    *,
    code: str | None,
    state: str | None,
    return_to: str | None,
    service: GitHubConnectionService,
) -> RedirectResponse:
    if not code or not state:
        return RedirectResponse(
            service.redirect_url_for_error(return_to, "GitHub did not return a valid authorization code.")
        )

    try:
        record = service.complete_callback(code=code, state=state)
    except GitHubOAuthError as exc:
        return RedirectResponse(service.redirect_url_for_error(return_to, str(exc)))

    # Redirect immediately so the browser is not stuck on the callback URL.
    # Token exchange runs during workflow launch (exchange_github_code node).
    return RedirectResponse(service.redirect_url_for_completed_callback(record))


def _repo_name_from_url(repo_url: str | None) -> str | None:
    if not repo_url:
        return None
    name = repo_url.rstrip("/").split("/")[-1].strip()
    return name or None


def _http_error(exc: GitHubOAuthError) -> HTTPException:
    message = str(exc) or "GitHub OAuth failed."
    status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if "not configured" in message.lower()
        else status.HTTP_400_BAD_REQUEST
    )
    return HTTPException(status_code=status_code, detail=message)


def _upload_error_response(
    result: dict[str, Any],
    *,
    repo_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = result.get("output") or {}
    return {
        "success": False,
        "repoUrl": (repo_result or {}).get("output", {}).get("repo_url"),
        "commitUrl": None,
        "branch": output.get("branch"),
        "filesUploaded": output.get("changed_files") or [],
        "errors": [result.get("error") or "GitHub upload failed."],
    }
