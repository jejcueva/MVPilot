from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile

from agent.dependencies import get_agent_service
from agent.schemas import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    MAX_UPLOADED_FILES,
    RunAgentRequest,
    RunAgentResponse,
    TaskDetailResponse,
)
from agent.frontend_intake import MAX_TEXT_FILE_BYTES
from agent.service import AgentService
from agent.task_store import ApprovalNotFoundError, TaskNotFoundError

router = APIRouter(prefix="/agent")
orchestrator_router = APIRouter(prefix="/api/orchestrator")
FORM_CONTENT_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data")


@router.post(
    "/run",
    response_model=RunAgentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_agent(
    request: Request,
    background_tasks: BackgroundTasks,
    service: AgentService = Depends(get_agent_service),
) -> RunAgentResponse:
    run_request = await _parse_run_agent_request(request)
    return await service.start_task(run_request, background_tasks=background_tasks)


@orchestrator_router.post(
    "/start-project",
    response_model=RunAgentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_project(
    request: Request,
    background_tasks: BackgroundTasks,
    service: AgentService = Depends(get_agent_service),
) -> RunAgentResponse:
    run_request = await _parse_run_agent_request(request)
    return await service.start_task(run_request, background_tasks=background_tasks)


@router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task(
    task_id: str,
    service: AgentService = Depends(get_agent_service),
) -> TaskDetailResponse:
    try:
        return await service.get_task_detail(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        ) from exc


@router.post("/approve", response_model=ApprovalDecisionResponse)
async def approve_action(
    request: ApprovalDecisionRequest,
    service: AgentService = Depends(get_agent_service),
) -> ApprovalDecisionResponse:
    try:
        return await service.approve_action(request)
    except TaskNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        ) from exc
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found",
        ) from exc


async def _parse_run_agent_request(request: Request) -> RunAgentRequest:
    content_type = request.headers.get("content-type", "").lower()
    if any(form_type in content_type for form_type in FORM_CONTENT_TYPES):
        payload = await _read_form_payload(request)
    else:
        payload = await _read_json_payload(request)

    try:
        return RunAgentRequest.model_validate(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        ) from exc

    if not isinstance(payload, dict):
        raise RequestValidationError(
            [
                {
                    "type": "model_attributes_type",
                    "loc": ("body",),
                    "msg": "Input should be a valid object",
                    "input": payload,
                }
            ]
        )
    return _normalize_run_payload(payload)


async def _read_form_payload(request: Request) -> dict[str, Any]:
    form = await request.form()
    additional_files, uploaded_file_contents = await _form_file_payloads(
        form,
        "additional_files",
    )
    payload = {
        "idea": form.get("idea"),
        "repo_visibility": form.get("repo_visibility") or form.get("visibility") or "public",
        "repo_preference": form.get("repo_preference") or form.get("repoPreference") or "create_new_repo",
        "repo_name": form.get("repo_name") or form.get("repoName"),
        "repo_description": form.get("repo_description") or form.get("repoDescription"),
        "repo_url": form.get("repo_url") or form.get("repoUrl"),
        "branch": form.get("branch") or "main",
        "demo_mode": form.get("demo_mode") or False,
        "title": form.get("title"),
        "primary_rules_url": form.get("primary_rules_url") or form.get("rules_url") or form.get("rulesUrl"),
        "rules_url": form.get("rules_url"),
        "additional_urls": _form_text_list(form, "additional_urls")
        + _form_text_list(form, "referenceUrls")
        + _form_text_list(form, "reference_urls"),
        "additional_files": additional_files,
        "uploaded_file_contents": uploaded_file_contents,
        "source": form.get("source"),
        "github_connected": bool(
            form.get("github_auth_code") or _truthy_form_value(form.get("github_connected"))
        ),
        "github_connection_id": form.get("github_connection_id"),
        "target_users": form.get("target_users") or form.get("targetUsers"),
        "tech_stack_preference": form.get("tech_stack_preference") or form.get("techStackPreference"),
        "required_features": _form_text_list(form, "required_features")
        + _form_text_list(form, "requiredFeatures"),
        "project_depth": form.get("project_depth")
        or form.get("projectDepth")
        or "Advanced Project",
        "target_platform": form.get("target_platform")
        or form.get("targetPlatform")
        or "web app",
        "use_openclaw_orchestration": (
            _truthy_form_value(
                form.get("use_openclaw_orchestration")
                or form.get("useOpenClawOrchestration")
            )
            if form.get("use_openclaw_orchestration") is not None
            or form.get("useOpenClawOrchestration") is not None
            else True
        ),
    }
    return _normalize_run_payload(payload)


def _normalize_run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {**payload}
    if not normalized.get("primary_rules_url") and normalized.get("rules_url"):
        normalized["primary_rules_url"] = normalized.get("rules_url")

    github_auth_code = normalized.pop("github_auth_code", None)
    if github_auth_code:
        normalized["github_connected"] = True

    return normalized


def _form_text_list(form: FormData, field_name: str) -> list[str]:
    values = form.getlist(field_name)
    return [
        str(value).strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


async def _form_file_payloads(
    form: FormData,
    field_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = [
        value
        for value in form.getlist(field_name)
        if isinstance(value, UploadFile) and value.filename
    ]
    if len(files) > MAX_UPLOADED_FILES:
        raise RequestValidationError(
            [
                {
                    "type": "too_long",
                    "loc": ("body", field_name),
                    "msg": f"At most {MAX_UPLOADED_FILES} uploaded files are supported",
                    "input": len(files),
                }
            ]
        )

    metadata: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    for file in files:
        if file.size is not None and file.size > MAX_TEXT_FILE_BYTES:
            content = b""
            size_bytes = file.size
        else:
            content = await file.read()
            await file.seek(0)
            size_bytes = len(content)
        item = {
            "name": file.filename,
            "content_type": file.content_type or "application/octet-stream",
            "size_bytes": size_bytes,
        }
        metadata.append(item)
        contents.append({**item, "content": content})

    return metadata, contents


def _truthy_form_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
