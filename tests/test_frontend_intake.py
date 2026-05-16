import pytest
from pydantic import ValidationError

from agent.frontend_intake import (
    FetchedUrl,
    FrontendIntake,
    SourceFetchError,
    build_frontend_intake_from_request,
    build_source_context,
)
from agent.schemas import RunAgentRequest, UploadedSourceFileContent


def test_frontend_intake_normalizes_request_fields() -> None:
    request = RunAgentRequest(
        title="  Study Planner  ",
        idea="  Build a study planner.  ",
        repo_visibility="public",
        source="  mvpilot_frontend  ",
        primary_rules_url="  https://example.com/rules  ",
        additional_urls=[" ", " https://example.com/judging ", ""],
        additional_files=[
            {
                "name": "notes.md",
                "content_type": "text/markdown",
                "size_bytes": 12,
            }
        ],
        github_connected=True,
        github_connection_id="  gh_conn_123  ",
    )

    intake = build_frontend_intake_from_request(request)

    assert intake.model_dump() == {
        "title": "Study Planner",
        "idea": "Build a study planner.",
        "source": "mvpilot_frontend",
        "primaryRulesUrl": "https://example.com/rules",
        "additionalUrls": ["https://example.com/judging"],
        "uploadedFiles": [
            {
                "name": "notes.md",
                "content_type": "text/markdown",
                "size_bytes": 12,
            }
        ],
        "githubConnected": True,
        "githubConnectionId": "gh_conn_123",
        "repoPreference": "create_new_repo",
        "repoName": None,
        "repoDescription": None,
        "repoUrl": None,
        "visibility": "public",
        "branch": "main",
        "targetUsers": None,
        "techStackPreference": None,
        "requiredFeatures": [],
    }


def test_frontend_intake_rejects_too_many_urls_and_files() -> None:
    with pytest.raises(ValidationError):
        RunAgentRequest(
            idea="Build a study planner.",
            repo_visibility="public",
            additional_urls=[f"https://example.com/{index}" for index in range(6)],
        )

    with pytest.raises(ValidationError):
        RunAgentRequest(
            idea="Build a study planner.",
            repo_visibility="public",
            additional_files=[
                {
                    "name": f"notes-{index}.md",
                    "content_type": "text/markdown",
                    "size_bytes": 10,
                }
                for index in range(6)
            ],
        )


@pytest.mark.asyncio
async def test_source_context_includes_primary_url_summary_when_fetch_succeeds() -> None:
    intake = FrontendIntake(
        title="Study Planner",
        idea="Build a study planner.",
        primaryRulesUrl="https://example.com/rules",
    )

    async def fake_fetch(url: str) -> FetchedUrl:
        return FetchedUrl(
            url=url,
            title="Hackathon Rules",
            content_type="text/markdown",
            text="# Hackathon Rules\n\nMust include a working demo and README.",
        )

    context = await build_source_context(intake, fetch_url=fake_fetch)

    assert context["primaryRulesUrl"]["url"] == "https://example.com/rules"
    assert "Must include a working demo" in context["primaryRulesUrl"]["summary"]
    assert context["sourceCounts"]["fetchedUrls"] == 1
    assert context["warnings"] == []


@pytest.mark.asyncio
async def test_source_context_records_additional_url_failures_as_warnings() -> None:
    intake = FrontendIntake(
        title="Study Planner",
        idea="Build a study planner.",
        additionalUrls=["https://example.com/missing"],
    )

    async def failing_fetch(url: str) -> FetchedUrl:
        raise SourceFetchError(url, "HTTP 404")

    context = await build_source_context(intake, fetch_url=failing_fetch)

    assert context["additionalUrls"] == []
    assert context["warnings"] == [
        {
            "source": "https://example.com/missing",
            "message": "Could not read submitted URL: HTTP 404",
        }
    ]


@pytest.mark.asyncio
async def test_source_context_summarizes_supported_text_uploads() -> None:
    intake = FrontendIntake(title="Study Planner", idea="Build a study planner.")
    uploaded = UploadedSourceFileContent(
        name="notes.md",
        content_type="text/markdown",
        size_bytes=43,
        content=b"# Notes\nUse spaced repetition and quiz cards.",
    )

    context = await build_source_context(intake, uploaded_files=[uploaded])

    assert context["uploadedFiles"][0]["name"] == "notes.md"
    assert "spaced repetition" in context["uploadedFiles"][0]["summary"]
    assert context["sourceCounts"]["summarizedFiles"] == 1
    assert context["warnings"] == []


@pytest.mark.asyncio
async def test_source_context_warns_for_unsupported_uploads_without_failing() -> None:
    intake = FrontendIntake(title="Study Planner", idea="Build a study planner.")
    uploaded = UploadedSourceFileContent(
        name="rubric.pdf",
        content_type="application/pdf",
        size_bytes=200,
        content=b"%PDF-1.7",
    )

    context = await build_source_context(intake, uploaded_files=[uploaded])

    assert context["uploadedFiles"] == []
    assert context["warnings"] == [
        {
            "source": "rubric.pdf",
            "message": "Text extraction is not supported for .pdf files yet.",
        }
    ]
