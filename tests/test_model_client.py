import json

import httpx
import pytest
import respx

from agent.config import Settings
from agent.idea_aware_partial import IdeaAwarePartialClient
from agent.model_client import DeterministicModelClient, ModelClientError, NemotronModelClient
from agent.prompts import build_file_manifest_prompt
from agent.model_outputs import (
    BlockerAnalysisOutput,
    DemoScriptOutput,
    FileManifestOutput,
    FinalReadmeOutput,
    MvpScopeOutput,
    PitchOutput,
    RepoPlanOutput,
)


PURPOSE_MODELS = [
    ("scope_mvp", MvpScopeOutput),
    ("plan_repo", RepoPlanOutput),
    ("file_manifest", FileManifestOutput),
    ("blocker_analysis", BlockerAnalysisOutput),
    ("final_readme", FinalReadmeOutput),
    ("demo_script", DemoScriptOutput),
    ("pitch", PitchOutput),
]


def live_settings(**overrides):
    values = {
        "_env_file": None,
        "adapter_mode": "live",
        "nvidia_api_key": "fake-nvidia-key",
        "nemotron_base_url": "https://nemotron.test/v1",
        "nemotron_timeout_seconds": 0.25,
        "nemotron_max_retries": 1,
        "nemotron_poll_attempts": 2,
        "nemotron_poll_interval_seconds": 0,
        **overrides,
    }
    return Settings(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize(("purpose", "response_model"), PURPOSE_MODELS)
async def test_deterministic_model_client_returns_all_structured_outputs(
    purpose,
    response_model,
):
    client = DeterministicModelClient(mode="mock")

    result = await client.complete_structured(
        purpose=purpose,
        model="mock-nemotron",
        prompt="Build a healthcare referral coordination agent.",
        response_model=response_model,
    )

    assert result.model == "mock-nemotron"
    assert result.purpose == purpose
    assert result.mode == "mock"
    assert result.latency_ms >= 0
    assert isinstance(result.output, response_model)
    assert result.output.decision_trace


@pytest.mark.asyncio
async def test_deterministic_mock_stays_explicitly_in_test_mode():
    client = DeterministicModelClient(
        mode="mock",
        fallback_reason="Unit test client.",
    )

    result = await client.complete_structured(
        purpose="scope_mvp",
        model="mock-nemotron",
        prompt="Build a healthcare referral coordination agent.",
        response_model=MvpScopeOutput,
    )

    assert result.mode == "mock"
    assert result.fallback_reason == "Unit test client."
    assert result.output.decision_trace[0].startswith("Grounded the scope")


@pytest.mark.asyncio
async def test_nemotron_client_posts_guided_json_and_parses_completion():
    settings = live_settings()
    content = MvpScopeOutput(
        target_user="clinic referral coordinator",
        must_have=["intake summary"],
        demo_boundary="single mocked clinic workflow",
        mode="live",
        decision_trace=["Scoped one visible referral workflow."],
    ).model_dump_json()

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": content}}]},
            )
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Scope this MVP.",
            response_model=MvpScopeOutput,
            max_tokens=500,
            reasoning_effort="medium",
        )

    posted = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == (
        "Bearer fake-nvidia-key"
    )
    assert posted["model"] == "nvidia/nemotron"
    assert posted["stream"] is False
    assert posted["guided_json"]["title"] == "MvpScopeOutput"
    assert posted["messages"][0]["role"] == "system"
    assert posted["messages"][1]["content"] == "Scope this MVP."
    assert posted["reasoning_effort"] == "medium"
    assert result.mode == "live"
    assert result.output.target_user == "clinic referral coordinator"
    assert "fake-nvidia-key" not in repr(result)


@pytest.mark.asyncio
async def test_nemotron_client_polls_status_after_accepted_response():
    settings = live_settings(nemotron_poll_attempts=1)
    content = RepoPlanOutput(
        files=["README.md"],
        test_plan=["unit workflow"],
        architecture_notes=["Keep adapters behind protocols."],
        mode="live",
        decision_trace=["Planned a small generated repo."],
    ).model_dump_json()

    with respx.mock(assert_all_called=True) as router:
        router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(202, json={"requestId": "req-123"})
        )
        status_route = router.get("https://nemotron.test/v1/status/req-123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "fulfilled",
                    "choices": [{"message": {"content": content}}],
                },
            )
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="plan_repo",
            model="nvidia/nemotron",
            prompt="Plan the repo.",
            response_model=RepoPlanOutput,
        )

    assert status_route.called
    assert result.mode == "live"
    assert result.output.files == ["README.md"]


@pytest.mark.asyncio
async def test_nemotron_client_uses_partial_degradation_when_live_key_is_missing():
    settings = live_settings(nvidia_api_key=None)

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert not route.called
    assert result.mode == "partial"
    assert result.fallback_reason == "Missing NVIDIA_API_KEY for live mode."
    assert result.output.decision_trace[0].startswith("Partial mode:")


@pytest.mark.asyncio
async def test_nemotron_client_fails_without_live_model_or_idea_specific_partial():
    settings = live_settings(
        nvidia_api_key=None,
        allow_idea_aware_partial=False,
    )

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={})
        )
        client = NemotronModelClient(settings)

        with pytest.raises(ModelClientError, match="Live Nemotron is required"):
            await client.complete_structured(
                purpose="scope_mvp",
                model="nvidia/nemotron",
                prompt="Scope this MVP.",
                response_model=MvpScopeOutput,
            )

    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        (httpx.Response(429, json={"error": "rate limited"}), "HTTP 429 from Nemotron."),
        (httpx.Response(500, json={"error": "down"}), "HTTP 500 from Nemotron."),
    ],
)
async def test_nemotron_client_retries_then_uses_partial_for_retryable_http_errors(
    response,
    expected_reason,
):
    settings = live_settings(nemotron_max_retries=1)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=response
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert route.call_count == 2
    assert result.mode == "partial"
    assert result.fallback_reason == expected_reason


@pytest.mark.asyncio
async def test_nemotron_client_retries_then_uses_partial_for_timeout():
    settings = live_settings(nemotron_max_retries=1)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("Nemotron timeout")
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert route.call_count == 2
    assert result.mode == "partial"
    assert result.fallback_reason == "Nemotron request timed out."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        (
            {"choices": [{"message": {"content": "{not valid json"}}]},
            "Nemotron returned invalid JSON.",
        ),
        (
            {"choices": [{"message": {"content": json.dumps({"target_user": "clinic"})}}]},
            "Nemotron response failed schema validation.",
        ),
    ],
)
async def test_nemotron_client_uses_partial_for_bad_structured_output(
    payload,
    expected_reason,
):
    settings = live_settings(nemotron_max_retries=0)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert result.mode == "partial"
    assert result.fallback_reason == expected_reason


@pytest.mark.asyncio
async def test_nemotron_client_uses_partial_when_pending_status_times_out():
    settings = live_settings(nemotron_poll_attempts=1)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(202, json={"requestId": "req-123"})
        )
        router.get("https://nemotron.test/v1/status/req-123").mock(
            return_value=httpx.Response(202, json={"status": "pending"})
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert result.mode == "partial"
    assert result.fallback_reason == "Nemotron request stayed pending."


@pytest.mark.asyncio
async def test_idea_aware_partial_file_manifest_uses_build_context():
    client = IdeaAwarePartialClient(partial_reason="test")
    prompt = build_file_manifest_prompt(
        idea="Study planner for college students",
        repo_plan={"files": [{"path": "frontend/app/page.tsx"}]},
        build_context={
            "resolvedTechStack": {"items": ["React", "FastAPI"]},
            "sourceContext": {"warnings": [{"code": "no_uploads", "message": "No files uploaded."}]},
        },
    )
    result = await client.complete_structured(
        purpose="file_manifest",
        model="test",
        prompt=prompt,
        response_model=FileManifestOutput,
    )
    assert result.mode == "partial"
    assert len(result.output.artifacts) >= 1
