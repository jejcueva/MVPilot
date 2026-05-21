import json

import httpx
import pytest
import respx

from agent.config import Settings
from agent.idea_aware_partial import IdeaAwarePartialClient
from agent.model_client import (
    DeterministicModelClient,
    ModelClientError,
    NemotronModelClient,
    _parse_json_text,
)
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


INVALID_JSON_SNIPPET = "{not valid json"


def _sample_project_requirements(**overrides) -> MvpScopeOutput:
    payload = {
        "target_users": "students",
        "user_personas": ["students", "Admin"],
        "core_features": ["planner", "dashboard", "auth"],
        "success_criteria": ["Complete the primary workflow"],
        "mode": "live",
        "decision_trace": ["scoped"],
    }
    payload.update(overrides)
    return MvpScopeOutput(**payload)


TRUNCATED_REPO_PLAN_JSON = (
    '{"files": ["README.md"], "test_plan": ["pytest"], "decision_trace": ["partial"], '
    '"mode": "live", "architecture_notes": ["note'
)


def test_fast_fallback_uses_shorter_live_limits() -> None:
    settings = Settings(
        _env_file=None,
        allow_idea_aware_partial=True,
        nemotron_fast_fallback=True,
        nemotron_timeout_seconds=300,
        nemotron_max_retries=3,
        nemotron_poll_max_seconds=600,
        nemotron_live_attempt_timeout_seconds=75,
        nemotron_fast_fallback_max_retries=0,
        nemotron_fast_fallback_poll_max_seconds=90,
    )
    assert settings.nemotron_fast_fallback_active is True
    assert settings.nemotron_effective_timeout_seconds == 75
    assert settings.nemotron_effective_max_retries == 0
    assert settings.nemotron_effective_poll_max_seconds == 90


def test_file_manifest_gets_longer_timeout_in_strict_live() -> None:
    settings = Settings(
        _env_file=None,
        allow_idea_aware_partial=False,
        nemotron_file_manifest_timeout_seconds=1200,
        nemotron_timeout_seconds=900,
    )
    assert settings.nemotron_read_timeout_seconds("file_manifest") == 1200
    assert settings.nemotron_poll_max_seconds_for("file_manifest") == 3600


def test_strict_live_keeps_long_limits_when_partial_disabled() -> None:
    settings = Settings(
        _env_file=None,
        allow_idea_aware_partial=False,
        nemotron_fast_fallback=True,
        nemotron_timeout_seconds=900,
        nemotron_strict_timeout_seconds=900,
        nemotron_max_retries=3,
        nemotron_poll_max_seconds=3600,
        nemotron_live_attempt_timeout_seconds=75,
    )
    assert settings.nemotron_fast_fallback_active is False
    assert settings.nemotron_effective_timeout_seconds == 900
    assert settings.nemotron_effective_max_retries == 3
    assert settings.nemotron_effective_poll_max_seconds == 3600


def live_settings(**overrides):
    values = {
        "_env_file": None,
        "adapter_mode": "live",
        "allow_idea_aware_partial": True,
        "nemotron_fast_fallback": False,
        "nvidia_api_key": "fake-nvidia-key",
        "nemotron_base_url": "https://nemotron.test/v1",
        "nemotron_timeout_seconds": 0.25,
        "nemotron_max_retries": 1,
        "nemotron_poll_attempts": 2,
        "nemotron_poll_interval_seconds": 0,
        "nemotron_poll_max_seconds": 0.01,
        "nemotron_reasoning_effort": "none",
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
    assert result.output.decision_trace[0].startswith("Grounded requirements")


@pytest.mark.asyncio
async def test_nemotron_client_posts_schema_prompt_and_parses_completion():
    settings = live_settings()
    content = _sample_project_requirements(
        target_users="clinic referral coordinator",
        user_personas=["clinic referral coordinator", "Admin"],
        core_features=["intake summary", "referral routing", "status dashboard"],
        success_criteria=["Complete the clinic workflow"],
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
        )

    posted = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == (
        "Bearer fake-nvidia-key"
    )
    assert posted["model"] == "nvidia/nemotron"
    assert posted["stream"] is False
    assert posted["guided_json"]["title"] == "ProjectRequirementsOutput"
    assert posted["messages"][0]["role"] == "system"
    assert "guided schema" in posted["messages"][0]["content"]
    assert posted["messages"][1]["content"] == "Scope this MVP."
    assert posted["reasoning_effort"] == "none"
    assert result.mode == "live"
    assert result.output.target_users == "clinic referral coordinator"
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


def test_parse_json_text_strips_markdown_fences() -> None:
    content = _sample_project_requirements().model_dump_json()
    parsed = _parse_json_text(f"```json\n{content}\n```")
    assert parsed["target_users"] == "students"


@pytest.mark.asyncio
async def test_nemotron_client_salvages_json_wrapped_in_markdown_fence():
    settings = live_settings()
    content = _sample_project_requirements().model_dump_json()

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": f"```json\n{content}\n```"}}
                    ]
                },
            )
        )
        client = NemotronModelClient(settings)
        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Scope this MVP.",
            response_model=MvpScopeOutput,
        )

        assert route.called
        assert result.mode == "live"
        assert result.output.target_users == "students"


@pytest.mark.asyncio
async def test_nemotron_client_fast_fallback_times_out_then_uses_partial():
    settings = live_settings(
        allow_idea_aware_partial=True,
        nemotron_fast_fallback=True,
        nemotron_live_attempt_timeout_seconds=0.01,
        nemotron_fast_fallback_max_retries=0,
        nemotron_fast_fallback_poll_max_seconds=0.01,
        nemotron_timeout_seconds=0.01,
        nemotron_poll_max_seconds=0.01,
    )

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

    assert route.call_count == 1
    assert result.mode in {"partial", "degraded"}
    assert "degraded" in (result.fallback_reason or "").lower()


@pytest.mark.asyncio
async def test_nemotron_client_retries_invalid_json_before_partial():
    settings = live_settings(nemotron_max_retries=2)
    valid = _sample_project_requirements().model_dump_json()

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": INVALID_JSON_SNIPPET}}
                        ]
                    },
                ),
                httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "{still bad"}}]},
                ),
                httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": valid}}]},
                ),
            ]
        )
        client = NemotronModelClient(settings)
        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Scope this MVP.",
            response_model=MvpScopeOutput,
        )

    assert route.call_count == 3
    assert result.mode == "live"
    assert result.output.target_users == "students"


@pytest.mark.asyncio
async def test_nemotron_client_strict_mode_raises_after_json_retries_exhausted():
    settings = live_settings(
        nemotron_max_retries=1,
        allow_idea_aware_partial=False,
    )

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": INVALID_JSON_SNIPPET}}]
                },
            )
        )
        client = NemotronModelClient(settings)

        with pytest.raises(ModelClientError, match="invalid JSON"):
            await client.complete_structured(
                purpose="scope_mvp",
                model="nvidia/nemotron",
                prompt="Scope this MVP.",
                response_model=MvpScopeOutput,
            )

    assert route.call_count == 2


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
    assert result.mode in {"partial", "degraded"}
    assert result.fallback_reason == "Missing NVIDIA_API_KEY for live mode."
    assert result.output.decision_trace[0].startswith("Degraded mode:")


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
    assert result.mode in {"partial", "degraded"}
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
    assert result.mode in {"partial", "degraded"}
    assert result.fallback_reason == "Nemotron request timed out."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        (
            {"choices": [{"message": {"content": INVALID_JSON_SNIPPET}}]},
            "Nemotron returned invalid JSON.",
        ),
        (
            {"choices": [{"message": {"content": json.dumps({"target_users": "clinic"})}}]},
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
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert route.call_count == 1
    assert result.mode in {"partial", "degraded"}
    assert result.fallback_reason == expected_reason


@pytest.mark.asyncio
async def test_nemotron_client_retries_bad_structured_output_before_partial():
    settings = live_settings(nemotron_max_retries=1)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": INVALID_JSON_SNIPPET}}]
                },
            )
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="scope_mvp",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students plan study sessions.\n\n",
            response_model=MvpScopeOutput,
        )

    assert route.call_count == 2
    assert result.mode in {"partial", "degraded"}
    assert result.fallback_reason == "Nemotron returned invalid JSON."


@pytest.mark.asyncio
async def test_nemotron_client_retries_truncated_plan_repo_with_higher_max_tokens():
    settings = live_settings(
        nemotron_max_retries=1,
        nemotron_repo_plan_max_tokens=2000,
    )
    valid = RepoPlanOutput(
        files=["README.md"],
        test_plan=["pytest"],
        mode="live",
        decision_trace=["Planned repo layout."],
    ).model_dump_json()
    captured_tokens: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_tokens.append(body["max_tokens"])
        if len(captured_tokens) == 1:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": TRUNCATED_REPO_PLAN_JSON}}]},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": valid}}]},
        )

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://nemotron.test/v1/chat/completions").mock(
            side_effect=handler
        )
        client = NemotronModelClient(settings)

        result = await client.complete_structured(
            purpose="plan_repo",
            model="nvidia/nemotron",
            prompt="Idea:\nStudyPilot helps students.\n\n",
            response_model=RepoPlanOutput,
            max_tokens=2000,
        )

    assert route.call_count == 2
    assert captured_tokens == [2000, 4000]
    assert result.mode == "live"


@pytest.mark.asyncio
async def test_nemotron_file_manifest_504_raises_when_live_manifest_required():
    settings = live_settings(
        allow_idea_aware_partial=True,
        require_live_file_manifest=True,
        nemotron_max_retries=0,
        nemotron_file_manifest_max_retries=0,
    )

    with respx.mock(assert_all_called=True) as router:
        router.post("https://nemotron.test/v1/chat/completions").mock(
            return_value=httpx.Response(504, json={"error": "gateway timeout"})
        )
        client = NemotronModelClient(settings)

        with pytest.raises(ModelClientError, match="504"):
            await client.complete_structured(
                purpose="file_manifest",
                model="nvidia/nemotron",
                prompt="Idea:\nStudyPilot helps students.\n\n",
                response_model=FileManifestOutput,
            )


@pytest.mark.asyncio
async def test_nemotron_client_uses_partial_when_pending_status_times_out():
    settings = live_settings(nemotron_poll_attempts=1, nemotron_poll_max_seconds=0.01)

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

    assert result.mode in {"partial", "degraded"}
    assert result.fallback_reason == "Nemotron request stayed pending."


@pytest.mark.asyncio
async def test_idea_aware_partial_file_manifest_uses_build_context():
    client = IdeaAwarePartialClient(partial_reason="test")
    prompt = build_file_manifest_prompt(
        idea="Study planner for college students",
        project_requirements={
            "vertical_pack": "planner",
            "demo_path": [{"step": "1", "screen": "Plan", "action": "Review", "api": "GET /api/items"}],
            "api_routes": ["/api/items"],
        },
        architecture_plan={"files": [{"path": "frontend/app/page.tsx"}]},
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
    assert result.mode in {"partial", "degraded"}
    assert len(result.output.artifacts) >= 1
