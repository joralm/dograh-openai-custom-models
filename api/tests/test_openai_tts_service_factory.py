from types import SimpleNamespace
from unittest.mock import patch

from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_tts_service


def test_create_openai_tts_service_uses_openai_pcm_sample_rate():
    user_config = SimpleNamespace(
        tts=SimpleNamespace(
            provider=ServiceProviders.OPENAI.value,
            api_key="test-key",
            model="gpt-4o-mini-tts",
            voice="alloy",
            base_url=None,
        )
    )
    audio_config = SimpleNamespace(
        transport_out_sample_rate=16000,
        transport_in_sample_rate=16000,
    )

    with patch("api.services.pipecat.service_factory.OpenAITTSService") as mock_service:
        create_tts_service(user_config, audio_config)

    assert mock_service.call_count == 1
    kwargs = mock_service.call_args.kwargs
    assert kwargs["sample_rate"] == OPENAI_SAMPLE_RATE
    assert kwargs["settings"].model == "gpt-4o-mini-tts"


def test_openai_tts_schema_defaults_and_allows_custom_model():
    from api.services.configuration.registry import OpenAITTSService

    schema = OpenAITTSService.model_json_schema()["properties"]["model"]
    assert schema["default"] == "gpt-4o-mini-tts"
    assert "gpt-4o-mini-tts" in schema["examples"]
    assert schema["allow_custom_input"] is True

    cfg = OpenAITTSService(
        api_key="k", model="tts-eu-pt", voice="pt-PT", base_url="http://x/v1"
    )
    assert cfg.model == "tts-eu-pt"
    assert OpenAITTSService(api_key="k").model == "gpt-4o-mini-tts"


def test_create_openai_tts_service_passes_custom_model_unchanged():
    user_config = SimpleNamespace(
        tts=SimpleNamespace(
            provider=ServiceProviders.OPENAI.value,
            api_key="test-key",
            model="tts-eu-pt",
            voice="pt-PT",
            base_url=None,
        )
    )
    audio_config = SimpleNamespace(
        transport_out_sample_rate=16000,
        transport_in_sample_rate=16000,
    )

    with patch("api.services.pipecat.service_factory.OpenAITTSService") as mock_service:
        create_tts_service(user_config, audio_config)

    settings = mock_service.call_args.kwargs["settings"]
    assert settings.model == "tts-eu-pt"
    assert settings.voice == "pt-PT"


def _tts_config(voice, base_url):
    return (
        SimpleNamespace(
            tts=SimpleNamespace(
                provider=ServiceProviders.OPENAI.value,
                api_key="test-key",
                model="tts-eu-pt",
                voice=voice,
                base_url=base_url,
            )
        ),
        SimpleNamespace(transport_out_sample_rate=16000, transport_in_sample_rate=16000),
    )


def test_custom_endpoint_keeps_pt_PT_spelling_and_allows_custom_voice():
    user_config, audio_config = _tts_config("pt-PT", "http://192.168.2.102:18500/v1")
    with (
        patch("api.services.pipecat.service_factory._validate_runtime_service_url"),
        patch("api.services.pipecat.service_factory.OpenAITTSService") as mock_service,
    ):
        create_tts_service(user_config, audio_config)

    kwargs = mock_service.call_args.kwargs
    assert kwargs["settings"].voice == "pt-PT"
    assert kwargs["allow_custom_voice"] is True


def test_custom_endpoint_with_empty_voice_is_rejected():
    import pytest

    for voice in ("", "  ", None):
        user_config, audio_config = _tts_config(voice, "http://192.168.2.102:18500/v1")
        with (
            patch("api.services.pipecat.service_factory._validate_runtime_service_url"),
            patch("api.services.pipecat.service_factory.OpenAITTSService"),
            pytest.raises(Exception, match="voice is required"),
        ):
            create_tts_service(user_config, audio_config)


def test_official_endpoint_does_not_allow_custom_voice():
    user_config, audio_config = _tts_config("alloy", "https://api.openai.com/v1")
    with patch("api.services.pipecat.service_factory.OpenAITTSService") as mock_service:
        create_tts_service(user_config, audio_config)

    kwargs = mock_service.call_args.kwargs
    assert kwargs["settings"].voice == "alloy"
    assert kwargs["allow_custom_voice"] is False


def test_openai_tts_voice_schema_keeps_official_voices_and_allows_custom():
    from api.services.configuration.registry import OpenAITTSService

    schema = OpenAITTSService.model_json_schema()["properties"]["voice"]
    assert schema["default"] == "alloy"
    assert "alloy" in schema["examples"]
    assert schema["allow_custom_input"] is True


def _run_tts(voice, allow_custom_voice):
    import asyncio
    from unittest.mock import MagicMock

    from pipecat.frames.frames import ErrorFrame

    from api.services.pipecat.service_factory import OpenAITTSService

    captured = {}

    class _Resp:
        status_code = 200

        async def iter_bytes(self, size):
            yield b"\x00\x01"

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    def create(**params):
        captured.update(params)
        return _Ctx()

    from pipecat.services.openai.tts import OpenAITTSSettings

    svc = OpenAITTSService(
        api_key="k",
        base_url="http://localhost:1/v1",
        settings=OpenAITTSSettings(model="tts-eu-pt", voice=voice),
        allow_custom_voice=allow_custom_voice,
    )
    svc._client = MagicMock()
    svc._client.audio.speech.with_streaming_response.create = create
    svc.start_tts_usage_metrics = lambda text: asyncio.sleep(0)
    svc.stop_ttfb_metrics = lambda: asyncio.sleep(0)

    async def collect():
        return [f async for f in svc.run_tts("olá", "ctx")]

    frames = asyncio.run(collect())
    return captured, frames, ErrorFrame


def test_run_tts_sends_pt_PT_verbatim_on_custom_endpoint():
    captured, frames, ErrorFrame = _run_tts("pt-PT", True)
    assert captured["voice"] == "pt-PT"
    assert captured["model"] == "tts-eu-pt"
    assert not any(isinstance(f, ErrorFrame) for f in frames)


def test_run_tts_rejects_unknown_voice_on_official_endpoint():
    captured, frames, ErrorFrame = _run_tts("pt-PT", False)
    assert captured == {}
    assert any(isinstance(f, ErrorFrame) for f in frames)
