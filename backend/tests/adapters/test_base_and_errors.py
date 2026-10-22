import dataclasses

import pytest

from app.adapters.base import GenerationRequest, ModelAdapter, StreamChunk, elapsed_ms
from app.adapters.errors import (
    AdapterError,
    AuthenticationError,
    InvalidRequestError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    error_from_status,
    parse_retry_after,
    unsupported_options_error,
)


def test_request_has_sensible_defaults() -> None:
    request = GenerationRequest(model="m", prompt="hi")

    assert request.system_prompt is None
    assert request.temperature is None
    assert request.max_tokens == 1024
    assert request.effort is None
    assert request.options == {}


def test_requests_do_not_share_their_options_dict() -> None:
    first = GenerationRequest(model="m", prompt="hi")
    second = GenerationRequest(model="m", prompt="hi")

    assert first.options is not second.options


def test_request_is_immutable() -> None:
    request = GenerationRequest(model="m", prompt="hi")

    with pytest.raises(dataclasses.FrozenInstanceError):
        request.model = "other"  # type: ignore[misc]


def test_stream_chunk_defaults_to_a_normal_chunk() -> None:
    chunk = StreamChunk(text="Hel")

    assert chunk.is_final is False
    assert chunk.input_tokens is None


def test_adapter_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]


def test_rate_limit_error_keeps_retry_after() -> None:
    error = RateLimitError("slow down", provider="anthropic", retry_after=2.5)

    assert error.retry_after == 2.5
    assert error.provider == "anthropic"
    assert str(error) == "slow down"


def test_rate_limit_error_retry_after_is_optional() -> None:
    assert RateLimitError("slow down", provider="openai").retry_after is None


@pytest.mark.parametrize(
    "error_class",
    [
        AuthenticationError,
        InvalidRequestError,
        ProviderUnavailableError,
        QuotaExceededError,
        RateLimitError,
    ],
)
def test_all_errors_can_be_caught_as_adapter_error(error_class: type[AdapterError]) -> None:
    with pytest.raises(AdapterError):
        raise error_class("boom", provider="ollama")


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        (400, InvalidRequestError),
        (404, InvalidRequestError),
        (413, InvalidRequestError),
        (408, ProviderUnavailableError),
        (409, ProviderUnavailableError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
        (529, ProviderUnavailableError),
    ],
)
def test_error_from_status_picks_the_right_error(status: int, error_class: type) -> None:
    error = error_from_status(status, "boom", "openai")

    assert type(error) is error_class
    assert error.provider == "openai"


def test_error_from_status_passes_retry_after_to_rate_limit() -> None:
    error = error_from_status(429, "slow down", "openai", retry_after=4.0)

    assert isinstance(error, RateLimitError)
    assert error.retry_after == 4.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7", 7.0), ("2.5", 2.5), (None, None), ("", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None)],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    assert parse_retry_after(value) == expected


def test_elapsed_ms_measures_time_since_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.adapters.base.time.perf_counter", lambda: 12.5)

    assert elapsed_ms(10.0) == 2500.0


def test_results_default_to_unknown_finish_reason() -> None:
    assert StreamChunk(text="x").finish_reason is None


def test_unsupported_options_error_names_the_provider() -> None:
    error = unsupported_options_error("openai")

    assert isinstance(error, InvalidRequestError)
    assert error.provider == "openai"
    assert "options" in str(error)
