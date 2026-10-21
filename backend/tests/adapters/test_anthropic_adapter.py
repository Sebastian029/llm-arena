import json

import anthropic
import httpx2  # the HTTP library used by anthropic 2.x; we only need its fake transport
import pytest

from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.base import GenerationRequest
from app.adapters.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
)

pytestmark = pytest.mark.anyio

MODEL = "claude-haiku-5-5"
REQUEST = GenerationRequest(
    model=MODEL, prompt="Hi", system_prompt="Be brief.", temperature=0.2, max_tokens=50
)


def make_adapter(handler) -> AnthropicAdapter:
    """Build an adapter whose SDK client answers with `handler` instead of using the network."""
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = anthropic.AsyncAnthropic(api_key="test-key", http_client=http_client, max_retries=0)
    return AnthropicAdapter(client=client)


def message_json(text: str = "Hello!", input_tokens: int = 10, output_tokens: int = 5) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def error_json(error_type: str, message: str) -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def sse(*events: tuple[str, dict]) -> bytes:
    """Format events the way the API streams them (Server-Sent Events)."""
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def stream_response(*texts: str) -> httpx2.Response:
    events: list[tuple[str, dict]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **message_json(text="", input_tokens=12, output_tokens=1),
                    "content": [],
                    "stop_reason": None,
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for text in texts:
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(*events))


async def test_generate_returns_text_tokens_and_latency() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message_json("Hello!", 10, 5))

    result = await make_adapter(handler).generate(REQUEST)

    assert result.text == "Hello!"
    assert result.model == MODEL
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0


async def test_generate_sends_system_separately_and_no_temperature() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=message_json())

    await make_adapter(handler).generate(REQUEST)

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/messages"
    assert body["model"] == MODEL
    assert body["max_tokens"] == 50
    assert body["system"] == "Be brief."
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert "temperature" not in body  # the API rejects it


async def test_generate_without_system_prompt_omits_system() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=message_json())

    await make_adapter(handler).generate(GenerationRequest(model=MODEL, prompt="Hi"))

    assert "system" not in json.loads(seen[0].content)


async def test_stream_yields_text_chunks_then_final_chunk_with_usage() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content)["stream"] is True
        return stream_response("Hel", "lo")

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert [c.text for c in chunks] == ["Hel", "lo", ""]
    assert [c.is_final for c in chunks] == [False, False, True]
    assert chunks[-1].input_tokens == 12
    assert chunks[-1].output_tokens == 7
    assert chunks[-1].latency_ms >= 0


@pytest.mark.parametrize(
    ("status", "error_type", "error_class"),
    [
        (400, "invalid_request_error", InvalidRequestError),
        (404, "not_found_error", InvalidRequestError),
        (401, "authentication_error", AuthenticationError),
        (403, "permission_error", AuthenticationError),
        (429, "rate_limit_error", RateLimitError),
        (500, "api_error", ProviderUnavailableError),
        (529, "overloaded_error", ProviderUnavailableError),
    ],
)
async def test_http_errors_become_adapter_errors(
    status: int, error_type: str, error_class: type
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=error_json(error_type, "something went wrong"))

    with pytest.raises(error_class, match="something went wrong"):
        await make_adapter(handler).generate(REQUEST)


async def test_http_errors_are_also_mapped_when_streaming() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json=error_json("not_found_error", "model: nope"))

    with pytest.raises(InvalidRequestError, match="nope"):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_rate_limit_error_reads_retry_after_header() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429, headers={"retry-after": "7"}, json=error_json("rate_limit_error", "slow down")
        )

    with pytest.raises(RateLimitError) as exc_info:
        await make_adapter(handler).generate(REQUEST)

    assert exc_info.value.retry_after == 7.0


async def test_rate_limit_without_header_has_no_retry_after() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, json=error_json("rate_limit_error", "slow down"))

    with pytest.raises(RateLimitError) as exc_info:
        await make_adapter(handler).generate(REQUEST)

    assert exc_info.value.retry_after is None


async def test_timeout_becomes_provider_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow")

    with pytest.raises(ProviderUnavailableError):
        await make_adapter(handler).generate(REQUEST)


async def test_connection_error_becomes_provider_unavailable_when_streaming() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_generate_joins_text_blocks_and_skips_other_blocks() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        blocks = [
            {"type": "text", "text": "Hello, "},
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "world"},
        ]
        return httpx2.Response(200, json=message_json() | {"content": blocks})

    assert (await make_adapter(handler).generate(REQUEST)).text == "Hello, world"


async def test_error_event_inside_stream_becomes_provider_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = sse(("error", error_json("overloaded_error", "Overloaded")))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    with pytest.raises(ProviderUnavailableError, match="Overloaded"):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_use"),
    ],
)
async def test_generate_translates_stop_reason_to_finish_reason(
    stop_reason: str, expected: str
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message_json() | {"stop_reason": stop_reason})

    assert (await make_adapter(handler).generate(REQUEST)).finish_reason == expected


async def test_stream_reports_finish_reason_on_the_final_chunk() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return stream_response("Hi")

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert chunks[-1].finish_reason == "stop"
