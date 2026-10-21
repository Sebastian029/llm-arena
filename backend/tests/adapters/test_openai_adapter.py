import json

import httpx2  # the HTTP library used by openai 3.x; we only need its fake transport
import openai
import pytest

from app.adapters.base import GenerationRequest
from app.adapters.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.adapters.openai_adapter import OpenAIAdapter

pytestmark = pytest.mark.anyio

MODEL = "gpt-test"
REQUEST = GenerationRequest(
    model=MODEL, prompt="Hi", system_prompt="Be brief.", temperature=0.2, max_tokens=50
)


def make_adapter(handler) -> OpenAIAdapter:
    """Build an adapter whose SDK client answers with `handler` instead of using the network."""
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = openai.AsyncOpenAI(api_key="test-key", http_client=http_client, max_retries=0)
    return OpenAIAdapter(client=client)


def completion_json(text: str = "Hello!", prompt_tokens: int = 10, completion_tokens: int = 5):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def error_json(message: str) -> dict:
    return {"error": {"message": message, "type": "error", "param": None, "code": None}}


def chunk_json(delta: dict, usage: dict | None = None, choices: bool = True) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}] if choices else [],
        "usage": usage,
    }


def stream_response(
    *texts: str, with_usage: bool = True, finish_reason: str = "stop"
) -> httpx2.Response:
    """Server-Sent Events: `data: <json>` blocks ended by `data: [DONE]`."""
    chunks = [chunk_json({"role": "assistant", "content": ""})]
    chunks += [chunk_json({"content": text}) for text in texts]
    last = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    chunks.append(chunk_json({}) | {"choices": [last]})
    if with_usage:
        usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        chunks.append(chunk_json({}, usage=usage, choices=False))
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def test_generate_returns_text_tokens_and_latency() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=completion_json("Hello!", 10, 5))

    result = await make_adapter(handler).generate(REQUEST)

    assert result.text == "Hello!"
    assert result.model == MODEL
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0


async def test_generate_sends_openai_shaped_request() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_json())

    await make_adapter(handler).generate(REQUEST)

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert body["model"] == MODEL
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert body["temperature"] == 0.2
    assert body["max_completion_tokens"] == 50
    assert "max_tokens" not in body  # deprecated, rejected by reasoning models


async def test_generate_without_system_prompt_sends_only_user_message() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_json())

    await make_adapter(handler).generate(GenerationRequest(model=MODEL, prompt="Hi"))

    assert json.loads(seen[0].content)["messages"] == [{"role": "user", "content": "Hi"}]


async def test_stream_yields_text_chunks_then_final_chunk_with_usage() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        return stream_response("Hel", "lo")

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert [c.text for c in chunks] == ["Hel", "lo", ""]
    assert [c.is_final for c in chunks] == [False, False, True]
    assert chunks[-1].input_tokens == 12
    assert chunks[-1].output_tokens == 7
    assert chunks[-1].latency_ms >= 0


async def test_stream_without_usage_chunk_reports_zero_tokens() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return stream_response("Hi", with_usage=False)  # e.g. the stream was cut short

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert chunks[-1].is_final is True
    assert chunks[-1].input_tokens == 0
    assert chunks[-1].output_tokens == 0


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        (400, InvalidRequestError),
        (404, InvalidRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
    ],
)
async def test_http_errors_become_adapter_errors(status: int, error_class: type) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=error_json("something went wrong"))

    with pytest.raises(error_class, match="something went wrong"):
        await make_adapter(handler).generate(REQUEST)


async def test_http_errors_are_also_mapped_when_streaming() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json=error_json("model nope not found"))

    with pytest.raises(InvalidRequestError, match="nope"):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_rate_limit_error_reads_retry_after_header() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, headers={"retry-after": "7"}, json=error_json("slow down"))

    with pytest.raises(RateLimitError) as exc_info:
        await make_adapter(handler).generate(REQUEST)

    assert exc_info.value.retry_after == 7.0


async def test_rate_limit_without_header_has_no_retry_after() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, json=error_json("slow down"))

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


async def test_429_for_insufficient_quota_is_not_a_rate_limit() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = {
            "error": {
                "message": "You exceeded your quota",
                "type": "insufficient_quota",
                "param": None,
                "code": "insufficient_quota",
            }
        }
        return httpx2.Response(429, json=body)

    with pytest.raises(InvalidRequestError, match="quota"):
        await make_adapter(handler).generate(REQUEST)


async def test_empty_choices_gives_empty_text() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=completion_json() | {"choices": []})

    assert (await make_adapter(handler).generate(REQUEST)).text == ""


async def test_error_event_inside_stream_becomes_provider_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = 'data: {"error": {"message": "server exploded", "type": "server_error"}}\n\n'
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    with pytest.raises(ProviderUnavailableError, match="server exploded"):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_temperature_is_sent_only_when_set() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_json())

    await make_adapter(handler).generate(GenerationRequest(model=MODEL, prompt="Hi"))

    assert "temperature" not in json.loads(seen[0].content)


async def test_generate_reports_finish_reason() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = completion_json()
        body["choices"][0]["finish_reason"] = "length"
        return httpx2.Response(200, json=body)

    assert (await make_adapter(handler).generate(REQUEST)).finish_reason == "length"


async def test_stream_reports_finish_reason_on_the_final_chunk() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return stream_response("Hi", finish_reason="length")

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert chunks[-1].finish_reason == "length"
