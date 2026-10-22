import json

import httpx
import pytest

from app.adapters.base import GenerationRequest
from app.adapters.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.adapters.ollama_adapter import OllamaAdapter

pytestmark = pytest.mark.anyio

REQUEST = GenerationRequest(
    model="llama3.2", prompt="Hi", system_prompt="Be brief.", temperature=0.2, max_tokens=50
)


def make_adapter(handler) -> OllamaAdapter:
    """Build an adapter whose HTTP client answers with `handler` instead of using the network."""
    client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return OllamaAdapter(client=client)


def ndjson(*objects: dict) -> bytes:
    return "".join(json.dumps(obj) + "\n" for obj in objects).encode()


async def test_generate_returns_text_tokens_and_latency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
                "prompt_eval_count": 26,
                "eval_count": 298,
            },
        )

    result = await make_adapter(handler).generate(REQUEST)

    assert result.text == "Hello!"
    assert result.model == "llama3.2"
    assert result.input_tokens == 26
    assert result.output_tokens == 298
    assert result.latency_ms >= 0


async def test_generate_sends_ollama_shaped_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    await make_adapter(handler).generate(REQUEST)

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/chat"
    assert json.loads(seen[0].content) == {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 50},
    }


async def test_generate_without_system_prompt_sends_only_user_message() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    await make_adapter(handler).generate(GenerationRequest(model="m", prompt="Hi"))

    assert json.loads(seen[0].content)["messages"] == [{"role": "user", "content": "Hi"}]


async def test_stream_yields_text_chunks_then_final_chunk_with_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "Hel"}, "done": False},
                {"message": {"content": "lo"}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                },
            ),
        )

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert [c.text for c in chunks] == ["Hel", "lo", ""]
    assert [c.is_final for c in chunks] == [False, False, True]
    assert chunks[-1].input_tokens == 5
    assert chunks[-1].output_tokens == 2
    assert chunks[-1].latency_ms >= 0


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        (404, InvalidRequestError),
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
    ],
)
async def test_http_errors_become_adapter_errors(status: int, error_class: type) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "something went wrong"})

    with pytest.raises(error_class, match="something went wrong"):
        await make_adapter(handler).generate(REQUEST)


async def test_http_errors_are_also_mapped_when_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'x' not found"})

    with pytest.raises(InvalidRequestError, match="not found"):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_error_without_json_body_falls_back_to_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Bad gateway")

    with pytest.raises(ProviderUnavailableError, match="502"):
        await make_adapter(handler).generate(REQUEST)


async def test_rate_limit_error_reads_retry_after_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": "slow down"})

    with pytest.raises(RateLimitError) as exc_info:
        await make_adapter(handler).generate(REQUEST)

    assert exc_info.value.retry_after == 3.0


async def test_timeout_becomes_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(ProviderUnavailableError):
        await make_adapter(handler).generate(REQUEST)


async def test_connection_refused_becomes_provider_unavailable_when_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_error_object_inside_stream_is_raised_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"error": "model crashed"}))

    with pytest.raises(ProviderUnavailableError, match="model crashed"):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_stream_that_ends_without_done_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"message": {"content": "Hel"}, "done": False}))

    chunks = []
    with pytest.raises(ProviderUnavailableError, match="ended before"):
        async for chunk in make_adapter(handler).stream(REQUEST):
            chunks.append(chunk)

    assert [c.text for c in chunks] == ["Hel"]


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'{"done": true}'])
async def test_broken_answer_becomes_provider_unavailable(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(ProviderUnavailableError):
        await make_adapter(handler).generate(REQUEST)


async def test_generate_raises_error_field_from_a_200_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "model crashed"})

    with pytest.raises(ProviderUnavailableError, match="model crashed"):
        await make_adapter(handler).generate(REQUEST)


async def test_generate_rejects_an_answer_that_is_not_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "Hel"}, "done": False})

    with pytest.raises(ProviderUnavailableError, match="incomplete"):
        await make_adapter(handler).generate(REQUEST)


async def test_stream_without_usage_reports_unknown_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson({"message": {"content": ""}, "done": True}))

    chunks = [chunk async for chunk in make_adapter(handler).stream(REQUEST)]

    assert chunks[-1].input_tokens is None
    assert chunks[-1].output_tokens is None


async def test_other_httpx_errors_become_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.DecodingError("bad gzip body")  # not a TransportError

    with pytest.raises(ProviderUnavailableError):
        await make_adapter(handler).generate(REQUEST)

    with pytest.raises(ProviderUnavailableError):
        async for _ in make_adapter(handler).stream(REQUEST):
            pass


async def test_temperature_is_sent_only_when_set() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    await make_adapter(handler).generate(GenerationRequest(model="m", prompt="Hi", max_tokens=9))

    assert json.loads(seen[0].content)["options"] == {"num_predict": 9}


async def test_finish_reason_is_reported_for_generate_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["stream"]:
            return httpx.Response(
                200,
                content=ndjson({"message": {"content": ""}, "done": True, "done_reason": "length"}),
            )
        return httpx.Response(
            200, json={"message": {"content": "ok"}, "done": True, "done_reason": "stop"}
        )

    adapter = make_adapter(handler)
    chunks = [chunk async for chunk in adapter.stream(REQUEST)]

    assert (await adapter.generate(REQUEST)).finish_reason == "stop"
    assert chunks[-1].finish_reason == "length"
