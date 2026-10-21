"""Adapter for a local Ollama server (https://github.com/ollama/ollama), talking plain HTTP."""

import json
import time
from collections.abc import AsyncIterator

import httpx

from app.adapters.base import (
    GenerationRequest,
    GenerationResult,
    ModelAdapter,
    StreamChunk,
    elapsed_ms,
)
from app.adapters.errors import ProviderUnavailableError, error_from_status, parse_retry_after

PROVIDER = "ollama"


class OllamaAdapter(ModelAdapter):
    """Talks to Ollama's `POST /api/chat`. Ollama is free, so it is our development model."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        # One shared client keeps connections open between calls. Tests pass in a client
        # with a fake transport, so no real network is used.
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._build_payload(request, stream=False)
        start = time.perf_counter()
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TransportError as exc:  # timeouts, refused connection, ...
            raise ProviderUnavailableError(f"Cannot reach Ollama: {exc}", PROVIDER) from exc
        _raise_for_status(response)

        data = _parse_json(response.text)
        return GenerationResult(
            text=_message_content(data),
            model=data.get("model", request.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=elapsed_ms(start),
            finish_reason=data.get("done_reason"),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(request, stream=True)
        start = time.perf_counter()
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.is_error:
                    await response.aread()  # the body must be loaded before we can inspect it
                    _raise_for_status(response)

                # Ollama streams NDJSON: one JSON object per line.
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = _parse_json(line)
                    if "error" in data:  # e.g. the model crashed while answering
                        raise ProviderUnavailableError(str(data["error"]), PROVIDER)
                    if data.get("done"):
                        yield StreamChunk(
                            is_final=True,
                            input_tokens=data.get("prompt_eval_count", 0),
                            output_tokens=data.get("eval_count", 0),
                            latency_ms=elapsed_ms(start),
                            finish_reason=data.get("done_reason"),
                        )
                        return
                    yield StreamChunk(text=_message_content(data))
            # The stream closed without a "done" line: the answer is incomplete.
            raise ProviderUnavailableError("Ollama stream ended before it finished", PROVIDER)
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(f"Cannot reach Ollama: {exc}", PROVIDER) from exc

    @staticmethod
    def _build_payload(request: GenerationRequest, stream: bool) -> dict:
        """Translate our request into Ollama's field names (max_tokens is `num_predict`)."""
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        options: dict = {"num_predict": request.max_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        return {"model": request.model, "messages": messages, "stream": stream, "options": options}


def _parse_json(text: str) -> dict:
    """Parse one JSON object from Ollama; anything else is a broken answer."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailableError("Ollama returned invalid JSON", PROVIDER) from exc
    if not isinstance(data, dict):
        raise ProviderUnavailableError("Ollama returned an unexpected JSON shape", PROVIDER)
    return data


def _message_content(data: dict) -> str:
    """Read `message.content` and turn a missing or malformed field into our error."""
    try:
        return data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ProviderUnavailableError("Ollama answer has no message content", PROVIDER) from exc


def _raise_for_status(response: httpx.Response) -> None:
    """Turn an HTTP error answer into one of our errors. Does nothing for 2xx."""
    if response.is_success:
        return
    retry_after = parse_retry_after(response.headers.get("retry-after"))
    raise error_from_status(response.status_code, _error_message(response), PROVIDER, retry_after)


def _error_message(response: httpx.Response) -> str:
    """Use Ollama's `error` field when present; fall back to the status code."""
    try:
        return str(response.json()["error"])
    except (ValueError, KeyError, TypeError):
        return f"Ollama answered HTTP {response.status_code}"
