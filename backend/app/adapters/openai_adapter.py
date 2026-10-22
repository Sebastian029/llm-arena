"""Adapter for OpenAI models, built on the official `openai` SDK (Chat Completions API)."""

import time
from collections.abc import AsyncIterator

import openai

from app.adapters.base import (
    GenerationRequest,
    GenerationResult,
    ModelAdapter,
    StreamChunk,
    elapsed_ms,
)
from app.adapters.errors import (
    AdapterError,
    ProviderUnavailableError,
    QuotaExceededError,
    error_from_status,
    parse_retry_after,
)

PROVIDER = "openai"


class OpenAIAdapter(ModelAdapter):
    """Talks to OpenAI's `chat.completions`.

    Temperature is sent only when the request sets one. Some reasoning models accept only
    the default; for those the API answers 400 and you get an `InvalidRequestError`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        timeout: float = 60.0,
    ) -> None:
        # max_retries=0: retries live in one place, our own code (see AnthropicAdapter).
        # With api_key=None the SDK reads OPENAI_API_KEY from the environment.
        self._client = client or openai.AsyncOpenAI(api_key=api_key, max_retries=0, timeout=timeout)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**self._build_kwargs(request))
        except openai.APIError as exc:
            raise _translate_error(exc) from exc

        usage = completion.usage
        choice = completion.choices[0] if completion.choices else None
        return GenerationResult(
            text=(choice.message.content if choice else None) or "",
            model=completion.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms(start),
            finish_reason=choice.finish_reason if choice else None,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        start = time.perf_counter()
        # Without include_usage the API never reports token counts when streaming.
        kwargs = self._build_kwargs(request) | {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # None = unknown: stays None if the stream ends without a usage chunk
        input_tokens = output_tokens = None
        finish_reason = None
        try:
            async with await self._client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    if chunk.usage:  # the last chunk: usage is set and `choices` is empty
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                    if chunk.choices:
                        choice = chunk.choices[0]
                        finish_reason = choice.finish_reason or finish_reason
                        if choice.delta.content:
                            yield StreamChunk(text=choice.delta.content)
        except openai.APIError as exc:
            raise _translate_error(exc) from exc

        yield StreamChunk(
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms(start),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _build_kwargs(request: GenerationRequest) -> dict:
        """Translate our request into SDK arguments. The system prompt is a normal message."""
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            # `max_tokens` is deprecated and rejected by reasoning models.
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        return kwargs


def _translate_error(exc: openai.APIError) -> AdapterError:
    """Map an SDK exception to one of our errors."""
    if not isinstance(exc, openai.APIStatusError) or exc.status_code < 400:
        # connection problems and timeouts, but also error events inside a stream (HTTP 200)
        # and unreadable responses: none of them is the caller's fault
        return ProviderUnavailableError(f"OpenAI call failed: {exc}", PROVIDER)

    if exc.status_code == 429 and exc.code == "insufficient_quota":
        # out of credit: waiting will not help, so this must not be treated as a rate limit
        return QuotaExceededError(exc.message, PROVIDER)
    retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
    return error_from_status(exc.status_code, exc.message, PROVIDER, retry_after)
