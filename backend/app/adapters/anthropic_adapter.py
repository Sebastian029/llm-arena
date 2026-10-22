"""Adapter for Anthropic's Claude models, built on the official `anthropic` SDK."""

import time
from collections.abc import AsyncIterator

import anthropic

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
    error_from_status,
    parse_retry_after,
    unsupported_options_error,
)

PROVIDER = "anthropic"

# Anthropic's stop reasons mapped to ours; others (e.g. "tool_use") are passed through.
FINISH_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


class AnthropicAdapter(ModelAdapter):
    """Talks to the Claude Messages API.

    `request.temperature` is NOT sent: the current API (SDK 2.x) removed sampling
    parameters and answers 400 when they are present. Claude samples with its defaults.
    `request.effort` is the way to make Claude variants differ (sent as `output_config`).
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        timeout: float = 60.0,
    ) -> None:
        # max_retries=0: the SDK would silently retry 429/5xx. We want retries in ONE place
        # (our own code, stages 3 and 6) so latency and cost are measured honestly.
        # With api_key=None the SDK reads ANTHROPIC_API_KEY from the environment.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=timeout
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        try:
            message = await self._client.messages.create(**self._build_kwargs(request))
        except anthropic.APIError as exc:
            raise _translate_error(exc) from exc

        return GenerationResult(
            text=_join_text(message),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=elapsed_ms(start),
            finish_reason=_finish_reason(message),
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        start = time.perf_counter()
        try:
            async with self._client.messages.stream(**self._build_kwargs(request)) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(text=text)
                message = await stream.get_final_message()  # has the token usage
        except anthropic.APIError as exc:
            raise _translate_error(exc) from exc

        yield StreamChunk(
            is_final=True,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=elapsed_ms(start),
            finish_reason=_finish_reason(message),
        )

    @staticmethod
    def _build_kwargs(request: GenerationRequest) -> dict:
        """Translate our request into SDK arguments. `system` is a separate argument."""
        if request.options:
            raise unsupported_options_error(PROVIDER)
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if request.effort:
            kwargs["output_config"] = {"effort": request.effort}
        return kwargs


def _join_text(message: anthropic.types.Message) -> str:
    """The answer is a list of content blocks; we keep only the text ones."""
    return "".join(block.text for block in message.content if block.type == "text")


def _finish_reason(message: anthropic.types.Message) -> str | None:
    """Translate Anthropic's `stop_reason` to our `finish_reason`."""
    reason = message.stop_reason
    return FINISH_REASONS.get(reason, reason) if reason else None


def _translate_error(exc: anthropic.APIError) -> AdapterError:
    """Map an SDK exception to one of our errors."""
    if not isinstance(exc, anthropic.APIStatusError) or exc.status_code < 400:
        # connection problems and timeouts, but also error events inside a stream (HTTP 200)
        # and unreadable responses: none of them is the caller's fault
        return ProviderUnavailableError(f"Anthropic call failed: {exc}", PROVIDER)

    retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
    return error_from_status(exc.status_code, exc.message, PROVIDER, retry_after)
