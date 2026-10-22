"""The shared interface and data types that every provider adapter uses."""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    """What we ask a model to do: the model, the prompt and the parameters of a variant.

    `None` means "use the provider's default".
    - `temperature`: Anthropic ignores it (its current API rejects it), so it only has an
      effect on OpenAI and Ollama.
    - `effort`: how hard the model should think ("low", "medium", "high", ...). Sent as
      `output_config.effort` (Anthropic), `reasoning_effort` (OpenAI) or `think` (Ollama).
      Valid values depend on the model; the provider answers 400 for a wrong one. Thinking
      tokens are billed as output tokens, so a higher effort costs more.
    - `options`: extra model parameters (top_p, top_k, seed, num_ctx, ...) passed as they are
      to Ollama. Other providers raise `InvalidRequestError` instead of silently ignoring them.
      The fields above win over the same key in `options`.
    """

    model: str
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int = 1024
    effort: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """A complete answer, with the numbers needed for cost and latency comparison.

    `finish_reason` is "stop" (the model finished) or "length" (it was cut off by
    `max_tokens`). Other provider values are passed through; None means unknown.
    Token counts are None when the provider did not report them (never 0 as a guess), so
    a missing count cannot be mistaken for a free call.
    """

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    finish_reason: str | None = None


@dataclass(frozen=True)
class StreamChunk:
    """One piece of a streamed answer.

    Normal chunks carry only `text`. The last chunk has `is_final=True`, empty text
    and the token counts, latency and finish reason of the whole answer.
    """

    text: str = ""
    is_final: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    finish_reason: str | None = None


def elapsed_ms(start: float) -> float:
    """Milliseconds since `start`, a value of `time.perf_counter()`.

    Every adapter measures latency this way, so providers are compared fairly.
    """
    return (time.perf_counter() - start) * 1000


class ModelAdapter(ABC):
    """Interface implemented by every provider (Anthropic, OpenAI, Ollama).

    Adapters raise only the errors from `app.adapters.errors`.
    """

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Send the request and wait for the complete answer."""

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Send the request and yield the answer piece by piece.

        This is a plain `def` on purpose: implementations are async generators
        (`async def` with `yield`), which you iterate with `async for` without `await`.
        """
