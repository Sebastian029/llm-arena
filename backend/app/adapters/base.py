"""The shared interface and data types that every provider adapter uses."""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    """What we ask a model to do: the model, the prompt and the parameters of a variant.

    `temperature=None` means "use the provider's default". Anthropic ignores temperature
    entirely (its current API rejects it), so a value only has an effect on OpenAI and Ollama.
    """

    model: str
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int = 1024


@dataclass(frozen=True)
class GenerationResult:
    """A complete answer, with the numbers needed for cost and latency comparison.

    `finish_reason` is "stop" (the model finished) or "length" (it was cut off by
    `max_tokens`). Other provider values are passed through; None means unknown.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
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
