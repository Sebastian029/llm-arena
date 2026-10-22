# Decisions

Short log: decision, reason. Add new entries at the bottom.

1. **FastAPI** - native async (parallel model calls, SSE streaming), Pydantic validation from type hints, automatic /docs.
2. **Compared object is a variant** (model + system prompt + parameters), not just a model - the main question is often "which prompt is better".
3. **Ranking: Elo + win rate + vote count.** Elo is simple to explain; win rate and vote count show how (un)certain the ranking is. Bradley-Terry is a possible extension. Test-set runs are scored by scorers, not Elo.
4. **Cost** = input_tokens * input_price + output_tokens * output_price (per 1M tokens). Token counts come from the provider response. Ollama costs 0. Prices live in one table; check current prices in provider docs before filling it.
5. **Development models:** Ollama (free, local) and Anthropic (paid, small budget). The OpenAI adapter is tested with mocks only. CI never calls real APIs.
6. **Extra libraries (assumed approved, confirm if not):** SQLAlchemy 2 + Alembic (database, migrations), official Anthropic and OpenAI SDKs, httpx (Ollama).
7. **No timeline, no online demo, no user accounts.** Quality and understanding over speed; deliverable is repo + README + GIF.
8. **pyproject.toml lives in backend/** - all commands run from there (`cd backend && pytest`); dependencies in `[project]`, dev tools (pytest, ruff) in the `dev` extra (httpx moved to `[project]` after code review: OllamaAdapter needs it at runtime), installed with `pip install -e ".[dev]"`.
9. **No pydantic-settings for now** - stage 0 needs no config; settings will come from `os.environ` until a real need appears (then ask).
10. **Ruff rules E, W, F, I, UP, B, line length 100** - style, unused code, import order, modern syntax, likely bugs. Lint and format are checked separately in CI.
11. **Base images:** python:3.12-slim, postgres:16, redis:7. Postgres and Redis are in compose from stage 0 but not used by the API until stages 3 and 6.
12. **Starlette warns that `httpx` in TestClient is deprecated (suggests `httpx2`).** Kept httpx (already approved); revisit if the warning turns into an error.
13. **Adapter contract:** frozen dataclasses (`GenerationRequest`, `GenerationResult`, `StreamChunk`) and an abstract `ModelAdapter` with async `generate` and `stream`. The last stream chunk (`is_final=True`) carries tokens, latency and finish reason, because providers report usage only at the end. Latency is measured the same way everywhere (`elapsed_ms`), so providers are compared fairly.
14. **One error hierarchy** (`AdapterError` and 4 subclasses) so callers never import SDK exceptions. HTTP status -> error lives in one function, `error_from_status`. Error events inside a stream, broken answers and connection problems are `ProviderUnavailableError` (retryable). OpenAI 429 `insufficient_quota` is `QuotaExceededError` (out of credit, waiting does not help; the budget logic and UI can tell it apart from a bad request). HTTP 408 and 409 are retryable (`ProviderUnavailableError`).
15. **SDK retries are switched off (`max_retries=0`).** Retry on 429 lives in our code (stages 3 and 6), so latency and cost are measured honestly and the budget/resume logic stays in control.
16. **`temperature` is optional (`None` = provider default) and Anthropic never receives it.** API version 2026-10-21 / SDK 2.x removed `temperature`, `top_p`, `top_k` (sending them gives 400). Consequence: two Claude variants that differ only in temperature behave the same; the UI should say so (stage 5).
17. **`finish_reason` ("stop" / "length")** is part of the result, so an answer cut off by `max_tokens` is not scored as a wrong answer in stages 7-8.
18. **OpenAI: `max_completion_tokens`**, not `max_tokens` (deprecated, rejected by reasoning models). Streaming asks for usage with `stream_options.include_usage`; if the stream is cut before the usage chunk, the final chunk reports `None` (unknown), not 0, so costs are never silently undercounted.
19. **Testing the adapters:** Anthropic/OpenAI are tested through their real SDKs with `httpx2.MockTransport` (httpx2 is the HTTP library of both SDKs); Ollama with `httpx.MockTransport`. `httpx2` and `anyio` are declared as dev dependencies; async tests use the anyio pytest plugin, so `pytest-asyncio` is not needed.
20. **Factory reads settings from environment variables only** and creates a new adapter per call. HTTP clients are not closed yet; stage 3 should keep one adapter per provider (and add `aclose()` if needed).
21. **Ollama error format is not documented**, so errors are mapped by HTTP status and the `error` field is used only when present. To check by hand against a live Ollama.
22. **Code review of stage 1 (agent, 10 findings).** Fixed: httpx as a runtime dependency, 408/409 retryable, `QuotaExceededError`, unknown tokens as `None` when streaming, all `httpx.HTTPError`s translated in the Ollama adapter, Ollama `generate()` checks `error` and `done` like `stream()`. Deferred on purpose:
    - Anthropic ignores `temperature` -> UI warning in stage 5 (see 16).
    - Clients never closed -> one adapter per provider + `aclose()` in stage 3 (see 20).
    - Latency is total time to the end of the stream, includes Ollama cold-start model load and the consumer's pauses between chunks -> consider time-to-first-token in stages 2/8.
    - `OllamaAdapter` ignores `base_url` when a `client` is passed, and `OLLAMA_BASE_URL` is not validated -> minor, only affects misconfiguration.
    - Non-streamed `generate()` still returns 0 tokens when the provider omits usage (`GenerationResult` tokens are `int`); decide in stage 2 whether they should become optional.
