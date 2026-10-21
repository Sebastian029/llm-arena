# Decisions

Short log: decision, reason. Add new entries at the bottom.

1. **FastAPI** - native async (parallel model calls, SSE streaming), Pydantic validation from type hints, automatic /docs.
2. **Compared object is a variant** (model + system prompt + parameters), not just a model - the main question is often "which prompt is better".
3. **Ranking: Elo + win rate + vote count.** Elo is simple to explain; win rate and vote count show how (un)certain the ranking is. Bradley-Terry is a possible extension. Test-set runs are scored by scorers, not Elo.
4. **Cost** = input_tokens * input_price + output_tokens * output_price (per 1M tokens). Token counts come from the provider response. Ollama costs 0. Prices live in one table; check current prices in provider docs before filling it.
5. **Development models:** Ollama (free, local) and Anthropic (paid, small budget). The OpenAI adapter is tested with mocks only. CI never calls real APIs.
6. **Extra libraries (assumed approved, confirm if not):** SQLAlchemy 2 + Alembic (database, migrations), official Anthropic and OpenAI SDKs, httpx (Ollama).
7. **No timeline, no online demo, no user accounts.** Quality and understanding over speed; deliverable is repo + README + GIF.
8. **pyproject.toml lives in backend/** - all commands run from there (`cd backend && pytest`); dependencies in `[project]`, dev tools (pytest, httpx, ruff) in the `dev` extra, installed with `pip install -e ".[dev]"`.
9. **No pydantic-settings for now** - stage 0 needs no config; settings will come from `os.environ` until a real need appears (then ask).
10. **Ruff rules E, W, F, I, UP, B, line length 100** - style, unused code, import order, modern syntax, likely bugs. Lint and format are checked separately in CI.
11. **Base images:** python:3.12-slim, postgres:16, redis:7. Postgres and Redis are in compose from stage 0 but not used by the API until stages 3 and 6.
12. **Starlette warns that `httpx` in TestClient is deprecated (suggests `httpx2`).** Kept httpx (already approved); revisit if the warning turns into an error.
