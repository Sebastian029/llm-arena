# Stages

Source of truth for the work plan. Tick a stage only after tests + ruff pass and the commit is made.
Each stage: plan -> approval -> small steps -> tests + ruff -> summary -> commit -> STOP, wait for "dalej".

Core idea: the thing we compare is a **variant** = model + system prompt + parameters (temperature, max_tokens).

- [x] **0. Foundations** - this file and docs/decisions.md, .env.example, pyproject.toml (ruff, pytest), docker-compose (api, postgres, redis), FastAPI /health + first test, GitHub Actions (ruff + pytest).
- [x] **1. Provider adapters** - one shared interface (text, tokens, latency, streaming), Ollama / Anthropic / OpenAI implementations, unified errors. Tests with mocked HTTP.
- [x] **2. Costs and budget** - price table, `calculate_cost`, budget limit check, all in one place in services/.
- [ ] **3. Arena backend** - Variant, Prompt, Run, Response, Vote models + Alembic migrations, parallel calls to 2-4 variants (asyncio), SSE streaming, voting, retry on 429.
- [ ] **4. Ranking** - Elo (pure function, tested) + win rate + vote count per variant, ranking endpoint.
- [ ] **5. Frontend: Arena** - Vite + React, variant picker, side-by-side streaming via a single SSE hook, voting, ranking page.
- [ ] **6. Test sets and runs (Celery)** - TestSet / TestCase / EvalRun / EvalResult, idempotent tasks with saved progress (resume), retry on 429, stop when budget is exceeded.
- [ ] **7. Automatic scoring** - matching (exact/contains/regex), rules (valid JSON, length), LLM-as-judge (mocked in tests).
- [ ] **8. Results dashboard** - quality / cost / latency per variant, A/B prompt comparison, charts.
- [ ] **9. Portfolio polish** - README with architecture diagram, GIF, seed data, CI badge, interview notes via /interview-prep.

## Out of scope
User accounts / login, public deployment (demo = repo + README + GIF), Bradley-Terry ranking (optional after stage 9).
