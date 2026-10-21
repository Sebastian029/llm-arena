# llm-arena
A platform for evaluating LLMs (a mini Chatbot Arena + LangSmith). It answers the question: "Which model (and which prompt) is better for my task, by how much, and at what cost?"
The author is a junior developer. Goal: a portfolio project and a real understanding of every decision.

## Features
Arena (one prompt to 2-4 models in parallel, streaming, voting), Elo ranking, background test sets (evals), automatic scoring (matching, rules, LLM-as-judge), results dashboard (quality/cost/latency), cost control, retry on 429 and resumable test runs.

## Stack
Python 3.12, FastAPI, PostgreSQL, Redis, Celery, pytest, ruff, React (Vite), Docker + docker-compose, GitHub Actions. Provider adapters: OpenAI, Anthropic, Ollama. No extra libraries without a justification and my approval.

## Structure
- backend/app/api: endpoints (thin, no logic)
- backend/app/models: database models
- backend/app/schemas: request and response schemas
- backend/app/services: business logic (Elo, scoring, costs)
- backend/app/adapters: integrations with model providers, one shared interface
- backend/app/tasks: Celery tasks
- backend/app/core: configuration, database, security
- backend/tests: pytest tests
- frontend/src: pages, components, api
- docs: stages, decisions, configuration map

## Commands
- Tests: cd backend && pytest
- Lint: cd backend && ruff check . && ruff format --check .
- Everything: docker compose up
(update as the commands come into existence)

## Code rules
- All code, comments, docstrings, documentation and commit messages are written in English.
- Keep it simple and readable: type hints, docstrings, no unnecessary patterns or abstractions.
- Do not change working code without a reason. Report problems outside the current stage, but do not fix them.
- Check syntax and APIs of libraries (FastAPI, Celery, provider SDKs, Docker, GitHub Actions) in the current documentation, do not guess.

## Security and tests
- API keys only in environment variables. Do not read .env, use .env.example.
- Tests and CI use mocks only, no real API calls.
- Every function has tests. Run tests and lint before a commit.

## Way of working
- Stages 0-9 from docs/stages.md. After each one: summary, tests, commit, STOP and wait for my "dalej".
- Before a bigger change, explain what you are doing and why. When there are several options, compare them and recommend the best one for a junior.
- Briefly explain new concepts (SSE, Celery, fixture, Elo, LLM-as-judge) the first time they appear.
- If you can reasonably assume something, assume it and say so instead of asking.
- One commit per stage, with a meaningful message.
- Talk to me in Polish, in simple language. Code and documentation stay in English.
