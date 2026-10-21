"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api import health

app = FastAPI(title="llm-arena")
app.include_router(health.router)
