"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a static OK status so Docker and CI can check that the API is up."""
    return {"status": "ok"}
