import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run `@pytest.mark.anyio` tests with asyncio only (trio is not installed)."""
    return "asyncio"
