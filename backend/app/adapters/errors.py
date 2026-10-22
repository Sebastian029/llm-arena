"""Errors shared by all provider adapters.

Each adapter catches the exceptions of its own SDK and raises one of these instead,
so the rest of the app never has to import `anthropic` or `openai` to handle a failure.
"""


class AdapterError(Exception):
    """Base class for every error raised by an adapter."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message)
        self.provider = provider


class RateLimitError(AdapterError):
    """The provider answered 429: too many requests. Safe to retry later."""

    def __init__(self, message: str, provider: str, retry_after: float | None = None) -> None:
        super().__init__(message, provider)
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said so


class AuthenticationError(AdapterError):
    """The API key is missing, wrong or not allowed to use this model. Retrying will not help."""


class InvalidRequestError(AdapterError):
    """The provider rejected the request itself (unknown model, bad parameter). Do not retry."""


class QuotaExceededError(AdapterError):
    """The account is out of credit or over its spending limit. Waiting will not help."""


class ProviderUnavailableError(AdapterError):
    """Timeout, connection problem or a 5xx answer. May work if retried."""


def error_from_status(
    status: int, message: str, provider: str, retry_after: float | None = None
) -> AdapterError:
    """Choose our error for an HTTP error status. Shared by all adapters.

    429 -> RateLimitError, 401/403 -> AuthenticationError, 408/409 and 5xx ->
    ProviderUnavailableError (408 is a timeout, 409 a conflict that usually clears;
    5xx includes Anthropic's 529 "overloaded"), anything else (400, 404 for an unknown
    model, 413, 422) -> InvalidRequestError.
    """
    if status == 429:
        return RateLimitError(message, provider, retry_after=retry_after)
    if status in (401, 403):
        return AuthenticationError(message, provider)
    if status in (408, 409) or status >= 500:
        return ProviderUnavailableError(message, provider)
    return InvalidRequestError(message, provider)


def parse_retry_after(value: str | None) -> float | None:
    """Read a `Retry-After` header given in seconds. Missing or other formats give None.

    The header may also hold a date; we ignore that form and let the caller use its own delay.
    """
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
