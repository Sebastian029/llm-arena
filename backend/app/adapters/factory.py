"""Create the right adapter for a provider name, so callers never import concrete adapters."""

import os

from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.base import ModelAdapter
from app.adapters.errors import AuthenticationError, InvalidRequestError
from app.adapters.ollama_adapter import OllamaAdapter
from app.adapters.openai_adapter import OpenAIAdapter

PROVIDERS = ("anthropic", "openai", "ollama")
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def get_adapter(provider: str) -> ModelAdapter:
    """Return a new adapter for `provider` ("anthropic", "openai" or "ollama").

    Settings come from environment variables only: ANTHROPIC_API_KEY, OPENAI_API_KEY and
    OLLAMA_BASE_URL (optional). Each call builds a new adapter; callers that make many
    requests should keep the one they get.
    """
    if provider == "anthropic":
        return AnthropicAdapter(api_key=_require_env("ANTHROPIC_API_KEY", provider))
    if provider == "openai":
        return OpenAIAdapter(api_key=_require_env("OPENAI_API_KEY", provider))
    if provider == "ollama":
        return OllamaAdapter(base_url=os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL)
    raise InvalidRequestError(
        f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}", provider
    )


def _require_env(name: str, provider: str) -> str:
    """Read a required secret, failing early with a clear message instead of an SDK error."""
    value = os.environ.get(name)
    if not value:
        raise AuthenticationError(f"Environment variable {name} is not set", provider)
    return value
