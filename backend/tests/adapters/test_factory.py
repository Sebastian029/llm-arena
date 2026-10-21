import pytest

from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.errors import AuthenticationError, InvalidRequestError
from app.adapters.factory import get_adapter
from app.adapters.ollama_adapter import OllamaAdapter
from app.adapters.openai_adapter import OpenAIAdapter


def test_returns_anthropic_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert isinstance(get_adapter("anthropic"), AnthropicAdapter)


def test_returns_openai_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert isinstance(get_adapter("openai"), OpenAIAdapter)


def test_returns_ollama_adapter_without_any_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert isinstance(get_adapter("ollama"), OllamaAdapter)


def test_ollama_uses_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.test:1234")

    adapter = get_adapter("ollama")

    assert str(adapter._client.base_url).startswith("http://example.test:1234")


@pytest.mark.parametrize(
    ("provider", "variable"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_missing_api_key_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, provider: str, variable: str
) -> None:
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(AuthenticationError, match=variable):
        get_adapter(provider)


def test_empty_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(AuthenticationError):
        get_adapter("openai")


def test_unknown_provider_lists_the_valid_ones() -> None:
    with pytest.raises(InvalidRequestError, match="anthropic, openai, ollama"):
        get_adapter("gemini")
