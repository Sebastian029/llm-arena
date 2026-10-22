from decimal import Decimal

import pytest

from app.services.pricing import (
    PRICES,
    ModelPrice,
    UnknownModelPriceError,
    calculate_cost,
    get_price,
)


def test_cost_is_input_and_output_tokens_times_their_prices() -> None:
    # claude-opus-5-5: $4 per 1M input, $20 per 1M output
    cost = calculate_cost("anthropic", "claude-opus-5-5", 1_000_000, 500_000)

    assert cost == Decimal("14")  # 4 + 0.5 * 20


def test_cost_of_a_small_call_is_exact() -> None:
    # gpt-6-luna: $0.10 / $0.50 -> 1234 * 0.10 + 567 * 0.50 per million
    cost = calculate_cost("openai", "gpt-6-luna", 1234, 567)

    assert cost == Decimal("0.0004069")


def test_decimal_avoids_float_drift_when_summing_many_costs() -> None:
    one_call = calculate_cost("openai", "gpt-5-nano", 1000, 0)

    assert sum([one_call] * 1000) == Decimal("0.05")


def test_zero_tokens_cost_nothing() -> None:
    assert calculate_cost("anthropic", "claude-haiku-5-5", 0, 0) == Decimal(0)


@pytest.mark.parametrize("model", ["llama3.2", "qwen3:8b", "anything"])
def test_ollama_is_free_for_every_model(model: str) -> None:
    assert calculate_cost("ollama", model, 10_000, 10_000) == Decimal(0)


@pytest.mark.parametrize(("input_tokens", "output_tokens"), [(None, 5), (5, None), (None, None)])
def test_unknown_token_count_gives_unknown_cost_not_zero(
    input_tokens: int | None, output_tokens: int | None
) -> None:
    assert calculate_cost("openai", "gpt-6.1-sol", input_tokens, output_tokens) is None


def test_unknown_model_raises_instead_of_guessing_zero() -> None:
    with pytest.raises(UnknownModelPriceError, match="gpt-nope") as exc_info:
        calculate_cost("openai", "gpt-nope", 10, 10)

    assert exc_info.value.provider == "openai"
    assert exc_info.value.model == "gpt-nope"


def test_unknown_model_raises_even_when_tokens_are_unknown() -> None:
    with pytest.raises(UnknownModelPriceError):
        calculate_cost("openai", "gpt-nope", None, None)


def test_same_model_name_at_another_provider_is_not_priced() -> None:
    with pytest.raises(UnknownModelPriceError):
        get_price("openai", "claude-opus-5-5")


def test_negative_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        calculate_cost("openai", "gpt-6.1-sol", -1, 5)


def test_price_table_holds_exact_decimals_and_no_floats() -> None:
    assert PRICES
    for price in PRICES.values():
        assert isinstance(price, ModelPrice)
        assert isinstance(price.input_per_1m, Decimal)
        assert isinstance(price.output_per_1m, Decimal)
        assert price.input_per_1m > 0
        assert price.output_per_1m >= price.input_per_1m  # output is never cheaper than input
