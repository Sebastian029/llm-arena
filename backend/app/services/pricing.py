"""Price table and cost calculation. The only place in the app that knows what a token costs.

Money is `Decimal`, not `float`: floats cannot store 0.1 exactly, so sums of many small
costs drift (0.1 + 0.2 != 0.3). `Decimal` does exact decimal arithmetic.
"""

from dataclasses import dataclass
from decimal import Decimal

TOKENS_PER_PRICE_UNIT = 1_000_000  # providers quote prices per 1M tokens


@dataclass(frozen=True)
class ModelPrice:
    """Price of one model in USD per 1 million tokens."""

    input_per_1m: Decimal
    output_per_1m: Decimal


def _price(input_per_1m: str, output_per_1m: str) -> ModelPrice:
    # Built from strings on purpose: Decimal(0.1) would copy the float's rounding error.
    return ModelPrice(Decimal(input_per_1m), Decimal(output_per_1m))


# Keys are (provider, model) with the model name we SEND in the request (not the dated name
# a provider may answer with). Prices as of October 2026:
#   Anthropic: Claude API model table (cached 2026-10-06)
#   OpenAI:    https://developers.openai.com/api/docs/pricing (read 2026-10-22; standard
#              prices, context up to 272K)
# Check them again before trusting the numbers for a real run: they change.
# Ollama runs on your own machine, so it is free and has no entries here (see `get_price`).
PRICES: dict[tuple[str, str], ModelPrice] = {
    ("anthropic", "claude-opus-5-5"): _price("4.00", "20.00"),
    ("anthropic", "claude-sonnet-5-5"): _price("2.00", "10.00"),
    # Haiku 5.5: up to 100K prompt tokens; above that Anthropic charges 0.50 / 2.50.
    ("anthropic", "claude-haiku-5-5"): _price("0.10", "0.50"),
    ("openai", "gpt-6.1-sol"): _price("2.00", "10.00"),
    ("openai", "gpt-6-luna"): _price("0.10", "0.50"),
    ("openai", "gpt-5-nano"): _price("0.05", "0.40"),
}

FREE = ModelPrice(Decimal(0), Decimal(0))


class UnknownModelPriceError(Exception):
    """No price is known for this model. We refuse to guess: a silent 0 would hide real cost."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            f"No price for model '{model}' of provider '{provider}'. Add it to PRICES in "
            "app/services/pricing.py (check the provider's pricing page)."
        )
        self.provider = provider
        self.model = model


def get_price(provider: str, model: str) -> ModelPrice:
    """Return the price of a model. Ollama is always free; an unknown model raises."""
    if provider == "ollama":
        return FREE
    try:
        return PRICES[(provider, model)]
    except KeyError:
        raise UnknownModelPriceError(provider, model) from None


def calculate_cost(
    provider: str, model: str, input_tokens: int | None, output_tokens: int | None
) -> Decimal | None:
    """Cost in USD: input_tokens * input price + output_tokens * output price (per 1M tokens).

    Returns None when a token count is unknown (None): the cost is then unknown too, which is
    not the same as free. Thinking tokens are already inside `output_tokens` (providers bill
    them as output). Raises `UnknownModelPriceError` for a model without a price.
    """
    price = get_price(provider, model)  # checked first: a missing price is always a mistake
    if input_tokens is None or output_tokens is None:
        return None
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    return (
        input_tokens * price.input_per_1m + output_tokens * price.output_per_1m
    ) / TOKENS_PER_PRICE_UNIT
