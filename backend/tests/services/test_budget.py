from decimal import Decimal

import pytest

from app.services.budget import BudgetExceededError, BudgetTracker


def test_new_tracker_has_spent_nothing() -> None:
    tracker = BudgetTracker(Decimal("1.00"))

    assert tracker.spent == Decimal(0)
    assert tracker.remaining == Decimal("1.00")
    tracker.check()  # does not raise


def test_add_accumulates_costs_exactly() -> None:
    tracker = BudgetTracker(Decimal("1.00"))

    for _ in range(3):
        tracker.add(Decimal("0.1"))

    assert tracker.spent == Decimal("0.3")
    assert tracker.remaining == Decimal("0.7")


def test_check_passes_below_the_limit() -> None:
    tracker = BudgetTracker(Decimal("1.00"))
    tracker.add(Decimal("0.99"))

    tracker.check()


def test_check_raises_when_the_limit_is_reached() -> None:
    tracker = BudgetTracker(Decimal("1.00"))
    tracker.add(Decimal("1.00"))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()

    assert exc_info.value.spent == Decimal("1.00")
    assert exc_info.value.limit == Decimal("1.00")


def test_last_call_may_cross_the_limit_and_remaining_stays_at_zero() -> None:
    tracker = BudgetTracker(Decimal("1.00"))
    tracker.add(Decimal("0.90"))
    tracker.check()  # still allowed to start another call
    tracker.add(Decimal("0.50"))  # that call turned out to cost more than what was left

    assert tracker.spent == Decimal("1.40")
    assert tracker.remaining == Decimal(0)
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_unknown_cost_is_counted_separately_not_as_zero_spend() -> None:
    tracker = BudgetTracker(Decimal("1.00"))

    tracker.add(None)
    tracker.add(Decimal("0.25"))
    tracker.add(None)

    assert tracker.spent == Decimal("0.25")
    assert tracker.unknown_cost_calls == 2


def test_zero_budget_blocks_the_first_call() -> None:
    with pytest.raises(BudgetExceededError):
        BudgetTracker(Decimal(0)).check()


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        BudgetTracker(Decimal("-1"))


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cost"):
        BudgetTracker(Decimal("1")).add(Decimal("-0.01"))
