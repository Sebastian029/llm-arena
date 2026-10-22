"""Budget limit for a run: stop calling models once the money is spent.

This keeps the state in memory only. Stages 3 and 6 will save the spent amount in the
database, so a resumed test run starts from what it already spent.
"""

from decimal import Decimal


class BudgetExceededError(Exception):
    """The spending limit was reached, so no further model call may be made."""

    def __init__(self, spent: Decimal, limit: Decimal) -> None:
        super().__init__(f"Budget exceeded: spent ${spent} of ${limit}")
        self.spent = spent
        self.limit = limit


class BudgetTracker:
    """Adds up costs and tells us when the limit is reached.

    Usage in a loop: `check()` before each model call, `add(cost)` after it.
    The call that crosses the limit is not interrupted (we only know its price afterwards),
    so the total can end slightly above the limit, by at most one call.
    """

    def __init__(self, limit_usd: Decimal) -> None:
        if limit_usd < 0:
            raise ValueError("Budget limit cannot be negative")
        self.limit = limit_usd
        self.spent = Decimal(0)
        # Calls whose cost was unknown (token counts missing). Not counted in `spent`,
        # so a caller or the UI can warn that the real total may be higher.
        self.unknown_cost_calls = 0

    @property
    def remaining(self) -> Decimal:
        """Money left, never below zero."""
        return max(self.limit - self.spent, Decimal(0))

    def add(self, cost: Decimal | None) -> None:
        """Record the cost of one call. None = cost unknown (see `unknown_cost_calls`)."""
        if cost is None:
            self.unknown_cost_calls += 1
            return
        if cost < 0:
            raise ValueError("Cost cannot be negative")
        self.spent += cost

    def check(self) -> None:
        """Raise `BudgetExceededError` when the limit is already reached."""
        if self.spent >= self.limit:
            raise BudgetExceededError(self.spent, self.limit)
