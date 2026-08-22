"""Stage wrapper: retries, a request-level deadline budget, and a circuit breaker.

Real behaviour, not decoration. A Stage that exhausts its retries raises a typed
StageError; a Stage that would overrun the request deadline is not attempted at
all, because burning the remaining budget on a call we cannot use is worse than
failing early with a specific message.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Generic, TypeVar

from app.core.types import ErrorCode, StageError

log = logging.getLogger("overhear.harness")

T = TypeVar("T")


class Deadline:
    """A wall-clock budget shared across the stages of one request."""

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms
        self.started = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    @property
    def remaining_ms(self) -> float:
        return self.budget_ms - self.elapsed_ms

    def check(self, stage: str, need_ms: float = 0.0) -> None:
        if self.remaining_ms <= need_ms:
            raise StageError(
                ErrorCode.DEADLINE_EXCEEDED,
                f"{stage} skipped: {self.remaining_ms:.0f}ms left of a "
                f"{self.budget_ms:.0f}ms budget, needs {need_ms:.0f}ms",
                detail={"stage": stage, "remaining_ms": round(self.remaining_ms, 1)},
            )

    def reset(self) -> None:
        self.started = time.perf_counter()


@dataclass
class CircuitBreaker:
    """Opens after `threshold` consecutive failures, half-opens after `cooldown_s`."""

    name: str
    threshold: int = 3
    cooldown_s: float = 20.0
    failures: int = 0
    opened_at: float | None = field(default=None)

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.perf_counter() - self.opened_at >= self.cooldown_s:
            # half-open: allow one probe through
            self.opened_at = None
            self.failures = self.threshold - 1
            log.info("circuit %s half-open, probing", self.name)
            return False
        return True

    def record_success(self) -> None:
        if self.failures or self.opened_at:
            log.info("circuit %s recovered", self.name)
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold and self.opened_at is None:
            self.opened_at = time.perf_counter()
            log.warning("circuit %s OPEN after %d failures", self.name, self.failures)

    def guard(self) -> None:
        if self.is_open:
            raise StageError(
                ErrorCode.PROVIDER_ERROR,
                f"{self.name} circuit open",
                retryable=False,
                detail={"circuit": self.name, "failures": self.failures},
            )


class Stage(Generic[T]):
    """One retriable unit of work with a typed failure mode."""

    def __init__(
        self,
        name: str,
        fn: Callable[..., Awaitable[T]],
        *,
        retries: int = 2,
        base_backoff_ms: float = 40.0,
        timeout_ms: float | None = None,
        expected_ms: float = 0.0,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.retries = retries
        self.base_backoff_ms = base_backoff_ms
        self.timeout_ms = timeout_ms
        self.expected_ms = expected_ms
        self.breaker = breaker

    async def run(self, *args, deadline: Deadline | None = None, **kwargs) -> T:
        if deadline is not None:
            deadline.check(self.name, self.expected_ms)
        if self.breaker is not None:
            self.breaker.guard()

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                timeout_s = self._timeout_s(deadline)
                if timeout_s is not None:
                    out = await asyncio.wait_for(self.fn(*args, **kwargs), timeout=timeout_s)
                else:
                    out = await self.fn(*args, **kwargs)
                if self.breaker is not None:
                    self.breaker.record_success()
                return out
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as e:
                last = StageError(
                    ErrorCode.DEADLINE_EXCEEDED,
                    f"{self.name} timed out after {self._timeout_s(deadline)}s",
                    retryable=True,
                )
                log.warning("%s attempt %d timed out", self.name, attempt + 1)
            except StageError as e:
                last = e
                if not e.retryable:
                    if self.breaker is not None:
                        self.breaker.record_failure()
                    raise
                log.warning("%s attempt %d failed: %s", self.name, attempt + 1, e.message)
            except Exception as e:  # noqa: BLE001 - deliberately broad, re-typed below
                last = e
                log.warning("%s attempt %d raised %s: %s",
                            self.name, attempt + 1, type(e).__name__, e)

            if attempt < self.retries:
                backoff = self.base_backoff_ms * (2 ** attempt)
                backoff += random.uniform(0, self.base_backoff_ms)  # jitter
                if deadline is not None and deadline.remaining_ms <= backoff + self.expected_ms:
                    break
                await asyncio.sleep(backoff / 1000)

        if self.breaker is not None:
            self.breaker.record_failure()
        if isinstance(last, StageError):
            raise last
        raise StageError(
            ErrorCode.INTERNAL,
            f"{self.name} failed after {self.retries + 1} attempts: "
            f"{type(last).__name__}: {last}",
            detail={"stage": self.name},
        ) from last

    def _timeout_s(self, deadline: Deadline | None) -> float | None:
        candidates = []
        if self.timeout_ms is not None:
            candidates.append(self.timeout_ms)
        if deadline is not None:
            candidates.append(max(deadline.remaining_ms, 1.0))
        return min(candidates) / 1000 if candidates else None
