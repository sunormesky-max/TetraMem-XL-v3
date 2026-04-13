"""
Circuit breaker and rate limiter for TetraMem-XL.

Protects the system from resource exhaustion during:
  - Dream cycles under high load
  - Emergence loops during stress periods
  - Self-organization cascading failures
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("tetramem.circuit_breaker")


@dataclass
class CircuitState:
    name: str
    is_open: bool = False
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = 0.0
    half_call_count: int = 0


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_call_limit: int = 3,
    ):
        self._state = CircuitState(name=name, last_state_change=time.time())
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_call_limit = half_call_limit
        self._lock = threading.RLock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state.is_open:
                if time.time() - self._state.last_failure_time > self._recovery_timeout:
                    self._state.is_open = False
                    self._state.half_call_count = 0
                    self._state.last_state_change = time.time()
                    logger.info("Circuit [%s] entering half-open", self._state.name)
            return self._state.is_open

    def record_success(self) -> None:
        with self._lock:
            self._state.success_count += 1
            self._state.failure_count = max(0, self._state.failure_count - 1)
            if not self._state.is_open:
                pass
            self._state.half_call_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()
            if self._state.failure_count >= self._failure_threshold:
                if not self._state.is_open:
                    self._state.is_open = True
                    self._state.last_state_change = time.time()
                    logger.warning(
                        "Circuit [%s] OPENED after %d failures",
                        self._state.name, self._state.failure_count,
                    )

    def allow_call(self) -> bool:
        with self._lock:
            if not self._state.is_open:
                return True
            if time.time() - self._state.last_failure_time > self._recovery_timeout:
                self._state.half_call_count += 1
                if self._state.half_call_count <= self._half_call_limit:
                    return True
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self._state.name,
            "is_open": self._state.is_open,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "last_failure_time": self._state.last_failure_time,
        }


class RateLimiter:
    def __init__(
        self,
        name: str,
        max_calls: int = 10,
        window_seconds: float = 60.0,
    ):
        self._name = name
        self._max_calls = max_calls
        self._window = window_seconds
        self._call_times: List[float] = []
        self._lock = threading.RLock()
        self._total_allowed = 0
        self._total_rejected = 0

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            cutoff = now - self._window
            self._call_times = [t for t in self._call_times if t > cutoff]

            if len(self._call_times) >= self._max_calls:
                self._total_rejected += 1
                return False

            self._call_times.append(now)
            self._total_allowed += 1
            return True

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            cutoff = now - self._window
            active = len([t for t in self._call_times if t > cutoff])
        return {
            "name": self._name,
            "max_calls": self._max_calls,
            "window_seconds": self._window,
            "active_calls": active,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
        }


class EmergenceProtector:
    def __init__(
        self,
        dream_failure_threshold: int = 3,
        emergence_failure_threshold: int = 5,
        max_dreams_per_minute: int = 6,
        max_emergence_per_minute: int = 4,
        recovery_timeout: float = 60.0,
    ):
        self.dream_breaker = CircuitBreaker(
            "dream", failure_threshold=dream_failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.emergence_breaker = CircuitBreaker(
            "emergence", failure_threshold=emergence_failure_threshold,
            recovery_timeout=recovery_timeout * 2,
        )
        self.dream_limiter = RateLimiter("dream", max_calls=max_dreams_per_minute, window_seconds=60.0)
        self.emergence_limiter = RateLimiter("emergence", max_calls=max_emergence_per_minute, window_seconds=60.0)

    def allow_dream(self) -> bool:
        if not self.dream_limiter.allow():
            logger.info("Dream rate limited")
            return False
        if not self.dream_breaker.allow_call():
            logger.info("Dream circuit breaker open")
            return False
        return True

    def allow_emergence(self) -> bool:
        if not self.emergence_limiter.allow():
            logger.info("Emergence rate limited")
            return False
        if not self.emergence_breaker.allow_call():
            logger.info("Emergence circuit breaker open")
            return False
        return True

    def record_dream_success(self) -> None:
        self.dream_breaker.record_success()

    def record_dream_failure(self) -> None:
        self.dream_breaker.record_failure()

    def record_emergence_success(self) -> None:
        self.emergence_breaker.record_success()

    def record_emergence_failure(self) -> None:
        self.emergence_breaker.record_failure()

    def get_status(self) -> Dict[str, Any]:
        return {
            "dream_breaker": self.dream_breaker.get_status(),
            "emergence_breaker": self.emergence_breaker.get_status(),
            "dream_limiter": self.dream_limiter.get_status(),
            "emergence_limiter": self.emergence_limiter.get_status(),
        }
