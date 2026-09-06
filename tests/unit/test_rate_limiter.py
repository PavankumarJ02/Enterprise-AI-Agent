"""Unit tests for SlidingWindowRateLimiter."""

import time

from enterprise_agent.middleware.rate_limit import SlidingWindowRateLimiter


def test_sliding_window_allows_under_limit() -> None:
    """Verify requests under the per-minute limit are permitted with decremented remaining count."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=5, burst_limit=5, window_seconds=60.0)

    for i in range(1, 6):
        allowed, remaining, reset_after, retry_after = limiter.check("client-1")
        assert allowed is True
        assert remaining == 5 - i
        assert reset_after > 0
        assert retry_after == 0


def test_sliding_window_blocks_over_limit() -> None:
    """Verify exceeding per-minute limit returns False with retry_after > 0."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=3, burst_limit=10, window_seconds=60.0)

    # 3 allowed
    for _ in range(3):
        allowed, _, _, _ = limiter.check("client-1")
        assert allowed is True

    # 4th request blocked
    allowed, remaining, reset_after, retry_after = limiter.check("client-1")
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0
    assert reset_after == retry_after

    # Different client is not blocked
    allowed2, remaining2, _, _ = limiter.check("client-2")
    assert allowed2 is True
    assert remaining2 == 2


def test_sliding_window_burst_protection() -> None:
    """Verify exceeding 1-second burst limit blocks immediately even if total capacity remains."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=100, burst_limit=2, window_seconds=60.0)

    # 2 requests in quick succession
    assert limiter.check("client-burst")[0] is True
    assert limiter.check("client-burst")[0] is True

    # 3rd request in same second violates burst limit
    allowed, remaining, _, retry_after = limiter.check("client-burst")
    assert allowed is False
    assert remaining == 0
    assert retry_after == 1


def test_rate_limiter_window_expiry() -> None:
    """Verify requests outside window are purged and quota resets."""
    # Fast 0.2 second window
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, burst_limit=10, window_seconds=0.2)

    assert limiter.check("client-exp")[0] is True
    assert limiter.check("client-exp")[0] is True
    assert limiter.check("client-exp")[0] is False

    # Sleep past window
    time.sleep(0.25)

    # Should be allowed again
    allowed, remaining, _, _ = limiter.check("client-exp")
    assert allowed is True
    assert remaining == 1


def test_rate_limiter_clear() -> None:
    """Verify clearing the limiter resets all client records."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=1, burst_limit=5)
    limiter.check("client-x")
    assert limiter.check("client-x")[0] is False

    limiter.clear()
    assert limiter.check("client-x")[0] is True
