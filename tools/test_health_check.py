#!/usr/bin/env python3
"""
Unit tests for the retry/backoff and circuit-breaker logic added to
tools/health_check.py.

These tests use no network: the single HTTP probe is patched with scripted
results, and the sleep/clock functions are injected, so the whole suite runs
in milliseconds and deterministically.

Run:
    python3 -m unittest tools.test_health_check -v
    # or, from the tools/ directory:
    python3 -m unittest test_health_check -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging  # noqa: E402

import health_check as hc  # noqa: E402

# Keep test output clean: route the module's WARNING logs to a null handler
# rather than the default stderr stream handler.
hc.logger.addHandler(logging.NullHandler())
hc.logger.propagate = False


# A small recorder so tests can assert on backoff delays without sleeping.
class _SleepRecorder:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


class _FakeClock:
    """Manually advanceable monotonic clock for circuit-breaker timing."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _probe_returning(*results):
    """Build a fake _single_http_probe that yields the given results in order."""
    seq = iter(results)

    def fake_probe(host, port, path, timeout):
        return next(seq)

    return fake_probe


OK = ("OK", "HTTP 200", 200)
CRIT = ("CRITICAL", "Connection refused", 0)
WARN = ("WARNING", "HTTP 404", 404)
SERVER_ERR = ("CRITICAL", "HTTP 503", 503)


# =============================================================================
# Retry + exponential backoff
# =============================================================================

class TestRetryBackoff(unittest.TestCase):
    def test_retry_succeeds_after_transient_failures(self):
        sleeper = _SleepRecorder()
        with mock.patch.object(hc, "_single_http_probe",
                               _probe_returning(CRIT, CRIT, OK)):
            status, _, code = hc.check_http_service(
                "h", 1, "/", 1, max_retries=3,
                base_delay=0.5, backoff_factor=2.0, sleep_func=sleeper,
            )
        self.assertEqual(status, "OK")
        self.assertEqual(code, 200)
        # Two failures => two backoff sleeps before the success.
        self.assertEqual(sleeper.delays, [0.5, 1.0])

    def test_backoff_delay_formula(self):
        sleeper = _SleepRecorder()
        with mock.patch.object(hc, "_single_http_probe",
                               _probe_returning(CRIT, CRIT, CRIT, CRIT)):
            hc.check_http_service(
                "h", 1, "/", 1, max_retries=3,
                base_delay=0.5, backoff_factor=2.0, sleep_func=sleeper,
            )
        # delay = base_delay * factor**attempt for attempts 0,1,2 (no sleep after last)
        self.assertEqual(sleeper.delays, [0.5, 1.0, 2.0])

    def test_exhausting_retries_returns_critical(self):
        sleeper = _SleepRecorder()
        with mock.patch.object(hc, "_single_http_probe",
                               _probe_returning(CRIT, CRIT, CRIT)):
            status, _, _ = hc.check_http_service(
                "h", 1, "/", 1, max_retries=2, sleep_func=sleeper,
            )
        self.assertEqual(status, "CRITICAL")
        self.assertEqual(len(sleeper.delays), 2)  # 3 attempts => 2 sleeps

    def test_no_retry_on_reachable_warning(self):
        """A 4xx WARNING means the service answered — do not retry."""
        sleeper = _SleepRecorder()
        probe = mock.Mock(return_value=WARN)
        with mock.patch.object(hc, "_single_http_probe", probe):
            status, _, code = hc.check_http_service(
                "h", 1, "/", 1, max_retries=5, sleep_func=sleeper,
            )
        self.assertEqual(status, "WARNING")
        self.assertEqual(code, 404)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(sleeper.delays, [])

    def test_server_error_is_retried(self):
        sleeper = _SleepRecorder()
        with mock.patch.object(hc, "_single_http_probe",
                               _probe_returning(SERVER_ERR, OK)):
            status, _, _ = hc.check_http_service(
                "h", 1, "/", 1, max_retries=2, sleep_func=sleeper,
            )
        self.assertEqual(status, "OK")
        self.assertEqual(len(sleeper.delays), 1)


# =============================================================================
# Circuit breaker
# =============================================================================

class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_threshold_consecutive_failures(self):
        cb = hc.CircuitBreaker(threshold=3, cooldown=30.0, time_func=_FakeClock())
        key = "h:1"
        cb.record_failure(key)
        cb.record_failure(key)
        self.assertEqual(cb.state(key), cb.CLOSED)
        self.assertTrue(cb.allows_request(key))
        cb.record_failure(key)  # third → opens
        self.assertEqual(cb.state(key), cb.OPEN)
        self.assertFalse(cb.allows_request(key))

    def test_half_open_after_cooldown(self):
        clock = _FakeClock()
        cb = hc.CircuitBreaker(threshold=2, cooldown=30.0, time_func=clock)
        key = "h:1"
        cb.record_failure(key)
        cb.record_failure(key)
        self.assertEqual(cb.state(key), cb.OPEN)
        clock.advance(31)
        self.assertEqual(cb.state(key), cb.HALF_OPEN)
        self.assertTrue(cb.allows_request(key))  # trial allowed

    def test_success_resets_circuit(self):
        cb = hc.CircuitBreaker(threshold=2, cooldown=30.0, time_func=_FakeClock())
        key = "h:1"
        cb.record_failure(key)
        cb.record_failure(key)
        self.assertEqual(cb.state(key), cb.OPEN)
        cb.record_success(key)
        self.assertEqual(cb.state(key), cb.CLOSED)
        self.assertEqual(cb.failure_count(key), 0)

    def test_open_circuit_skips_probe(self):
        """When the circuit is OPEN, check_http_service must not probe."""
        clock = _FakeClock()
        cb = hc.CircuitBreaker(threshold=1, cooldown=30.0, time_func=clock)
        cb.record_failure("h:1")  # opens immediately (threshold=1)
        probe = mock.Mock(return_value=OK)
        with mock.patch.object(hc, "_single_http_probe", probe):
            status, detail, _ = hc.check_http_service(
                "h", 1, "/", 1, max_retries=2, circuit_breaker=cb,
                sleep_func=_SleepRecorder(),
            )
        self.assertEqual(status, "CRITICAL")
        self.assertIn("Circuit open", detail)
        probe.assert_not_called()

    def test_exhausted_probe_opens_circuit(self):
        clock = _FakeClock()
        cb = hc.CircuitBreaker(threshold=1, cooldown=30.0, time_func=clock)
        with mock.patch.object(hc, "_single_http_probe",
                               _probe_returning(CRIT)):
            hc.check_http_service(
                "h", 1, "/", 1, max_retries=0, circuit_breaker=cb,
                sleep_func=_SleepRecorder(),
            )
        self.assertEqual(cb.state("h:1"), cb.OPEN)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            hc.CircuitBreaker(threshold=0)


# =============================================================================
# Aggregation / summary
# =============================================================================

class TestSummarize(unittest.TestCase):
    def test_summarize_counts_and_degraded(self):
        results = {
            "services": {
                "backend": {"status": "OK", "detail": ""},
                "market": {"status": "CRITICAL", "detail": "down"},
            },
            "infrastructure": {
                "redis": {"status": "WARNING", "detail": "slow"},
            },
            "system": {
                "disk": {"status": "OK", "detail": ""},
            },
        }
        summary = hc.summarize_results(results)
        self.assertEqual(summary["total_checks"], 4)
        self.assertEqual(summary["ok"], 2)
        self.assertEqual(summary["warning"], 1)
        self.assertEqual(summary["critical"], 1)
        self.assertEqual(summary["healthy_pct"], 50.0)
        self.assertIn("services/market: CRITICAL", summary["degraded"])
        self.assertIn("infrastructure/redis: WARNING", summary["degraded"])

    def test_summarize_includes_nested_certificate_check(self):
        results = {
            "services": {
                "frontend": {
                    "status": "OK",
                    "detail": "",
                    "certificate": {"status": "WARNING", "detail": "expires soon"},
                },
            },
            "infrastructure": {},
            "system": {},
        }
        summary = hc.summarize_results(results)
        self.assertEqual(summary["total_checks"], 2)  # service + nested cert
        self.assertIn("services/frontend.certificate: WARNING", summary["degraded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
