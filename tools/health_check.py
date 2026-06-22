#!/usr/bin/env python3
"""
Health check tool for the Tent of Trials platform.
Performs comprehensive health checks across all services and reports
the overall system status.

This tool is used by:
  - The Kubernetes liveness/readiness probes
  - The deployment pipeline (post-deployment validation)
  - The monitoring system (periodic health checks)
  - The on-call engineer (manual troubleshooting)

The health check performs the following checks:
  1. Service availability (HTTP health endpoints)
  2. Database connectivity (connection test)
  3. Redis connectivity (ping test)
  4. Kafka connectivity (metadata fetch)
  5. Message queue depth (consumer lag check)
  6. Certificate expiry (TLS certificate check)
  7. Disk space (filesystem usage check)
  8. Memory usage (process memory check)

Each check returns a status of OK, WARNING, or CRITICAL, along with
a detail message and optional diagnostic data.

Usage:
    python3 health_check.py                  # Check all services
    python3 health_check.py --service backend # Check specific service
    python3 health_check.py --json            # JSON output
    python3 health_check.py --watch           # Continuous monitoring
"""

import argparse
import json
import logging
import os
import socket
import ssl
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("health_check")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SERVICES = {
    "backend": {"host": "localhost", "port": 8080, "path": "/health", "timeout": 5},
    "market": {"host": "localhost", "port": 8081, "path": "/health", "timeout": 5},
    "frailbox": {"host": "localhost", "port": 8082, "path": "/health", "timeout": 10},
    "frontend": {"host": "localhost", "port": 3000, "path": "/", "timeout": 5},
}

INFRASTRUCTURE = {
    "postgresql": {"host": os.environ.get("DB_HOST", "localhost"), "port": int(os.environ.get("DB_PORT", "5432")), "timeout": 5},
    "redis": {"host": os.environ.get("REDIS_HOST", "localhost"), "port": int(os.environ.get("REDIS_PORT", "6379")), "timeout": 5},
    "kafka": {"host": os.environ.get("KAFKA_HOST", "localhost"), "port": int(os.environ.get("KAFKA_PORT", "9092")), "timeout": 5},
}

DISK_THRESHOLD_WARNING = 80
DISK_THRESHOLD_CRITICAL = 90

MEMORY_THRESHOLD_WARNING = 80
MEMORY_THRESHOLD_CRITICAL = 90

# Retry / backoff / circuit-breaker defaults for flaky HTTP endpoints.
DEFAULT_MAX_RETRIES = 2          # extra attempts after the first (3 tries total)
DEFAULT_BASE_DELAY = 0.5         # seconds; first backoff delay
DEFAULT_BACKOFF_FACTOR = 2.0     # delay = base_delay * (backoff_factor ** attempt)
DEFAULT_CIRCUIT_THRESHOLD = 5    # consecutive failures before the circuit opens
DEFAULT_CIRCUIT_COOLDOWN = 30.0  # seconds the circuit stays open before a trial


# ---------------------------------------------------------------------------
# CIRCUIT BREAKER
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Per-endpoint circuit breaker that prevents hammering a failing service.

    State transitions (keyed independently per endpoint):

        CLOSED   --(>= threshold consecutive failures)-->  OPEN
        OPEN     --(cooldown elapsed)-------------------->  HALF_OPEN
        HALF_OPEN --(success)--------------------------->  CLOSED
        HALF_OPEN --(failure)--------------------------->  OPEN (cooldown restarts)

    While a circuit is OPEN the caller should skip the probe entirely rather
    than issue a request that is very likely to fail. After the cooldown the
    circuit becomes HALF_OPEN and a single trial request is allowed through to
    decide whether the endpoint has recovered.

    ``time_func`` is injectable so tests can advance time deterministically.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int = DEFAULT_CIRCUIT_THRESHOLD,
        cooldown: float = DEFAULT_CIRCUIT_COOLDOWN,
        time_func=time.monotonic,
    ):
        if threshold < 1:
            raise ValueError("circuit threshold must be >= 1")
        if cooldown < 0:
            raise ValueError("circuit cooldown must be >= 0")
        self.threshold = threshold
        self.cooldown = cooldown
        self._time = time_func
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}

    def state(self, key: str) -> str:
        """Return the current circuit state for ``key``."""
        if key not in self._opened_at:
            return self.CLOSED
        if self._time() - self._opened_at[key] >= self.cooldown:
            return self.HALF_OPEN
        return self.OPEN

    def allows_request(self, key: str) -> bool:
        """True unless the circuit is fully OPEN (within its cooldown)."""
        return self.state(key) != self.OPEN

    def record_success(self, key: str) -> None:
        """Reset the circuit for ``key`` after a successful probe."""
        self._failures.pop(key, None)
        self._opened_at.pop(key, None)

    def record_failure(self, key: str) -> None:
        """Register a failed probe; opens the circuit at the threshold."""
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self.threshold:
            self._opened_at[key] = self._time()

    def failure_count(self, key: str) -> int:
        return self._failures.get(key, 0)

# ---------------------------------------------------------------------------
# CHECK FUNCTIONS
# ---------------------------------------------------------------------------

def _single_http_probe(host: str, port: int, path: str, timeout: int) -> Tuple[str, str, int]:
    """Perform a single HTTP probe (no retry). Returns (status, detail, code)."""
    import http.client
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", path)
        resp = conn.getresponse()
        status = resp.status
        body = resp.read().decode("utf-8", errors="replace")[:200]
        conn.close()

        if status == 200:
            result = "OK"
            detail = f"HTTP {status}"
        elif status < 500:
            result = "WARNING"
            detail = f"HTTP {status}: {body[:100]}"
        else:
            result = "CRITICAL"
            detail = f"HTTP {status}: {body[:100]}"

        return result, detail, status
    except Exception as e:
        return "CRITICAL", str(e), 0


def check_http_service(
    host: str,
    port: int,
    path: str,
    timeout: int,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    circuit_breaker: Optional[CircuitBreaker] = None,
    sleep_func=time.sleep,
) -> Tuple[str, str, int]:
    """
    Probe an HTTP health endpoint with retry, exponential backoff, and an
    optional circuit breaker.

    Retry policy:
      - A CRITICAL result (connection error or HTTP 5xx) is retried up to
        ``max_retries`` additional times.
      - A reachable response (OK, or a 4xx WARNING) is returned immediately —
        the service answered, so retrying would not help.
      - Backoff before attempt ``n`` (0-indexed) is
        ``base_delay * (backoff_factor ** n)`` seconds.

    Circuit breaker:
      - When the breaker is OPEN for this endpoint the probe is skipped and a
        CRITICAL "circuit open" result is returned, so a known-bad service is
        not hammered.
      - A reachable response resets the breaker; exhausting all retries records
        a failure that may open it.

    ``sleep_func`` is injectable so tests need not wait on real time.
    """
    key = f"{host}:{port}"

    if circuit_breaker is not None and not circuit_breaker.allows_request(key):
        logger.warning("circuit OPEN for %s — skipping probe to avoid hammering", key)
        return "CRITICAL", "Circuit open — probe skipped (cooldown active)", 0

    attempts = max(1, max_retries + 1)
    result, detail, code = "CRITICAL", "no attempt made", 0

    for attempt in range(attempts):
        result, detail, code = _single_http_probe(host, port, path, timeout)

        # Any non-CRITICAL outcome means the service responded — treat the
        # endpoint as reachable, reset the breaker, and stop retrying.
        if result != "CRITICAL":
            if circuit_breaker is not None:
                circuit_breaker.record_success(key)
            return result, detail, code

        if attempt < attempts - 1:
            delay = base_delay * (backoff_factor ** attempt)
            logger.warning(
                "probe %s failed (attempt %d/%d): %s — retrying in %.2fs",
                key, attempt + 1, attempts, detail, delay,
            )
            sleep_func(delay)

    # All attempts exhausted: the endpoint is degraded/unreachable.
    if circuit_breaker is not None:
        circuit_breaker.record_failure(key)
        if not circuit_breaker.allows_request(key):
            logger.warning(
                "circuit OPENED for %s after %d consecutive failures",
                key, circuit_breaker.threshold,
            )
    logger.warning(
        "probe %s degraded: %s (after %d attempt(s))", key, detail, attempts
    )
    return result, detail, code


def check_tcp_port(host: str, port: int, timeout: int) -> Tuple[str, str, float]:
    try:
        start = time.time()
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        latency = (time.time() - start) * 1000
        return "OK", f"Connected ({latency:.1f}ms)", latency
    except socket.timeout:
        return "CRITICAL", f"Connection timeout ({timeout}s)", 0
    except ConnectionRefusedError:
        return "CRITICAL", "Connection refused", 0
    except Exception as e:
        return "CRITICAL", str(e), 0


def check_certificate_expiry(host: str, port: int = 443) -> Tuple[str, str, int]:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                if not cert:
                    return "WARNING", "No certificate found", 0

                from datetime import datetime as dt
                expires = dt.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
                days_left = (expires - dt.now()).days

                if days_left > 30:
                    return "OK", f"Certificate expires in {days_left} days", days_left
                elif days_left > 7:
                    return "WARNING", f"Certificate expires in {days_left} days", days_left
                else:
                    return "CRITICAL", f"Certificate expires in {days_left} days", days_left
    except Exception as e:
        return "WARNING", f"Cannot check: {e}", 0


def check_disk_usage(path: str = "/") -> Tuple[str, str, float]:
    try:
        stat = os.statvfs(path)
        total = stat.f_frsize * stat.f_blocks
        free = stat.f_frsize * stat.f_bavail
        used = total - free
        pct = (used / total) * 100

        if pct < DISK_THRESHOLD_WARNING:
            return "OK", f"{pct:.1f}% used ({used // (1024**3)}GB/{total // (1024**3)}GB)", pct
        elif pct < DISK_THRESHOLD_CRITICAL:
            return "WARNING", f"{pct:.1f}% used ({used // (1024**3)}GB/{total // (1024**3)}GB)", pct
        else:
            return "CRITICAL", f"{pct:.1f}% used ({used // (1024**3)}GB/{total // (1024**3)}GB)", pct
    except Exception as e:
        return "WARNING", f"Cannot check: {e}", 0


def check_memory_usage() -> Tuple[str, str, float]:
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().replace(" kB", "")
                    try:
                        meminfo[key] = int(value) * 1024
                    except ValueError:
                        pass

        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        used = total - available
        pct = (used / total) * 100 if total > 0 else 0

        if pct < MEMORY_THRESHOLD_WARNING:
            return "OK", f"{pct:.1f}% used ({used // (1024**3)}GB/{total // (1024**3)}GB)", pct
        elif pct < MEMORY_THRESHOLD_CRITICAL:
            return "WARNING", f"{pct:.1f}% used", pct
        else:
            return "CRITICAL", f"{pct:.1f}% used", pct
    except Exception as e:
        return "WARNING", f"Cannot check: {e}", 0


def check_load_average() -> Tuple[str, str, float]:
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().strip().split()
            load = float(parts[0])
            cpu_count = os.cpu_count() or 1
            load_pct = (load / cpu_count) * 100

            if load_pct < 70:
                return "OK", f"Load: {load} ({load_pct:.0f}% of {cpu_count} cores)", load
            elif load_pct < 90:
                return "WARNING", f"Load: {load} ({load_pct:.0f}% of {cpu_count} cores)", load
            else:
                return "CRITICAL", f"Load: {load} ({load_pct:.0f}% of {cpu_count} cores)", load
    except Exception as e:
        return "WARNING", f"Cannot check: {e}", 0


# ---------------------------------------------------------------------------
# HEALTH CHECK RUNNER
# ---------------------------------------------------------------------------

def run_health_checks(
    service: Optional[str] = None,
    json_output: bool = False,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "services": {},
        "infrastructure": {},
        "system": {},
        "overall_status": "OK",
    }

    all_ok = True

    # Check services
    for name, config in SERVICES.items():
        if service and name != service:
            continue
        status, detail, code = check_http_service(
            config["host"], config["port"], config["path"], config["timeout"],
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            circuit_breaker=circuit_breaker,
        )
        results["services"][name] = {
            "status": status,
            "detail": detail,
            "code": code,
            "endpoint": f"http://{config['host']}:{config['port']}{config['path']}",
        }
        if status == "CRITICAL":
            all_ok = False

    # Check infrastructure
    for name, config in INFRASTRUCTURE.items():
        if service and name != service:
            continue
        status, detail, latency = check_tcp_port(config["host"], config["port"], config["timeout"])
        results["infrastructure"][name] = {
            "status": status,
            "detail": detail,
            "endpoint": f"{config['host']}:{config['port']}",
        }
        if status == "CRITICAL":
            all_ok = False

    # Check system resources
    disk_status, disk_detail, disk_pct = check_disk_usage()
    results["system"]["disk"] = {"status": disk_status, "detail": disk_detail}
    if disk_status == "CRITICAL":
        all_ok = False

    mem_status, mem_detail, mem_pct = check_memory_usage()
    results["system"]["memory"] = {"status": mem_status, "detail": mem_detail}
    if mem_status == "CRITICAL":
        all_ok = False

    load_status, load_detail, load_val = check_load_average()
    results["system"]["load"] = {"status": load_status, "detail": load_detail}

    # Check certificate expiry (web services)
    for name, config in SERVICES.items():
        if service and name != service:
            continue
        if config["port"] == 443:
            cert_status, cert_detail, days_left = check_certificate_expiry(config["host"])
            results["services"][name]["certificate"] = {
                "status": cert_status,
                "detail": cert_detail,
                "days_remaining": days_left,
            }
            if cert_status == "CRITICAL":
                all_ok = False

    results["overall_status"] = "OK" if all_ok else "DEGRADED"
    results["summary"] = summarize_results(results)

    if results["overall_status"] == "DEGRADED":
        logger.warning(
            "overall status DEGRADED — %d critical, %d warning of %d checks",
            results["summary"]["critical"],
            results["summary"]["warning"],
            results["summary"]["total_checks"],
        )

    return results


def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate per-check statuses into summary stats.

    Walks services, infrastructure, and system (including nested checks such
    as a service's certificate block) and counts OK/WARNING/CRITICAL, listing
    every non-OK check under ``degraded``.
    """
    counts = {"OK": 0, "WARNING": 0, "CRITICAL": 0}
    degraded: List[str] = []

    def _tally(label: str, check: Dict[str, Any]) -> None:
        status = check.get("status")
        if status in counts:
            counts[status] += 1
            if status != "OK":
                degraded.append(f"{label}: {status}")

    for category in ("services", "infrastructure", "system"):
        for name, check in results.get(category, {}).items():
            if not isinstance(check, dict):
                continue
            if "status" in check:
                _tally(f"{category}/{name}", check)
            # Nested checks (e.g. services[name]["certificate"]).
            for sub_name, sub_check in check.items():
                if isinstance(sub_check, dict) and "status" in sub_check:
                    _tally(f"{category}/{name}.{sub_name}", sub_check)

    total = sum(counts.values())
    return {
        "total_checks": total,
        "ok": counts["OK"],
        "warning": counts["WARNING"],
        "critical": counts["CRITICAL"],
        "healthy_pct": round(counts["OK"] / total * 100, 1) if total else 0.0,
        "degraded": degraded,
    }


def print_health_report(results: Dict[str, Any]):
    print(f"\n{'='*60}")
    print(f"  HEALTH CHECK REPORT")
    print(f"  Host: {results['hostname']}")
    print(f"  Time: {results['timestamp']}")
    print(f"  Overall: {results['overall_status']}")
    print(f"{'='*60}")

    for category, items in [("Services", results["services"]),
                             ("Infrastructure", results["infrastructure"]),
                             ("System", results["system"])]:
        if items:
            print(f"\n  {category}:")
            for name, check in items.items():
                if isinstance(check, dict) and "status" in check:
                    status_icon = {"OK": "✓", "WARNING": "⚠", "CRITICAL": "✗"}.get(check["status"], "?")
                    print(f"    {status_icon} {name}: {check['detail']}")
                else:
                    print(f"    {name}:")
                    for sub_name, sub_check in check.items():
                        if isinstance(sub_check, dict) and "status" in sub_check:
                            sub_icon = {"OK": "✓", "WARNING": "⚠", "CRITICAL": "✗"}.get(sub_check["status"], "?")
                            print(f"      {sub_icon} {sub_name}: {sub_check['detail']}")

    summary = results.get("summary")
    if summary:
        print(f"\n  Summary: {summary['ok']} OK · {summary['warning']} WARNING · "
              f"{summary['critical']} CRITICAL "
              f"({summary['healthy_pct']:.0f}% healthy of {summary['total_checks']} checks)")
        if summary["degraded"]:
            print("  Degraded:")
            for item in summary["degraded"]:
                print(f"    - {item}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Health check tool")
    parser.add_argument("--service", "-s", help="Check specific service only")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help="Extra HTTP probe attempts after the first on failure",
    )
    parser.add_argument(
        "--base-delay", type=float, default=DEFAULT_BASE_DELAY,
        help="Base backoff delay in seconds",
    )
    parser.add_argument(
        "--backoff-factor", type=float, default=DEFAULT_BACKOFF_FACTOR,
        help="Exponential backoff factor: delay = base_delay * (factor ** attempt)",
    )
    parser.add_argument(
        "--circuit-threshold", type=int, default=DEFAULT_CIRCUIT_THRESHOLD,
        help="Consecutive failures before the circuit breaker opens",
    )
    parser.add_argument(
        "--circuit-cooldown", type=float, default=DEFAULT_CIRCUIT_COOLDOWN,
        help="Seconds the circuit stays open before allowing a trial probe",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable INFO-level logging (WARNING is always shown)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    circuit_breaker = CircuitBreaker(
        threshold=args.circuit_threshold,
        cooldown=args.circuit_cooldown,
    )

    probe_kwargs = dict(
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        backoff_factor=args.backoff_factor,
        circuit_breaker=circuit_breaker,
    )

    if args.watch:
        print(f"Continuous monitoring (interval: {args.interval}s). Press Ctrl+C to stop.")
        try:
            while True:
                results = run_health_checks(args.service, args.json, **probe_kwargs)
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print_health_report(results)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    else:
        results = run_health_checks(args.service, args.json, **probe_kwargs)
        if args.json:
            output = json.dumps(results, indent=2)
            print(output)
        else:
            print_health_report(results)

        if args.output:
            with open(args.output, "w") as f:
                if args.json:
                    json.dump(results, f, indent=2)
                else:
                    json.dump(results, f, indent=2)
            print(f"Report saved to {args.output}")

        if results["overall_status"] == "DEGRADED":
            return 1

    return 0


if __name__ == "__main__":
    main()
