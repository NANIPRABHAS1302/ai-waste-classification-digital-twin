"""
Latency measurement and benchmarking module for the waste sorting pipeline.

Provides structured, empirical timing benchmarks for:
1. Classifier inference latency (warm-up followed by repeated runs)
2. UDP loopback transmission latency
3. Simulator physics step latency

IMPORTANT METHODOLOGICAL PRINCIPLE:
- Never report cold-start latency as representative operational performance.
- Perform explicit warm-up iterations before measuring.
- Collect multiple runs and report mean, median, min, max, and std deviation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.telemetry.metrics import _compute_stats


def benchmark_function(
    func: Callable[[], Any],
    warmup_runs: int = 5,
    benchmark_runs: int = 30,
) -> Dict[str, Any]:
    """
    Benchmarks any zero-argument callable with warm-up and repeated timings.

    Parameters
    ----------
    func : callable
        Zero-argument function to benchmark.
    warmup_runs : int
        Number of unmeasured warm-up iterations.
    benchmark_runs : int
        Number of measured benchmark iterations.

    Returns
    -------
    dict
        Statistical summary of latencies in milliseconds.
    """
    if warmup_runs < 0 or benchmark_runs <= 0:
        raise ValueError("warmup_runs must be >= 0 and benchmark_runs must be > 0")

    # Warm-up phase (unmeasured)
    for _ in range(warmup_runs):
        func()

    # Measured phase
    timings_ms: List[float] = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    stats = _compute_stats(timings_ms)
    stats["warmup_runs"] = warmup_runs
    stats["benchmark_runs"] = benchmark_runs
    return stats


def measure_classifier_latency(
    classifier_predict_fn: Callable[[], Any],
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> Dict[str, Any]:
    """
    Measures classifier inference latency.

    Parameters
    ----------
    classifier_predict_fn : callable
        Function executing one classification pass on an input array.
    warmup_runs : int
    benchmark_runs : int

    Returns
    -------
    dict
        Latency metrics in milliseconds.
    """
    return benchmark_function(
        classifier_predict_fn,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )


def measure_udp_roundtrip_latency(
    sender_fn: Callable[[Dict[str, Any]], bool],
    receiver_fn: Callable[[], Dict[str, Any]],
    sample_payload: Dict[str, Any],
    warmup_runs: int = 5,
    benchmark_runs: int = 30,
) -> Dict[str, Any]:
    """
    Measures UDP localhost datagram transmission and reception latency.

    Parameters
    ----------
    sender_fn : callable
        Function to send a message dict.
    receiver_fn : callable
        Function to receive and decode a message.
    sample_payload : dict
        Standard test message payload.
    warmup_runs : int
    benchmark_runs : int

    Returns
    -------
    dict
        Roundtrip latency metrics in milliseconds.
    """
    def _roundtrip():
        sender_fn(sample_payload)
        receiver_fn()

    return benchmark_function(
        _roundtrip,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )


def measure_simulator_step_latency(
    sim_step_fn: Callable[[float], Any],
    dt: float = 0.016,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, Any]:
    """
    Measures digital twin simulator physics step latency.

    Parameters
    ----------
    sim_step_fn : callable
        Function executing one step(dt).
    dt : float
        Simulation timestep in seconds.
    warmup_runs : int
    benchmark_runs : int

    Returns
    -------
    dict
        Step latency metrics in milliseconds.
    """
    def _step():
        sim_step_fn(dt)

    return benchmark_function(
        _step,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
