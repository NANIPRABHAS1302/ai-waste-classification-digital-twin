"""
Metrics and evaluation module for waste classification telemetry.

Calculates actual summary statistics and key performance indicators (KPIs)
from structured telemetry records:
- Total objects processed
- Class distribution
- Reliability statistics (ACCEPT vs UNCERTAIN count and rate)
- Human-in-the-loop statistics (override count and rate, unresolved count and rate)
- Sorting efficiency and accuracy (sorted vs fallen)
- Latency statistics (mean, median, min, max, std dev for all latency fields)
- System throughput (events per second)

If certain metrics cannot yet be computed (e.g. before end-to-end integration
in Phase 7), an explicit "not available" indicator or None is returned.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.telemetry.telemetry_logger import TelemetryRecord


def _compute_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Helper to compute mean, median, min, max, std_dev for a list of numbers."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std_dev": None,
        }
    vals = list(values)
    count = len(vals)
    mean_val = float(statistics.mean(vals))
    med_val = float(statistics.median(vals))
    min_val = float(min(vals))
    max_val = float(max(vals))
    std_val = float(statistics.stdev(vals)) if count > 1 else 0.0
    return {
        "count": count,
        "mean": round(mean_val, 4),
        "median": round(med_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "std_dev": round(std_val, 4),
    }


def compute_metrics(
    records: Union[Sequence[TelemetryRecord], Sequence[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Computes system and operational metrics from a sequence of telemetry records.

    Parameters
    ----------
    records : sequence of TelemetryRecord or dicts
        The telemetry events to analyze.

    Returns
    -------
    dict
        Aggregated metrics dictionary with typed, deterministic values.
    """
    raw_records: List[Dict[str, Any]] = [
        r.to_dict() if isinstance(r, TelemetryRecord) else dict(r) for r in records
    ]

    total_records = len(raw_records)
    if total_records == 0:
        return {
            "total_records": 0,
            "total_objects": 0,
            "class_distribution": {},
            "reliability": {
                "accept_count": 0,
                "uncertain_count": 0,
                "accept_rate": None,
                "uncertain_rate": None,
            },
            "supervision": {
                "override_count": 0,
                "unresolved_count": 0,
                "override_rate": None,
                "unresolved_rate": None,
            },
            "sorting": {
                "sorted_count": 0,
                "fallen_count": 0,
                "total_processed": 0,
                "sorting_accuracy": None,
            },
            "latency": {
                "classifier_ms": _compute_stats([]),
                "udp_ms": _compute_stats([]),
                "simulator_ms": _compute_stats([]),
                "e2e_ms": _compute_stats([]),
            },
            "throughput_events_per_sec": None,
            "status": "empty_dataset",
        }

    # 1. Total objects and class distribution
    class_dist: Dict[str, int] = {}
    unique_objects = set()

    for r in raw_records:
        obj_id = r.get("object_id")
        if obj_id is not None:
            unique_objects.add(obj_id)

        cname = r.get("class_name")
        if cname:
            class_dist[cname] = class_dist.get(cname, 0) + 1

    total_objects = len(unique_objects)

    # 2. Reliability metrics
    accept_count = sum(1 for r in raw_records if r.get("reliability_decision") == "ACCEPT")
    uncertain_count = sum(1 for r in raw_records if r.get("reliability_decision") == "UNCERTAIN")
    rel_total = accept_count + uncertain_count
    accept_rate = (accept_count / rel_total) if rel_total > 0 else None
    uncertain_rate = (uncertain_count / rel_total) if rel_total > 0 else None

    # 3. Supervision metrics
    override_count = sum(1 for r in raw_records if r.get("override") is True)
    unresolved_count = sum(1 for r in raw_records if r.get("decision_source") == "unresolved")
    supervision_events = [r for r in raw_records if r.get("reliability_decision") == "UNCERTAIN"]
    sup_denom = len(supervision_events) if supervision_events else rel_total
    override_rate = (override_count / sup_denom) if sup_denom > 0 else None
    unresolved_rate = (unresolved_count / sup_denom) if sup_denom > 0 else None

    # 4. Sorting metrics
    sorted_count = sum(1 for r in raw_records if r.get("sorting_result") == "SORTED")
    fallen_count = sum(1 for r in raw_records if r.get("sorting_result") == "FALLEN")
    total_sorted_fallen = sorted_count + fallen_count
    sorting_accuracy = (
        (sorted_count / total_sorted_fallen) if total_sorted_fallen > 0 else None
    )

    # 5. Latency distributions
    classifier_latencies = [
        float(r["classifier_latency_ms"])
        for r in raw_records
        if r.get("classifier_latency_ms") is not None
    ]
    udp_latencies = [
        float(r["udp_latency_ms"])
        for r in raw_records
        if r.get("udp_latency_ms") is not None
    ]
    simulator_latencies = [
        float(r["simulator_latency_ms"])
        for r in raw_records
        if r.get("simulator_latency_ms") is not None
    ]
    e2e_latencies = [
        float(r["e2e_latency_ms"])
        for r in raw_records
        if r.get("e2e_latency_ms") is not None
    ]

    # 6. Throughput
    timestamps = [
        float(r["timestamp"]) for r in raw_records if r.get("timestamp") is not None
    ]
    if len(timestamps) > 1:
        time_span = max(timestamps) - min(timestamps)
        throughput = (
            round((total_records - 1) / time_span, 2) if time_span > 0 else None
        )
    else:
        throughput = None

    return {
        "total_records": total_records,
        "total_objects": total_objects,
        "class_distribution": class_dist,
        "reliability": {
            "accept_count": accept_count,
            "uncertain_count": uncertain_count,
            "accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
            "uncertain_rate": (
                round(uncertain_rate, 4) if uncertain_rate is not None else None
            ),
        },
        "supervision": {
            "override_count": override_count,
            "unresolved_count": unresolved_count,
            "override_rate": (
                round(override_rate, 4) if override_rate is not None else None
            ),
            "unresolved_rate": (
                round(unresolved_rate, 4) if unresolved_rate is not None else None
            ),
        },
        "sorting": {
            "sorted_count": sorted_count,
            "fallen_count": fallen_count,
            "total_processed": total_sorted_fallen,
            "sorting_accuracy": (
                round(sorting_accuracy, 4) if sorting_accuracy is not None else None
            ),
        },
        "latency": {
            "classifier_ms": _compute_stats(classifier_latencies),
            "udp_ms": _compute_stats(udp_latencies),
            "simulator_ms": _compute_stats(simulator_latencies),
            "e2e_ms": _compute_stats(e2e_latencies),
        },
        "throughput_events_per_sec": throughput,
        "status": "valid",
    }


def load_telemetry_csv(csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads telemetry records from a CSV file into a list of parsed dictionaries.

    Parameters
    ----------
    csv_path : str or Path
        Path to the telemetry CSV file.

    Returns
    -------
    list of dict
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d: Dict[str, Any] = {}
            for k, v in row.items():
                if v == "" or v is None:
                    d[k] = None
                elif k in ("object_id", "class_id", "target_bin"):
                    try:
                        d[k] = int(v)
                    except ValueError:
                        d[k] = None
                elif k in (
                    "timestamp",
                    "confidence",
                    "entropy",
                    "classifier_latency_ms",
                    "udp_latency_ms",
                    "simulator_latency_ms",
                    "e2e_latency_ms",
                ):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = None
                elif k == "override":
                    d[k] = v.lower() == "true"
                else:
                    d[k] = v
            records.append(d)
    return records
