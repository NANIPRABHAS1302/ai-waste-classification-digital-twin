"""
Telemetry recording module for the waste classification & sorting system.

Records structured operational events:
- Classification events (confidence, entropy, reliability gate decision)
- Human gesture supervisor events (override, unresolved)
- UDP transmission and reception events
- Simulator sorting events (object spawn, gate divert, bin sort, fall-off)

Outputs clean, tabular CSV logs and structured JSONL logs for later analysis.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Canonical CSV field order
TELEMETRY_FIELDNAMES: List[str] = [
    "timestamp",
    "event_type",
    "object_id",
    "class_id",
    "class_name",
    "confidence",
    "entropy",
    "reliability_decision",
    "decision_source",
    "override",
    "simulation_state",
    "sorting_result",
    "target_bin",
    "classifier_latency_ms",
    "udp_latency_ms",
    "simulator_latency_ms",
    "e2e_latency_ms",
]


@dataclass
class TelemetryRecord:
    """Represents a single structured telemetry event."""

    timestamp: float = field(default_factory=time.time)
    event_type: str = "unknown"
    object_id: Optional[int] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    reliability_decision: Optional[str] = None
    decision_source: Optional[str] = None
    override: Optional[bool] = None
    simulation_state: Optional[str] = None
    sorting_result: Optional[str] = None
    target_bin: Optional[int] = None
    classifier_latency_ms: Optional[float] = None
    udp_latency_ms: Optional[float] = None
    simulator_latency_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert record to format suitable for CSV writing."""
        d = self.to_dict()
        row: Dict[str, Any] = {}
        for k in TELEMETRY_FIELDNAMES:
            val = d.get(k)
            if val is None:
                row[k] = ""
            elif isinstance(val, float):
                row[k] = round(val, 6)
            elif isinstance(val, bool):
                row[k] = "true" if val else "false"
            else:
                row[k] = str(val)
        return row


class TelemetryLogger:
    """
    Logs structured telemetry events to in-memory history, CSV file, and/or JSONL file.

    Parameters
    ----------
    csv_path : str or Path, optional
        Target path for CSV log file.
    jsonl_path : str or Path, optional
        Target path for JSONL log file.
    flush_immediate : bool
        If True, flush file buffers after every write.
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        jsonl_path: Optional[Union[str, Path]] = None,
        flush_immediate: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.flush_immediate = flush_immediate

        self.records: List[TelemetryRecord] = []
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None

        self._init_files()

    def _init_files(self) -> None:
        """Initializes target file streams and writes CSV header if needed."""
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            self._csv_file = open(self.csv_path, mode="a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=TELEMETRY_FIELDNAMES)
            if not file_exists:
                self._csv_writer.writeheader()
                if self.flush_immediate:
                    self._csv_file.flush()

        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = open(self.jsonl_path, mode="a", encoding="utf-8")

    def log(self, record: Union[TelemetryRecord, Dict[str, Any]]) -> TelemetryRecord:
        """
        Logs a telemetry record.

        Parameters
        ----------
        record : TelemetryRecord or dict
            The telemetry record to log.
        """
        if isinstance(record, dict):
            # Filter known fields, ignore unknown or malformed fields gracefully
            known = {k: v for k, v in record.items() if k in TELEMETRY_FIELDNAMES}
            rec = TelemetryRecord(**known)
        elif isinstance(record, TelemetryRecord):
            rec = record
        else:
            raise TypeError(f"record must be TelemetryRecord or dict, got {type(record)}")

        self.records.append(rec)

        if self._csv_writer and self._csv_file:
            self._csv_writer.writerow(rec.to_csv_row())
            if self.flush_immediate:
                self._csv_file.flush()

        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(rec.to_dict()) + "\n")
            if self.flush_immediate:
                self._jsonl_file.flush()

        return rec

    def get_records(self) -> List[TelemetryRecord]:
        """Returns the in-memory list of records."""
        return list(self.records)

    def clear(self) -> None:
        """Clears in-memory records (does not wipe file history)."""
        self.records.clear()

    def close(self) -> None:
        """Closes any open file handlers."""
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

        if self._jsonl_file:
            try:
                self._jsonl_file.close()
            except Exception:
                pass
            self._jsonl_file = None
