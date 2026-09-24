"""
Unit and integration tests for telemetry, latency tracking, and metrics — Phase 6.

Test Coverage:
  1. Telemetry record creation & default values
  2. Required & optional fields
  3. CSV serialization & roundtrip reading
  4. JSONL serialization
  5. Missing optional fields handled gracefully
  6. Malformed record / unknown field handling
  7. Latency calculation helper (_compute_stats)
  8. Mean latency calculation
  9. Median latency calculation
  10. Sorting accuracy calculation
  11. Uncertainty rate calculation
  12. Override rate calculation
  13. Unresolved rate calculation
  14. Empty dataset handling
  15. Deterministic metric calculation
  16. Repeated telemetry logging & stream flushing
  17. Benchmark function with warm-up
  18. Classifier latency measurement simulation
  19. UDP roundtrip measurement simulation
  20. Simulator step measurement simulation
  21. Phase 1-5 regression imports
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from src.telemetry.latency_tracker import (
    benchmark_function,
    measure_classifier_latency,
    measure_simulator_step_latency,
    measure_udp_roundtrip_latency,
)
from src.telemetry.metrics import _compute_stats, compute_metrics, load_telemetry_csv
from src.telemetry.telemetry_logger import (
    TELEMETRY_FIELDNAMES,
    TelemetryLogger,
    TelemetryRecord,
)


class TestTelemetryRecord(unittest.TestCase):
    """Tests for TelemetryRecord dataclass and formatting."""

    def test_01_creation_defaults(self):
        rec = TelemetryRecord()
        self.assertIsInstance(rec.timestamp, float)
        self.assertEqual(rec.event_type, "unknown")
        self.assertIsNone(rec.object_id)
        self.assertIsNone(rec.class_id)
        self.assertIsNone(rec.confidence)
        self.assertIsNone(rec.override)

    def test_02_field_assignment(self):
        rec = TelemetryRecord(
            event_type="classification",
            object_id=42,
            class_id=4,
            class_name="plastic",
            confidence=0.965,
            entropy=0.18,
            reliability_decision="ACCEPT",
            decision_source="classifier",
            override=False,
            classifier_latency_ms=12.4,
        )
        self.assertEqual(rec.object_id, 42)
        self.assertEqual(rec.class_name, "plastic")
        self.assertEqual(rec.confidence, 0.965)
        self.assertEqual(rec.reliability_decision, "ACCEPT")
        self.assertFalse(rec.override)

    def test_03_csv_row_formatting(self):
        rec = TelemetryRecord(
            object_id=5,
            class_id=1,
            class_name="glass",
            confidence=0.88,
            override=True,
        )
        row = rec.to_csv_row()
        self.assertEqual(row["object_id"], "5")
        self.assertEqual(row["class_id"], "1")
        self.assertEqual(row["class_name"], "glass")
        self.assertEqual(row["confidence"], 0.88)
        self.assertEqual(row["override"], "true")
        self.assertEqual(row["entropy"], "")  # None -> empty string


class TestTelemetryLogger(unittest.TestCase):
    """Tests for TelemetryLogger file output and management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test_telemetry.csv"
        self.jsonl_path = Path(self.temp_dir) / "test_telemetry.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_04_csv_and_jsonl_output(self):
        logger = TelemetryLogger(csv_path=self.csv_path, jsonl_path=self.jsonl_path)
        rec1 = TelemetryRecord(event_type="spawn", object_id=1, class_id=0, class_name="cardboard")
        rec2 = TelemetryRecord(
            event_type="decision",
            object_id=1,
            class_id=0,
            class_name="cardboard",
            confidence=0.91,
            reliability_decision="ACCEPT",
        )
        logger.log(rec1)
        logger.log(rec2)
        logger.close()

        # Verify CSV
        self.assertTrue(self.csv_path.exists())
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, TELEMETRY_FIELDNAMES)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1], "spawn")
            self.assertEqual(rows[1][1], "decision")

        # Verify JSONL
        self.assertTrue(self.jsonl_path.exists())
        with open(self.jsonl_path, mode="r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["event_type"], "spawn")
            self.assertEqual(lines[1]["class_name"], "cardboard")

    def test_05_log_from_dict_and_malformed_fields(self):
        logger = TelemetryLogger(csv_path=self.csv_path)
        # Pass dictionary with both valid and unknown/malformed keys
        logger.log({
            "event_type": "divert",
            "object_id": 99,
            "class_id": 2,
            "random_junk_key": "ignore_me",
        })
        logger.close()

        records = logger.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, "divert")
        self.assertEqual(records[0].object_id, 99)
        self.assertEqual(records[0].class_id, 2)
        self.assertFalse(hasattr(records[0], "random_junk_key"))

    def test_06_load_telemetry_csv(self):
        logger = TelemetryLogger(csv_path=self.csv_path)
        logger.log(TelemetryRecord(object_id=10, class_id=3, confidence=0.75, override=True))
        logger.close()

        loaded = load_telemetry_csv(self.csv_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["object_id"], 10)
        self.assertEqual(loaded[0]["class_id"], 3)
        self.assertEqual(loaded[0]["confidence"], 0.75)
        self.assertTrue(loaded[0]["override"])


class TestTelemetryMetrics(unittest.TestCase):
    """Tests for compute_metrics and statistical calculations."""

    def test_07_empty_dataset(self):
        res = compute_metrics([])
        self.assertEqual(res["total_records"], 0)
        self.assertEqual(res["status"], "empty_dataset")
        self.assertIsNone(res["reliability"]["accept_rate"])
        self.assertIsNone(res["sorting"]["sorting_accuracy"])
        self.assertIsNone(res["latency"]["classifier_ms"]["mean"])

    def test_08_reliability_and_supervision_rates(self):
        records = [
            TelemetryRecord(object_id=1, reliability_decision="ACCEPT", decision_source="classifier"),
            TelemetryRecord(object_id=2, reliability_decision="ACCEPT", decision_source="classifier"),
            TelemetryRecord(object_id=3, reliability_decision="UNCERTAIN", decision_source="gesture", override=True),
            TelemetryRecord(object_id=4, reliability_decision="UNCERTAIN", decision_source="unresolved", override=False),
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["total_records"], 4)
        self.assertEqual(metrics["reliability"]["accept_count"], 2)
        self.assertEqual(metrics["reliability"]["uncertain_count"], 2)
        self.assertAlmostEqual(metrics["reliability"]["accept_rate"], 0.5)
        self.assertAlmostEqual(metrics["reliability"]["uncertain_rate"], 0.5)

        self.assertEqual(metrics["supervision"]["override_count"], 1)
        self.assertEqual(metrics["supervision"]["unresolved_count"], 1)
        self.assertAlmostEqual(metrics["supervision"]["override_rate"], 0.5)
        self.assertAlmostEqual(metrics["supervision"]["unresolved_rate"], 0.5)

    def test_09_sorting_accuracy(self):
        records = [
            TelemetryRecord(object_id=1, sorting_result="SORTED"),
            TelemetryRecord(object_id=2, sorting_result="SORTED"),
            TelemetryRecord(object_id=3, sorting_result="SORTED"),
            TelemetryRecord(object_id=4, sorting_result="FALLEN"),
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["sorting"]["sorted_count"], 3)
        self.assertEqual(metrics["sorting"]["fallen_count"], 1)
        self.assertAlmostEqual(metrics["sorting"]["sorting_accuracy"], 0.75)

    def test_10_latency_statistics(self):
        records = [
            TelemetryRecord(classifier_latency_ms=10.0, udp_latency_ms=1.0),
            TelemetryRecord(classifier_latency_ms=20.0, udp_latency_ms=2.0),
            TelemetryRecord(classifier_latency_ms=30.0, udp_latency_ms=3.0),
        ]
        metrics = compute_metrics(records)
        c_stats = metrics["latency"]["classifier_ms"]
        self.assertEqual(c_stats["count"], 3)
        self.assertAlmostEqual(c_stats["mean"], 20.0)
        self.assertAlmostEqual(c_stats["median"], 20.0)
        self.assertAlmostEqual(c_stats["min"], 10.0)
        self.assertAlmostEqual(c_stats["max"], 30.0)

    def test_11_determinism(self):
        records = [
            TelemetryRecord(object_id=i, class_name="metal", classifier_latency_ms=15.0 + i)
            for i in range(10)
        ]
        m1 = compute_metrics(records)
        m2 = compute_metrics(records)
        self.assertEqual(m1, m2)


class TestLatencyTracker(unittest.TestCase):
    """Tests for benchmarking helper functions."""

    def test_12_benchmark_function_execution(self):
        def dummy_task():
            time.sleep(0.002)

        stats = benchmark_function(dummy_task, warmup_runs=2, benchmark_runs=5)
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["warmup_runs"], 2)
        self.assertEqual(stats["benchmark_runs"], 5)
        self.assertGreater(stats["mean"], 1.0)
        self.assertGreater(stats["median"], 1.0)

    def test_13_measure_classifier_latency(self):
        def dummy_classifier():
            pass

        stats = measure_classifier_latency(dummy_classifier, warmup_runs=2, benchmark_runs=5)
        self.assertEqual(stats["count"], 5)
        self.assertIsNotNone(stats["mean"])

    def test_14_measure_udp_roundtrip_latency(self):
        storage = []
        def sender(msg):
            storage.append(msg)
            return True
        def receiver():
            return storage.pop(0) if storage else {}

        stats = measure_udp_roundtrip_latency(
            sender, receiver, {"test": 1}, warmup_runs=2, benchmark_runs=5
        )
        self.assertEqual(stats["count"], 5)

    def test_15_measure_simulator_step_latency(self):
        def sim_step(dt):
            pass

        stats = measure_simulator_step_latency(sim_step, warmup_runs=2, benchmark_runs=5)
        self.assertEqual(stats["count"], 5)


class TestRegressionImports(unittest.TestCase):
    """Verifies that all Phase 1-5 modules remain functional and intact."""

    def test_phase1_imports(self):
        from src.realtime.realtime_classifier import WasteClassifier
        self.assertTrue(callable(WasteClassifier))

    def test_phase2_imports(self):
        from src.reliability.confidence_gate import ConfidenceGate
        from src.reliability.ood_filter import OodFilter
        self.assertTrue(callable(ConfidenceGate))
        self.assertTrue(callable(OodFilter))

    def test_phase3_imports(self):
        from src.gesture.gesture_detector import GestureDetector
        from src.gesture.gesture_supervisor import GestureSupervisor
        self.assertTrue(callable(GestureDetector))
        self.assertTrue(callable(GestureSupervisor))

    def test_phase4_imports(self):
        from src.communication.udp_receiver import UdpReceiver
        from src.communication.udp_sender import UdpSender, build_message
        self.assertTrue(callable(UdpSender))
        self.assertTrue(callable(UdpReceiver))
        self.assertTrue(callable(build_message))

    def test_phase5_imports(self):
        from src.simulator.conveyor import Conveyor
        from src.simulator.renderer import PygameRenderer
        from src.simulator.sorting_gate import SortingGate
        from src.simulator.sorting_simulator import SortingSimulator
        from src.simulator.waste_object import WasteObject
        self.assertTrue(callable(Conveyor))
        self.assertTrue(callable(SortingGate))
        self.assertTrue(callable(SortingSimulator))
        self.assertTrue(callable(WasteObject))
        self.assertTrue(callable(PygameRenderer))


if __name__ == "__main__":
    unittest.main()
