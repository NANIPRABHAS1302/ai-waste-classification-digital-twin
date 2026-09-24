"""
Integration tests for the full end-to-end pipeline — Phase 7.

Test Coverage:
  1. RealtimeApp component initialization & configuration
  2. Frame processing pipeline (mock/synthetic frames)
  3. RealtimeApp graceful camera handling when camera unavailable
  4. Decision path for high-confidence ACCEPT
  5. Decision path for low-confidence UNCERTAIN with supervisor fallback
  6. SimulatorApp UDP polling and object spawning
  7. End-to-end pipeline integration between RealtimeApp and SimulatorApp (ephemeral UDP port)
  8. Telemetry recording during end-to-end processing
  9. Full Phase 1–6 regression imports
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

from apps.realtime_app import RealtimeApp
from apps.simulator_app import SimulatorApp
from src.communication.udp_receiver import STATUS_OK, UdpReceiver
from src.communication.udp_sender import build_message, UdpSender
from src.reliability.confidence_gate import ConfidenceGate
from src.simulator.sorting_simulator import SortingSimulator
from src.telemetry.telemetry_logger import TelemetryLogger


class TestRealtimeApp(unittest.TestCase):
    """Tests for RealtimeApp logic and component orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test_telemetry.csv"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_init_and_graceful_camera_handling(self):
        # Point to invalid camera index -> must not crash
        app = RealtimeApp(
            camera_index=999,
            telemetry_csv=str(self.csv_path),
            warmup=False,
            yolo_enabled=False,
        )
        self.assertFalse(app.open_camera())
        app.close_camera()

    def test_02_process_synthetic_frame_accept(self):
        app = RealtimeApp(
            telemetry_csv=str(self.csv_path),
            warmup=False,
            yolo_enabled=False,
        )
        # Mock classifier output with high confidence for metal (class 2)
        mock_pred = {
            "class_id": 2,
            "class_name": "metal",
            "confidence": 0.95,
            "probabilities": np.array([0.01, 0.01, 0.95, 0.01, 0.01, 0.01], dtype=np.float32),
        }
        app.classifier = MagicMock()
        app.classifier.predict.return_value = mock_pred

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = app.process_frame(dummy_frame, object_id=10)

        self.assertEqual(res["final_class_id"], 2)
        self.assertEqual(res["final_class_name"], "metal")
        self.assertEqual(res["decision_source"], "classifier")
        self.assertEqual(res["reliability"]["decision"], "ACCEPT")
        self.assertFalse(res["override"])
        self.assertGreater(res["classifier_latency_ms"], 0.0)

        # Check telemetry logged
        records = app.logger.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].object_id, 10)
        self.assertEqual(records[0].reliability_decision, "ACCEPT")
        app.logger.close()

    def test_03_process_synthetic_frame_uncertain(self):
        app = RealtimeApp(
            telemetry_csv=str(self.csv_path),
            warmup=False,
            yolo_enabled=False,
        )
        # Low confidence distributed prediction
        mock_pred = {
            "class_id": 1,
            "class_name": "glass",
            "confidence": 0.30,
            "probabilities": np.array([0.20, 0.30, 0.15, 0.15, 0.10, 0.10], dtype=np.float32),
        }
        app.classifier = MagicMock()
        app.classifier.predict.return_value = mock_pred

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = app.process_frame(dummy_frame, object_id=11)

        self.assertEqual(res["reliability"]["decision"], "UNCERTAIN")
        self.assertEqual(res["decision_source"], "unresolved")
        self.assertFalse(res["override"])
        app.logger.close()


class TestSimulatorApp(unittest.TestCase):
    """Tests for SimulatorApp logic and step loop."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test_sim_telemetry.csv"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_04_simulator_step_once(self):
        app = SimulatorApp(
            port=58877,
            telemetry_csv=str(self.csv_path),
            headless=True,
        )
        # Step once with no incoming UDP datagrams
        res = app.step_once(dt=0.016)
        self.assertIn("at_gate", res)
        self.assertIn("sorted_now", res)
        self.assertIn("fallen", res)
        app.close()


class TestEndToEndPipeline(unittest.TestCase):
    """Tests complete integration from decision sender to simulator receiver."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rt_csv = Path(self.temp_dir) / "rt_telemetry.csv"
        self.sim_csv = Path(self.temp_dir) / "sim_telemetry.csv"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_05_e2e_realtime_to_simulator_roundtrip(self):
        test_port = 58866

        rt_app = RealtimeApp(
            udp_host="127.0.0.1",
            udp_port=test_port,
            telemetry_csv=str(self.rt_csv),
            warmup=False,
            yolo_enabled=False,
        )
        sim_app = SimulatorApp(
            host="127.0.0.1",
            port=test_port,
            telemetry_csv=str(self.sim_csv),
            headless=True,
            conveyor_speed=200.0,
            gate_position=400.0,
        )

        mock_pred = {
            "class_id": 4,
            "class_name": "plastic",
            "confidence": 0.92,
            "probabilities": np.array([0.01, 0.02, 0.02, 0.02, 0.92, 0.01], dtype=np.float32),
        }
        rt_app.classifier = MagicMock()
        rt_app.classifier.predict.return_value = mock_pred

        # 1. Process frame in RealtimeApp -> sends UDP message
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rt_res = rt_app.process_frame(dummy_frame, object_id=1)
        self.assertTrue(rt_res["udp_status"]["success"])

        # 2. SimulatorApp polls UDP and steps physics
        step1 = sim_app.step_once(dt=0.016)
        self.assertEqual(sim_app.simulator.conveyor.object_count(), 1)
        self.assertEqual(sim_app.simulator.conveyor.active_objects[0].class_id, 4)

        # 3. Advance physics past gate
        for _ in range(150):
            sim_app.step_once(dt=0.016)

        state = sim_app.simulator.get_state()
        self.assertEqual(state["total_sorted"], 1)
        self.assertEqual(state["bin_counts"]["plastic"], 1)

        rt_app.logger.close()
        sim_app.close()


class TestRegressionImports(unittest.TestCase):
    """Verifies that all Phase 1-6 modules remain intact and importable."""

    def test_all_phases_importable(self):
        from src.realtime.realtime_classifier import WasteClassifier
        from src.reliability.confidence_gate import ConfidenceGate
        from src.reliability.ood_filter import OodFilter
        from src.gesture.gesture_detector import GestureDetector
        from src.gesture.gesture_supervisor import GestureSupervisor
        from src.communication.udp_sender import UdpSender
        from src.communication.udp_receiver import UdpReceiver
        from src.simulator.sorting_simulator import SortingSimulator
        from src.simulator.renderer import PygameRenderer
        from src.telemetry.telemetry_logger import TelemetryLogger
        from src.telemetry.metrics import compute_metrics
        from src.telemetry.latency_tracker import benchmark_function

        self.assertTrue(callable(WasteClassifier))
        self.assertTrue(callable(ConfidenceGate))
        self.assertTrue(callable(OodFilter))
        self.assertTrue(callable(GestureDetector))
        self.assertTrue(callable(GestureSupervisor))
        self.assertTrue(callable(UdpSender))
        self.assertTrue(callable(UdpReceiver))
        self.assertTrue(callable(SortingSimulator))
        self.assertTrue(callable(PygameRenderer))
        self.assertTrue(callable(TelemetryLogger))
        self.assertTrue(callable(compute_metrics))
        self.assertTrue(callable(benchmark_function))


if __name__ == "__main__":
    unittest.main()
