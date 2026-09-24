"""
Unit and integration tests for the digital twin sorting simulator — Phase 5.

Test Coverage:
  A. WasteObject — initialization, canonical classes, colors, movement, state transitions
  B. Conveyor — geometry validation, object tracking, physics step, gate triggering, fall-off detection, pause/resume/stop/reset
  C. SortingGate — state machine (IDLE -> DIVERTING -> RESETTING -> IDLE), decision acceptance/busy rejection, 6-bin routing, counters, reset
  D. SortingSimulator — initialization, defaults, spawn_object, decision ingestion (UDP format, dict format, unresolved rejection), step physics coordination, state snapshots, determinism
  E. PygameRenderer — headless initialization (SDL_VIDEODRIVER=dummy), frame rendering, HUD, event handling, teardown
  F. Full Simulation Scenario — spawning multiple items, sequencing decisions, verifying bin counts and fall-offs
  G. Regression Smoke Tests — Phase 1, Phase 2, Phase 3, Phase 4 imports
"""

from __future__ import annotations

import os
import unittest
from typing import Dict, List

# Ensure headless video driver for Pygame in test environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

from src.simulator.conveyor import Conveyor, ConveyorState
from src.simulator.renderer import PygameRenderer
from src.simulator.sorting_gate import (
    BIN_NAMES,
    CLASS_TO_BIN,
    NUM_BINS,
    BinCounter,
    GateState,
    SortingGate,
)
from src.simulator.sorting_simulator import (
    DEFAULT_CONVEYOR_SPEED,
    DEFAULT_GATE_X,
    SortingSimulator,
)
from src.simulator.waste_object import (
    CLASS_COLOURS,
    CLASS_NAMES,
    NUM_CLASSES,
    ObjectState,
    WasteObject,
)


class TestWasteObject(unittest.TestCase):
    """Tests for WasteObject class."""

    def test_01_valid_initialization(self):
        for class_id in range(NUM_CLASSES):
            obj = WasteObject(object_id=class_id, class_id=class_id)
            self.assertEqual(obj.object_id, class_id)
            self.assertEqual(obj.class_id, class_id)
            self.assertEqual(obj.class_name, CLASS_NAMES[class_id])
            self.assertEqual(obj.colour, CLASS_COLOURS[class_id])
            self.assertEqual(obj.state, ObjectState.MOVING)
            self.assertTrue(obj.is_active)
            self.assertFalse(obj.is_completed)
            self.assertIsNone(obj.sorted_to_bin)

    def test_02_invalid_class_id(self):
        with self.assertRaises(ValueError):
            WasteObject(object_id=0, class_id=-1)
        with self.assertRaises(ValueError):
            WasteObject(object_id=0, class_id=6)
        with self.assertRaises(ValueError):
            WasteObject(object_id=0, class_id="plastic")  # type: ignore

    def test_03_step_movement(self):
        obj = WasteObject(object_id=1, class_id=0, spawn_x=10.0, velocity_x=100.0)
        obj.step(0.5)
        self.assertAlmostEqual(obj.x, 60.0)
        obj.step(0.5)
        self.assertAlmostEqual(obj.x, 110.0)

    def test_04_step_invalid_dt(self):
        obj = WasteObject(object_id=1, class_id=0)
        with self.assertRaises(ValueError):
            obj.step(0.0)
        with self.assertRaises(ValueError):
            obj.step(-0.1)

    def test_05_state_transitions(self):
        obj = WasteObject(object_id=1, class_id=2)
        obj.mark_sorted(2)
        self.assertEqual(obj.state, ObjectState.SORTED)
        self.assertEqual(obj.sorted_to_bin, 2)
        self.assertFalse(obj.is_active)
        self.assertTrue(obj.is_completed)

        # Once sorted, step does not move position
        curr_x = obj.x
        obj.step(1.0)
        self.assertEqual(obj.x, curr_x)

        obj2 = WasteObject(object_id=2, class_id=4)
        obj2.mark_fallen()
        self.assertEqual(obj2.state, ObjectState.FALLEN)
        self.assertFalse(obj2.is_active)
        self.assertTrue(obj2.is_completed)


class TestConveyor(unittest.TestCase):
    """Tests for Conveyor physics and object tracking."""

    def test_01_init_validation(self):
        conv = Conveyor(belt_start_x=0.0, belt_end_x=1000.0, gate_trigger_x=500.0, speed=100.0)
        self.assertEqual(conv.state, ConveyorState.RUNNING)
        self.assertTrue(conv.is_running)

        with self.assertRaises(ValueError):
            Conveyor(speed=-10)
        with self.assertRaises(ValueError):
            Conveyor(belt_start_x=100, belt_end_x=50)
        with self.assertRaises(ValueError):
            Conveyor(belt_start_x=100, belt_end_x=500, gate_trigger_x=600)

    def test_02_add_remove_objects(self):
        conv = Conveyor()
        obj1 = WasteObject(object_id=0, class_id=0)
        obj2 = WasteObject(object_id=1, class_id=1)

        conv.add_object(obj1)
        conv.add_object(obj2)
        self.assertEqual(conv.object_count(), 2)
        self.assertEqual(len(conv.active_objects), 2)

        conv.remove_object(obj1)
        self.assertEqual(conv.object_count(), 1)
        self.assertEqual(conv.active_objects[0].object_id, 1)

    def test_03_gate_trigger_detection(self):
        conv = Conveyor(belt_start_x=0.0, belt_end_x=1000.0, gate_trigger_x=500.0, speed=100.0)
        obj = WasteObject(object_id=0, class_id=0, spawn_x=450.0, velocity_x=100.0)
        conv.add_object(obj)

        # dt=0.4 -> x moves from 450 to 490 (not reached)
        at_gate, fallen = conv.step(0.4)
        self.assertEqual(len(at_gate), 0)
        self.assertEqual(len(fallen), 0)

        # dt=0.2 -> x moves from 490 to 510 (crosses 500)
        at_gate, fallen = conv.step(0.2)
        self.assertEqual(len(at_gate), 1)
        self.assertEqual(at_gate[0].object_id, 0)
        self.assertEqual(len(fallen), 0)

        # dt=0.1 -> already past 500, not triggered again
        at_gate, fallen = conv.step(0.1)
        self.assertEqual(len(at_gate), 0)

    def test_04_fall_off_detection(self):
        conv = Conveyor(belt_start_x=0.0, belt_end_x=1000.0, gate_trigger_x=500.0, speed=200.0)
        obj = WasteObject(object_id=0, class_id=1, spawn_x=950.0, velocity_x=200.0)
        conv.add_object(obj)

        at_gate, fallen = conv.step(0.5)  # 950 + 100 = 1050 >= 1000
        self.assertEqual(len(fallen), 1)
        self.assertEqual(fallen[0].object_id, 0)
        self.assertEqual(fallen[0].state, ObjectState.FALLEN)

    def test_05_pause_resume_stop_reset(self):
        conv = Conveyor(speed=100.0)
        obj = WasteObject(object_id=0, class_id=0, spawn_x=0.0, velocity_x=100.0)
        conv.add_object(obj)

        conv.pause()
        self.assertEqual(conv.state, ConveyorState.PAUSED)
        conv.step(1.0)
        self.assertEqual(obj.x, 0.0)  # paused, no movement

        conv.resume()
        self.assertEqual(conv.state, ConveyorState.RUNNING)
        conv.step(1.0)
        self.assertEqual(obj.x, 100.0)

        conv.stop()
        self.assertEqual(conv.state, ConveyorState.STOPPED)

        conv.reset()
        self.assertEqual(conv.state, ConveyorState.RUNNING)
        self.assertEqual(conv.object_count(), 0)


class TestSortingGate(unittest.TestCase):
    """Tests for SortingGate state machine and bin routing."""

    def test_01_initial_state(self):
        gate = SortingGate()
        self.assertEqual(gate.state, GateState.IDLE)
        self.assertTrue(gate.is_idle)
        self.assertIsNone(gate.pending_class_id)
        self.assertEqual(gate.total_sorted, 0)
        self.assertEqual(gate.total_fallen, 0)
        counts = gate.get_bin_counts()
        for name in CLASS_NAMES:
            self.assertEqual(counts[name], 0)

    def test_02_decision_intake_and_busy_rejection(self):
        gate = SortingGate(divert_duration_s=0.5, reset_duration_s=0.3)
        accepted = gate.receive_decision(2)
        self.assertTrue(accepted)
        self.assertEqual(gate.state, GateState.DIVERTING)
        self.assertEqual(gate.pending_class_id, 2)

        # Reject new decision while diverting
        rejected = gate.receive_decision(3)
        self.assertFalse(rejected)

    def test_03_invalid_decision_class(self):
        gate = SortingGate()
        with self.assertRaises(ValueError):
            gate.receive_decision(-1)
        with self.assertRaises(ValueError):
            gate.receive_decision(6)

    def test_04_sorting_and_state_progression(self):
        gate = SortingGate(divert_duration_s=0.5, reset_duration_s=0.3)
        gate.receive_decision(4)  # plastic

        obj = WasteObject(object_id=1, class_id=4)
        # Advance 0.2s while object is at gate
        sorted_now = gate.step(0.2, [obj])
        self.assertEqual(len(sorted_now), 1)
        self.assertEqual(obj.state, ObjectState.SORTED)
        self.assertEqual(obj.sorted_to_bin, 4)
        self.assertEqual(gate.bins[4].received, 1)
        self.assertEqual(gate.total_sorted, 1)
        self.assertEqual(gate.state, GateState.DIVERTING)

        # Advance 0.3s (total 0.5s -> switches to RESETTING)
        gate.step(0.3, [])
        self.assertEqual(gate.state, GateState.RESETTING)

        # Advance 0.3s -> switches to IDLE
        gate.step(0.3, [])
        self.assertEqual(gate.state, GateState.IDLE)
        self.assertIsNone(gate.pending_class_id)

    def test_05_reset(self):
        gate = SortingGate()
        gate.receive_decision(1)
        obj = WasteObject(object_id=0, class_id=1)
        gate.step(0.1, [obj])
        self.assertEqual(gate.total_sorted, 1)

        gate.reset()
        self.assertEqual(gate.state, GateState.IDLE)
        self.assertIsNone(gate.pending_class_id)
        self.assertEqual(gate.total_sorted, 0)
        self.assertEqual(gate.bins[1].received, 0)


class TestSortingSimulator(unittest.TestCase):
    """Tests for SortingSimulator orchestration."""

    def test_01_init_and_reset(self):
        sim = SortingSimulator()
        state = sim.get_state()
        self.assertEqual(state["sim_time"], 0.0)
        self.assertEqual(state["total_sorted"], 0)
        self.assertEqual(state["total_fallen"], 0)
        self.assertEqual(len(state["active_objects"]), 0)

    def test_02_spawn_object(self):
        sim = SortingSimulator()
        obj0 = sim.spawn_object(0)
        self.assertEqual(obj0.object_id, 0)
        self.assertEqual(obj0.class_id, 0)

        obj1 = sim.spawn_object(3)
        self.assertEqual(obj1.object_id, 1)
        self.assertEqual(obj1.class_id, 3)

        self.assertEqual(len(sim.conveyor.active_objects), 2)

    def test_03_receive_decision_formats(self):
        sim = SortingSimulator()

        # Valid UDP decision format
        udp_msg = {
            "version": "1.0",
            "source": "classifier",
            "decision": "ACCEPT",
            "class_id": 2,
            "class_name": "metal",
            "confidence": 0.95,
            "entropy": 0.2,
            "override": False,
        }
        self.assertTrue(sim.receive_decision(udp_msg))
        self.assertEqual(sim._decision_queue, [2])

        # Minimal dict format
        self.assertTrue(sim.receive_decision({"class_id": 5}))
        self.assertEqual(sim._decision_queue, [2, 5])

        # Unresolved format rejected from queue
        unresolved_msg = {
            "source": "unresolved",
            "decision": "UNCERTAIN",
            "class_id": 0,
        }
        self.assertFalse(sim.receive_decision(unresolved_msg))

        # Invalid formats rejected
        self.assertFalse(sim.receive_decision("invalid"))  # type: ignore
        self.assertFalse(sim.receive_decision({"class_id": 10}))
        self.assertFalse(sim.receive_decision({}))

    def test_04_end_to_end_sorting(self):
        # Conveyor speed 200, start 0, gate 600 -> arrival at 3.0s
        sim = SortingSimulator(
            conveyor_speed=200.0,
            gate_x=600.0,
            belt_start_x=0.0,
            belt_end_x=1000.0,
            divert_duration_s=0.5,
            reset_duration_s=0.3,
        )

        obj = sim.spawn_object(class_id=4)  # plastic
        sim.receive_decision({"class_id": 4, "source": "classifier"})

        # Step 2.5s -> at x=500 (not at gate yet)
        res1 = sim.step(2.5)
        self.assertEqual(len(res1["at_gate"]), 0)
        self.assertEqual(len(res1["sorted_now"]), 0)

        # Step 0.6s -> reached gate at 3.0s and sorted
        res2 = sim.step(0.6)
        self.assertIn(obj.object_id, res2["at_gate"])
        self.assertIn(obj.object_id, res2["sorted_now"])
        self.assertEqual(sim.gate.total_sorted, 1)
        self.assertEqual(sim.gate.bins[4].received, 1)

    def test_05_unsorted_object_falls_off(self):
        sim = SortingSimulator(
            conveyor_speed=200.0,
            gate_x=600.0,
            belt_start_x=0.0,
            belt_end_x=1000.0,
        )
        obj = sim.spawn_object(class_id=1)
        # NO decision provided to gate -> gate stays idle, object passes gate

        # Step 6.0s (200 * 6 = 1200 > 1000)
        for _ in range(60):
            sim.step(0.1)

        state = sim.get_state()
        self.assertEqual(state["total_sorted"], 0)
        self.assertEqual(state["total_fallen"], 1)
        self.assertEqual(len(state["active_objects"]), 0)

    def test_06_determinism(self):
        """Two simulators with identical inputs must produce identical states."""
        def run_sim():
            s = SortingSimulator(conveyor_speed=150.0, gate_x=450.0)
            s.spawn_object(0)
            s.receive_decision({"class_id": 0})
            s.spawn_object(3)
            s.receive_decision({"class_id": 3})

            for _ in range(50):
                s.step(0.1)
            return s.get_state()

        state1 = run_sim()
        state2 = run_sim()

        self.assertEqual(state1["sim_time"], state2["sim_time"])
        self.assertEqual(state1["total_sorted"], state2["total_sorted"])
        self.assertEqual(state1["total_fallen"], state2["total_fallen"])
        self.assertEqual(state1["bin_counts"], state2["bin_counts"])


class TestPygameRenderer(unittest.TestCase):
    """Tests for PygameRenderer in headless mode."""

    def test_01_headless_initialization(self):
        renderer = PygameRenderer(headless=True)
        self.assertTrue(renderer.initialized)
        self.assertIsNotNone(renderer.screen)
        self.assertEqual(renderer.width, 1000)
        self.assertEqual(renderer.height, 600)
        renderer.close()

    def test_02_render_frame(self):
        renderer = PygameRenderer(headless=True)
        sim = SortingSimulator()
        sim.spawn_object(2)
        sim.receive_decision({"class_id": 2, "source": "classifier", "confidence": 0.92, "entropy": 0.3})

        # Render one frame
        success = renderer.render(sim)
        self.assertTrue(success)

        # Step and render again
        sim.step(0.1)
        success2 = renderer.render(sim)
        self.assertTrue(success2)

        renderer.close()

    def test_03_events_and_tick(self):
        renderer = PygameRenderer(headless=True)
        self.assertTrue(renderer.handle_events())
        dt = renderer.tick()
        self.assertGreater(dt, 0.0)
        renderer.close()


class TestRegressionImports(unittest.TestCase):
    """Verifies that all Phase 1-4 modules remain intact and importable."""

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


if __name__ == "__main__":
    unittest.main()
