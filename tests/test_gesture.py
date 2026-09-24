"""
Unit tests for the gesture supervision layer — Phase 3.

Covers:
  gesture_detector.py  — GestureDetector, FINGER_COUNT_TO_CLASS_ID,
                          count_extended_fingers, _DebounceBuffer,
                          process_finger_count()
  gesture_supervisor.py — GestureSupervisor.decide(), edge cases,
                          safety rule (ACCEPT cannot be overridden)

All tests are fully headless:
  - No physical webcam required
  - MediaPipe is NOT imported (tests use process_finger_count() or
    the GestureDetector._mp_hands_mock injection interface)
  - No TensorFlow model loading required for Phase 3 tests

Test groups:
  A. Gesture mapping (finger count → class ID → class name)
  B. GestureDetector.process_finger_count() direct API
  C. _DebounceBuffer temporal stability
  D. GestureSupervisor — ACCEPT path (safety rule)
  E. GestureSupervisor — UNCERTAIN + valid gesture (override)
  F. GestureSupervisor — UNCERTAIN + no gesture (unresolved)
  G. GestureSupervisor — edge cases and malformed input
  H. Determinism checks
  I. Regression: Phase 1 + Phase 2 smoke import
"""

import math
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.gesture.gesture_detector import (
    FINGER_COUNT_TO_CLASS_ID,
    GESTURE_CLASS_NAMES,
    GestureDetector,
    MediaPipeNotAvailableError,
    _DebounceBuffer,
    count_extended_fingers,
)
from src.gesture.gesture_supervisor import (
    GestureSupervisor,
    SOURCE_CLASSIFIER,
    SOURCE_GESTURE,
    SOURCE_UNRESOLVED,
    STATE_AUTO,
    STATE_MANUAL_OVERRIDE,
    STATE_UNRESOLVED,
    STATE_ERROR,
)
from src.reliability.confidence_gate import DECISION_ACCEPT, DECISION_UNCERTAIN


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_classifier_result(class_id: int, confidence: float = 0.85) -> dict:
    """Creates a minimal classifier result dict."""
    n = 6
    remainder = (1.0 - confidence) / (n - 1) if confidence < 1.0 else 0.0
    probs = [remainder] * n
    probs[class_id] = confidence
    return {
        "class_id": class_id,
        "class_name": GESTURE_CLASS_NAMES[class_id],
        "confidence": confidence,
        "probabilities": probs,
    }


def _make_reliability_result(decision: str, confidence: float = 0.85, entropy: float = 0.5) -> dict:
    return {
        "decision": decision,
        "max_confidence": confidence,
        "entropy": entropy,
        "predicted_class_id": 0,
        "predicted_class_name": "cardboard",
        "confidence_threshold": 0.70,
        "entropy_threshold": 1.20,
    }


def _make_gesture_result(finger_count: int) -> dict:
    """Creates a gesture result dict from a finger count (1-6)."""
    class_id = FINGER_COUNT_TO_CLASS_ID.get(finger_count)
    if class_id is None:
        return {
            "valid": False,
            "finger_count": finger_count,
            "class_id": None,
            "class_name": None,
            "confidence": 0.0,
            "hands_detected": 1,
            "reason": f"unmapped_finger_count_{finger_count}",
        }
    return {
        "valid": True,
        "finger_count": finger_count,
        "class_id": class_id,
        "class_name": GESTURE_CLASS_NAMES[class_id],
        "confidence": 1.0,
        "hands_detected": 1,
        "reason": "direct_finger_count",
    }


# ---------------------------------------------------------------------------
# A. Gesture Mapping Tests
# ---------------------------------------------------------------------------

class TestGestureMapping(unittest.TestCase):
    """Tests for the canonical finger-count → class mapping."""

    def test_01_one_finger_maps_to_cardboard(self):
        self.assertEqual(FINGER_COUNT_TO_CLASS_ID[1], 0)
        self.assertEqual(GESTURE_CLASS_NAMES[0], "cardboard")

    def test_02_two_fingers_maps_to_glass(self):
        self.assertEqual(FINGER_COUNT_TO_CLASS_ID[2], 1)
        self.assertEqual(GESTURE_CLASS_NAMES[1], "glass")

    def test_03_three_fingers_maps_to_metal(self):
        self.assertEqual(FINGER_COUNT_TO_CLASS_ID[3], 2)
        self.assertEqual(GESTURE_CLASS_NAMES[2], "metal")

    def test_04_four_fingers_maps_to_paper(self):
        self.assertEqual(FINGER_COUNT_TO_CLASS_ID[4], 3)
        self.assertEqual(GESTURE_CLASS_NAMES[3], "paper")

    def test_05_five_fingers_maps_to_plastic(self):
        self.assertEqual(FINGER_COUNT_TO_CLASS_ID[5], 4)
        self.assertEqual(GESTURE_CLASS_NAMES[4], "plastic")

    def test_06_six_fingers_maps_to_trash(self):
        self.assertEqual(FINGER_COUNT_TO_CLASS_ID[6], 5)
        self.assertEqual(GESTURE_CLASS_NAMES[5], "trash")

    def test_10_class_mapping_is_complete_six_classes(self):
        """Exactly 6 valid finger counts (1-6) must be in the mapping."""
        self.assertEqual(len(FINGER_COUNT_TO_CLASS_ID), 6)
        for finger_count in range(1, 7):
            self.assertIn(finger_count, FINGER_COUNT_TO_CLASS_ID)

    def test_10b_class_names_length_is_six(self):
        self.assertEqual(len(GESTURE_CLASS_NAMES), 6)

    def test_10c_class_names_match_canonical_order(self):
        expected = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        self.assertEqual(GESTURE_CLASS_NAMES, expected)

    def test_10d_unmapped_finger_count_not_in_dict(self):
        """0 and 7 should not be in the mapping."""
        self.assertNotIn(0, FINGER_COUNT_TO_CLASS_ID)
        self.assertNotIn(7, FINGER_COUNT_TO_CLASS_ID)


# ---------------------------------------------------------------------------
# B. GestureDetector.process_finger_count() Tests
# ---------------------------------------------------------------------------

class TestGestureDetectorDirectAPI(unittest.TestCase):
    """Tests for the mock-free direct API (no MediaPipe needed)."""

    def setUp(self):
        # Use a no-op mock so MediaPipe is not imported
        mock_hands = MagicMock()
        self.detector = GestureDetector(_mp_hands_mock=mock_hands)

    def test_01_finger_1_returns_valid_cardboard(self):
        result = self.detector.process_finger_count(1)
        self.assertTrue(result["valid"])
        self.assertEqual(result["class_id"], 0)
        self.assertEqual(result["class_name"], "cardboard")
        self.assertEqual(result["finger_count"], 1)

    def test_02_finger_2_returns_valid_glass(self):
        result = self.detector.process_finger_count(2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["class_id"], 1)
        self.assertEqual(result["class_name"], "glass")

    def test_03_finger_3_returns_valid_metal(self):
        result = self.detector.process_finger_count(3)
        self.assertTrue(result["valid"])
        self.assertEqual(result["class_id"], 2)
        self.assertEqual(result["class_name"], "metal")

    def test_04_finger_4_returns_valid_paper(self):
        result = self.detector.process_finger_count(4)
        self.assertTrue(result["valid"])
        self.assertEqual(result["class_id"], 3)
        self.assertEqual(result["class_name"], "paper")

    def test_05_finger_5_returns_valid_plastic(self):
        result = self.detector.process_finger_count(5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["class_id"], 4)
        self.assertEqual(result["class_name"], "plastic")

    def test_06_finger_6_returns_valid_trash(self):
        result = self.detector.process_finger_count(6)
        self.assertTrue(result["valid"])
        self.assertEqual(result["class_id"], 5)
        self.assertEqual(result["class_name"], "trash")

    def test_07_no_gesture_zero_is_invalid(self):
        """Finger count 0 is not mapped → invalid result."""
        result = self.detector.process_finger_count(0)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["class_id"])

    def test_08_unsupported_finger_count_7_is_invalid(self):
        result = self.detector.process_finger_count(7)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["class_id"])

    def test_09_non_integer_finger_count_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.detector.process_finger_count(2.5)

    def test_18_deterministic_repeated_calls(self):
        """Same finger count → identical result every time."""
        results = [self.detector.process_finger_count(3) for _ in range(10)]
        for r in results[1:]:
            self.assertEqual(r["class_id"], results[0]["class_id"])
            self.assertEqual(r["class_name"], results[0]["class_name"])
            self.assertEqual(r["valid"], results[0]["valid"])


# ---------------------------------------------------------------------------
# C. _DebounceBuffer Tests
# ---------------------------------------------------------------------------

class TestDebounceBuffer(unittest.TestCase):
    """Tests for the temporal debounce logic."""

    def test_single_reading_is_unstable(self):
        """One reading cannot satisfy the stability window."""
        buf = _DebounceBuffer(stable_duration_s=1.0)
        buf.push(3, timestamp=0.0)
        self.assertIsNone(buf.stable_count())

    def test_consistent_readings_over_duration_are_stable(self):
        """Multiple identical readings spanning >= stable_duration_s → stable."""
        buf = _DebounceBuffer(stable_duration_s=1.0)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            buf.push(3, timestamp=t)
        self.assertEqual(buf.stable_count(), 3)

    def test_inconsistent_readings_are_unstable(self):
        """Mixed finger counts → not stable even over a full window."""
        buf = _DebounceBuffer(stable_duration_s=1.0)
        buf.push(2, timestamp=0.0)
        buf.push(3, timestamp=0.3)
        buf.push(2, timestamp=0.6)
        buf.push(3, timestamp=1.0)
        self.assertIsNone(buf.stable_count())

    def test_reset_clears_buffer(self):
        buf = _DebounceBuffer(stable_duration_s=1.0)
        for t in [0.0, 0.5, 1.0]:
            buf.push(4, timestamp=t)
        buf.reset()
        self.assertIsNone(buf.stable_count())

    def test_window_shorter_than_duration_is_unstable(self):
        """Window spans only 0.4 s when 1.0 s required → unstable."""
        buf = _DebounceBuffer(stable_duration_s=1.0)
        buf.push(5, timestamp=0.0)
        buf.push(5, timestamp=0.4)
        self.assertIsNone(buf.stable_count())


# ---------------------------------------------------------------------------
# D. GestureSupervisor — ACCEPT path (safety rule)
# ---------------------------------------------------------------------------

class TestSupervisorAcceptPath(unittest.TestCase):
    """Safety rule: ACCEPT predictions must never be overridden by a gesture."""

    def setUp(self):
        self.supervisor = GestureSupervisor()
        self.clf_plastic = _make_classifier_result(class_id=4, confidence=0.94)
        self.rel_accept = _make_reliability_result(DECISION_ACCEPT, confidence=0.94)

    def test_11_accept_without_gesture_is_classifier(self):
        """ACCEPT + no gesture → source = classifier, no override."""
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_accept, gesture_result=None
        )
        self.assertEqual(result["state"], STATE_AUTO)
        self.assertEqual(result["decision_source"], SOURCE_CLASSIFIER)
        self.assertFalse(result["override_occurred"])
        self.assertEqual(result["final_class_id"], 4)
        self.assertEqual(result["final_class_name"], "plastic")

    def test_11b_accept_with_gesture_gesture_is_ignored(self):
        """ACCEPT + conflicting gesture (metal) → classifier wins."""
        gesture = _make_gesture_result(3)  # 3 fingers → metal
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_accept, gesture_result=gesture
        )
        self.assertEqual(result["state"], STATE_AUTO)
        self.assertEqual(result["decision_source"], SOURCE_CLASSIFIER)
        self.assertFalse(result["override_occurred"])
        self.assertEqual(result["final_class_id"], 4, "plastic must win")

    def test_14_decision_source_is_classifier_string(self):
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_accept
        )
        self.assertEqual(result["decision_source"], "classifier")

    def test_override_flag_is_false_on_accept(self):
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_accept
        )
        self.assertFalse(result["override_occurred"])

    def test_classifier_fields_preserved_on_accept(self):
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_accept
        )
        self.assertEqual(result["classifier_class_id"], 4)
        self.assertEqual(result["classifier_class_name"], "plastic")
        self.assertEqual(result["reliability_decision"], DECISION_ACCEPT)


# ---------------------------------------------------------------------------
# E. GestureSupervisor — UNCERTAIN + valid gesture (override)
# ---------------------------------------------------------------------------

class TestSupervisorOverridePath(unittest.TestCase):
    """UNCERTAIN + valid gesture → MANUAL_OVERRIDE."""

    def setUp(self):
        self.supervisor = GestureSupervisor()
        self.clf_plastic = _make_classifier_result(class_id=4, confidence=0.52)
        self.rel_uncertain = _make_reliability_result(DECISION_UNCERTAIN, confidence=0.52)

    def test_12_uncertain_with_gesture_overrides(self):
        """UNCERTAIN + metal gesture → final = metal, source = human_gesture."""
        gesture = _make_gesture_result(3)  # 3 fingers → metal
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_uncertain, gesture_result=gesture
        )
        self.assertEqual(result["state"], STATE_MANUAL_OVERRIDE)
        self.assertEqual(result["decision_source"], SOURCE_GESTURE)
        self.assertTrue(result["override_occurred"])
        self.assertEqual(result["final_class_id"], 2)
        self.assertEqual(result["final_class_name"], "metal")

    def test_15_decision_source_is_human_gesture_string(self):
        gesture = _make_gesture_result(2)
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_uncertain, gesture_result=gesture
        )
        self.assertEqual(result["decision_source"], "human_gesture")

    def test_17_override_flag_is_true_on_override(self):
        gesture = _make_gesture_result(1)
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_uncertain, gesture_result=gesture
        )
        self.assertTrue(result["override_occurred"])

    def test_all_six_gestures_produce_correct_override(self):
        """All finger counts 1-6 produce the correct override class."""
        expected = [(1, 0, "cardboard"), (2, 1, "glass"), (3, 2, "metal"),
                    (4, 3, "paper"), (5, 4, "plastic"), (6, 5, "trash")]
        for fingers, exp_id, exp_name in expected:
            gesture = _make_gesture_result(fingers)
            result = self.supervisor.decide(
                self.clf_plastic, self.rel_uncertain, gesture_result=gesture
            )
            self.assertEqual(result["final_class_id"], exp_id,
                             f"fingers={fingers}: expected class_id={exp_id}")
            self.assertEqual(result["final_class_name"], exp_name,
                             f"fingers={fingers}: expected class_name={exp_name}")

    def test_classifier_fields_preserved_on_override(self):
        gesture = _make_gesture_result(2)
        result = self.supervisor.decide(
            self.clf_plastic, self.rel_uncertain, gesture_result=gesture
        )
        # Original classifier info must always be preserved
        self.assertEqual(result["classifier_class_id"], 4)
        self.assertEqual(result["classifier_class_name"], "plastic")
        self.assertEqual(result["reliability_decision"], DECISION_UNCERTAIN)


# ---------------------------------------------------------------------------
# F. GestureSupervisor — UNCERTAIN + no gesture (unresolved)
# ---------------------------------------------------------------------------

class TestSupervisorUnresolvedPath(unittest.TestCase):
    """UNCERTAIN + no valid gesture → UNRESOLVED."""

    def setUp(self):
        self.supervisor = GestureSupervisor()
        self.clf = _make_classifier_result(class_id=4, confidence=0.52)
        self.rel_uncertain = _make_reliability_result(DECISION_UNCERTAIN, confidence=0.52)

    def test_13_uncertain_no_gesture_is_unresolved(self):
        """UNCERTAIN + gesture=None → UNRESOLVED."""
        result = self.supervisor.decide(self.clf, self.rel_uncertain, gesture_result=None)
        self.assertEqual(result["state"], STATE_UNRESOLVED)
        self.assertEqual(result["decision_source"], SOURCE_UNRESOLVED)
        self.assertFalse(result["override_occurred"])
        self.assertIsNone(result["final_class_id"])
        self.assertIsNone(result["final_class_name"])

    def test_13b_uncertain_invalid_gesture_is_unresolved(self):
        """UNCERTAIN + invalid gesture (finger count 0) → UNRESOLVED."""
        gesture = _make_gesture_result(0)  # 0 is not mapped
        self.assertFalse(gesture["valid"])
        result = self.supervisor.decide(self.clf, self.rel_uncertain, gesture_result=gesture)
        self.assertEqual(result["state"], STATE_UNRESOLVED)
        self.assertFalse(result["override_occurred"])

    def test_16_decision_source_is_unresolved_string(self):
        result = self.supervisor.decide(self.clf, self.rel_uncertain, gesture_result=None)
        self.assertEqual(result["decision_source"], "unresolved")

    def test_17b_override_flag_false_on_unresolved(self):
        result = self.supervisor.decide(self.clf, self.rel_uncertain, gesture_result=None)
        self.assertFalse(result["override_occurred"])


# ---------------------------------------------------------------------------
# G. GestureSupervisor — edge cases and malformed input
# ---------------------------------------------------------------------------

class TestSupervisorEdgeCases(unittest.TestCase):

    def setUp(self):
        self.supervisor = GestureSupervisor()

    def test_19_missing_classifier_class_id_raises_or_errors(self):
        """Malformed classifier_result (missing class_id) → ERROR state."""
        bad_clf = {"class_name": "plastic", "confidence": 0.9, "probabilities": [0]*6}
        rel = _make_reliability_result(DECISION_ACCEPT)
        result = self.supervisor.decide(bad_clf, rel)
        self.assertEqual(result["state"], STATE_ERROR)

    def test_19b_out_of_range_class_id_errors(self):
        """class_id out of [0, 5] → ERROR state."""
        bad_clf = {
            "class_id": 99, "class_name": "unknown",
            "confidence": 0.9, "probabilities": [0]*6
        }
        rel = _make_reliability_result(DECISION_ACCEPT)
        result = self.supervisor.decide(bad_clf, rel)
        self.assertEqual(result["state"], STATE_ERROR)

    def test_19c_invalid_reliability_decision_errors(self):
        """Invalid reliability decision string → ERROR state."""
        clf = _make_classifier_result(0)
        bad_rel = {"decision": "MAYBE"}
        result = self.supervisor.decide(clf, bad_rel)
        self.assertEqual(result["state"], STATE_ERROR)

    def test_gesture_with_none_class_id_treated_as_invalid(self):
        """Gesture dict where class_id is None → treated as invalid gesture."""
        clf = _make_classifier_result(3, confidence=0.5)
        rel = _make_reliability_result(DECISION_UNCERTAIN, confidence=0.5)
        bad_gesture = {
            "valid": True,  # claims valid but has no class_id
            "finger_count": 3,
            "class_id": None,
            "class_name": None,
            "confidence": 1.0,
            "hands_detected": 1,
            "reason": "test",
        }
        result = self.supervisor.decide(clf, rel, gesture_result=bad_gesture)
        # No usable class_id → must be UNRESOLVED
        self.assertEqual(result["state"], STATE_UNRESOLVED)

    def test_apply_gesture_override_convenience_method(self):
        """apply_gesture_override() wraps decide() correctly."""
        clf = _make_classifier_result(4, confidence=0.5)
        rel = _make_reliability_result(DECISION_UNCERTAIN, confidence=0.5)
        result = self.supervisor.apply_gesture_override(clf, rel, finger_count=2)
        self.assertEqual(result["state"], STATE_MANUAL_OVERRIDE)
        self.assertEqual(result["final_class_id"], 1)
        self.assertEqual(result["final_class_name"], "glass")

    def test_apply_gesture_override_invalid_count(self):
        """Unmapped finger count → UNRESOLVED."""
        clf = _make_classifier_result(4, confidence=0.5)
        rel = _make_reliability_result(DECISION_UNCERTAIN, confidence=0.5)
        result = self.supervisor.apply_gesture_override(clf, rel, finger_count=99)
        self.assertEqual(result["state"], STATE_UNRESOLVED)


# ---------------------------------------------------------------------------
# H. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def setUp(self):
        self.supervisor = GestureSupervisor()

    def test_18_supervisor_decide_is_deterministic(self):
        """Same inputs → identical result across 10 repeated calls."""
        clf = _make_classifier_result(2, confidence=0.55)
        rel = _make_reliability_result(DECISION_UNCERTAIN, confidence=0.55)
        gesture = _make_gesture_result(4)
        results = [self.supervisor.decide(clf, rel, gesture) for _ in range(10)]
        for r in results[1:]:
            self.assertEqual(r["state"], results[0]["state"])
            self.assertEqual(r["final_class_id"], results[0]["final_class_id"])
            self.assertEqual(r["decision_source"], results[0]["decision_source"])
            self.assertEqual(r["override_occurred"], results[0]["override_occurred"])


# ---------------------------------------------------------------------------
# I. MediaPipe unavailability handling
# ---------------------------------------------------------------------------

class TestMediaPipeUnavailable(unittest.TestCase):

    def test_no_mediapipe_raises_not_available_error(self):
        """
        Attempting to create GestureDetector without MediaPipe and without a
        mock must raise MediaPipeNotAvailableError.

        Since MediaPipe is not installed in the project WSL environment
        (it would downgrade NumPy and break TensorFlow), this test verifies
        the error path is exercised correctly.
        """
        import unittest.mock as mock
        import builtins
        real_import = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name == "mediapipe":
                raise ImportError("mediapipe not installed (mocked)")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mocked_import):
            with self.assertRaises(MediaPipeNotAvailableError):
                GestureDetector()  # no mock, no mediapipe


# ---------------------------------------------------------------------------
# J. Regression: import Phase 1 + Phase 2 modules without error
# ---------------------------------------------------------------------------

class TestPhase1Phase2Regression(unittest.TestCase):
    """Smoke-import tests ensuring Phase 3 additions do not break prior phases."""

    def test_classifier_module_imports(self):
        from src.realtime.realtime_classifier import WasteClassifier, DEFAULT_CLASSES
        self.assertEqual(len(DEFAULT_CLASSES), 6)

    def test_confidence_gate_imports(self):
        from src.reliability.confidence_gate import ConfidenceGate, DECISION_ACCEPT, DECISION_UNCERTAIN
        gate = ConfidenceGate()
        self.assertAlmostEqual(gate.confidence_threshold, 0.70)

    def test_ood_filter_imports(self):
        from src.reliability.ood_filter import OodFilter
        f = OodFilter()
        self.assertAlmostEqual(f.confidence_threshold, 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
