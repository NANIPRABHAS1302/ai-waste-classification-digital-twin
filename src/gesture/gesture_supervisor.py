"""
Gesture supervisor for human-in-the-loop waste classification.

ARCHITECTURE ROLE
-----------------
The supervisor sits between the reliability gate and the downstream
decision consumer (UDP sender / digital twin). It has one responsibility:

    Given a classifier prediction, a reliability decision, and an
    optional gesture result — produce a single authoritative final decision
    that clearly identifies the decision source and whether a human override
    occurred.

SAFETY RULE
-----------
A gesture must NEVER silently replace a high-confidence classifier prediction.

    classifier: ACCEPT → supervisor ignores gesture, forwards classifier result
    classifier: UNCERTAIN + valid gesture → supervisor applies human override
    classifier: UNCERTAIN + no valid gesture → result remains UNRESOLVED

STATE MACHINE
-------------
    AUTO            — classifier produced ACCEPT; gesture not consulted
    MANUAL_OVERRIDE — classifier was UNCERTAIN; gesture provided override
    UNRESOLVED      — classifier was UNCERTAIN; no valid gesture available
    ERROR           — unexpected input; safe fallback applied

DECISION SOURCE STRINGS
-----------------------
    "classifier"     — automatic, gate accepted the classifier prediction
    "human_gesture"  — human override via finger gesture
    "unresolved"     — UNCERTAIN with no valid gesture; requires further action

All source strings and state names are exposed as module-level constants.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from src.reliability.confidence_gate import DECISION_ACCEPT, DECISION_UNCERTAIN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decision sources
SOURCE_CLASSIFIER = "classifier"
SOURCE_GESTURE = "human_gesture"
SOURCE_UNRESOLVED = "unresolved"

# Supervisor states
STATE_AUTO = "AUTO"
STATE_MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
STATE_UNRESOLVED = "UNRESOLVED"
STATE_ERROR = "ERROR"

# Canonical 6-class mapping — must match GESTURE_CLASS_NAMES in gesture_detector.py
DEFAULT_CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _make_supervisor_result(
    *,
    state: str,
    final_class_id: Optional[int],
    final_class_name: Optional[str],
    decision_source: str,
    classifier_class_id: Optional[int],
    classifier_class_name: Optional[str],
    reliability_decision: str,
    gesture_class_id: Optional[int],
    gesture_class_name: Optional[str],
    override_occurred: bool,
    reason: str,
) -> Dict:
    return {
        "state": state,
        "final_class_id": final_class_id,
        "final_class_name": final_class_name,
        "decision_source": decision_source,
        "classifier_class_id": classifier_class_id,
        "classifier_class_name": classifier_class_name,
        "reliability_decision": reliability_decision,
        "gesture_class_id": gesture_class_id,
        "gesture_class_name": gesture_class_name,
        "override_occurred": override_occurred,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# GestureSupervisor
# ---------------------------------------------------------------------------

class GestureSupervisor:
    """
    Decides whether a human gesture should override the classifier prediction.

    Does NOT perform classification or gesture detection itself.
    Consumes:
      - classifier prediction dict (from WasteClassifier.predict())
      - reliability gate result dict (from ConfidenceGate.evaluate())
      - optional gesture result dict (from GestureDetector.process_finger_count()
        or GestureDetector.process_frame())

    Parameters
    ----------
    class_names : sequence of str, optional
        Canonical class names. Defaults to the 6-class mapping.
    """

    def __init__(self, class_names: Optional[Sequence[str]] = None) -> None:
        self.class_names = (
            list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        )

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    def _validate_classifier_result(self, classifier_result: Dict) -> None:
        required = {"class_id", "class_name", "confidence", "probabilities"}
        missing = required - set(classifier_result.keys())
        if missing:
            raise ValueError(
                f"classifier_result is missing required keys: {sorted(missing)}"
            )
        class_id = classifier_result["class_id"]
        if not isinstance(class_id, int) or not (0 <= class_id < len(self.class_names)):
            raise ValueError(
                f"classifier_result class_id={class_id!r} is out of range [0, {len(self.class_names)-1}]."
            )

    def _validate_reliability_result(self, reliability_result: Dict) -> None:
        required = {"decision"}
        missing = required - set(reliability_result.keys())
        if missing:
            raise ValueError(
                f"reliability_result is missing required keys: {sorted(missing)}"
            )
        decision = reliability_result["decision"]
        if decision not in (DECISION_ACCEPT, DECISION_UNCERTAIN):
            raise ValueError(
                f"reliability_result decision={decision!r} must be "
                f"'{DECISION_ACCEPT}' or '{DECISION_UNCERTAIN}'."
            )

    # ------------------------------------------------------------------
    # Core decision method
    # ------------------------------------------------------------------

    def decide(
        self,
        classifier_result: Dict,
        reliability_result: Dict,
        gesture_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Produces the authoritative final decision.

        Parameters
        ----------
        classifier_result : dict
            Output of WasteClassifier.predict(). Must contain:
            'class_id', 'class_name', 'confidence', 'probabilities'.
        reliability_result : dict
            Output of ConfidenceGate.evaluate(). Must contain 'decision'.
        gesture_result : dict or None
            Output of GestureDetector.process_finger_count() or
            GestureDetector.process_frame(). May be None or invalid.

        Returns
        -------
        dict with keys:
            state                : str (AUTO | MANUAL_OVERRIDE | UNRESOLVED | ERROR)
            final_class_id       : int or None
            final_class_name     : str or None
            decision_source      : str (classifier | human_gesture | unresolved)
            classifier_class_id  : int
            classifier_class_name: str
            reliability_decision : str (ACCEPT | UNCERTAIN)
            gesture_class_id     : int or None
            gesture_class_name   : str or None
            override_occurred    : bool
            reason               : str
        """
        # --- Validate inputs ---
        try:
            self._validate_classifier_result(classifier_result)
            self._validate_reliability_result(reliability_result)
        except (ValueError, TypeError, KeyError) as exc:
            return _make_supervisor_result(
                state=STATE_ERROR,
                final_class_id=None,
                final_class_name=None,
                decision_source=SOURCE_UNRESOLVED,
                classifier_class_id=classifier_result.get("class_id"),
                classifier_class_name=classifier_result.get("class_name"),
                reliability_decision=reliability_result.get("decision", "UNKNOWN"),
                gesture_class_id=None,
                gesture_class_name=None,
                override_occurred=False,
                reason=f"input_validation_error: {exc}",
            )

        classifier_class_id: int = classifier_result["class_id"]
        classifier_class_name: str = classifier_result["class_name"]
        reliability_decision: str = reliability_result["decision"]

        # Extract gesture fields (None-safe)
        gesture_class_id: Optional[int] = None
        gesture_class_name: Optional[str] = None
        gesture_valid = False

        if gesture_result is not None and isinstance(gesture_result, dict):
            gesture_valid = bool(gesture_result.get("valid", False))
            if gesture_valid:
                gesture_class_id = gesture_result.get("class_id")
                gesture_class_name = gesture_result.get("class_name")
                # Validate gesture class_id range
                if (
                    gesture_class_id is None
                    or not isinstance(gesture_class_id, int)
                    or not (0 <= gesture_class_id < len(self.class_names))
                ):
                    gesture_valid = False
                    gesture_class_id = None
                    gesture_class_name = None

        # ---------------------------------------------------------------
        # DECISION LOGIC
        # ---------------------------------------------------------------

        # CASE 1: Classifier was accepted → preserve it unconditionally
        if reliability_decision == DECISION_ACCEPT:
            return _make_supervisor_result(
                state=STATE_AUTO,
                final_class_id=classifier_class_id,
                final_class_name=classifier_class_name,
                decision_source=SOURCE_CLASSIFIER,
                classifier_class_id=classifier_class_id,
                classifier_class_name=classifier_class_name,
                reliability_decision=reliability_decision,
                gesture_class_id=gesture_class_id,
                gesture_class_name=gesture_class_name,
                override_occurred=False,
                reason="classifier_accepted",
            )

        # CASE 2: Classifier was uncertain + valid gesture → human override
        if reliability_decision == DECISION_UNCERTAIN and gesture_valid:
            return _make_supervisor_result(
                state=STATE_MANUAL_OVERRIDE,
                final_class_id=gesture_class_id,
                final_class_name=gesture_class_name,
                decision_source=SOURCE_GESTURE,
                classifier_class_id=classifier_class_id,
                classifier_class_name=classifier_class_name,
                reliability_decision=reliability_decision,
                gesture_class_id=gesture_class_id,
                gesture_class_name=gesture_class_name,
                override_occurred=True,
                reason="human_gesture_override",
            )

        # CASE 3: Classifier was uncertain + no valid gesture → unresolved
        return _make_supervisor_result(
            state=STATE_UNRESOLVED,
            final_class_id=None,
            final_class_name=None,
            decision_source=SOURCE_UNRESOLVED,
            classifier_class_id=classifier_class_id,
            classifier_class_name=classifier_class_name,
            reliability_decision=reliability_decision,
            gesture_class_id=None,
            gesture_class_name=None,
            override_occurred=False,
            reason="uncertain_no_gesture",
        )

    # ------------------------------------------------------------------
    # Convenience wrapper for plain gesture-only evaluation
    # ------------------------------------------------------------------

    def apply_gesture_override(
        self,
        classifier_result: Dict,
        reliability_result: Dict,
        finger_count: int,
    ) -> Dict:
        """
        Convenience method. Builds a minimal gesture_result from a raw
        finger count (1-6) and delegates to decide().

        Suitable for direct testing without importing GestureDetector.

        Parameters
        ----------
        classifier_result : dict
        reliability_result : dict
        finger_count : int (1–6)

        Returns
        -------
        dict — same structure as decide()
        """
        from src.gesture.gesture_detector import FINGER_COUNT_TO_CLASS_ID, GESTURE_CLASS_NAMES

        class_id = FINGER_COUNT_TO_CLASS_ID.get(finger_count)
        if class_id is not None:
            gesture_result = {
                "valid": True,
                "finger_count": finger_count,
                "class_id": class_id,
                "class_name": GESTURE_CLASS_NAMES[class_id],
                "confidence": 1.0,
                "hands_detected": 1,
                "reason": "direct_finger_count",
            }
        else:
            gesture_result = {
                "valid": False,
                "finger_count": finger_count,
                "class_id": None,
                "class_name": None,
                "confidence": 0.0,
                "hands_detected": 1,
                "reason": f"unmapped_finger_count_{finger_count}",
            }
        return self.decide(classifier_result, reliability_result, gesture_result)
