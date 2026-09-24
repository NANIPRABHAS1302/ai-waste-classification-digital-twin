"""
Gesture detector for human-in-the-loop waste classification override.

ARCHITECTURE ROLE
-----------------
This module is a standalone component that sits AFTER the reliability gate.
It is invoked only when the classifier produces an UNCERTAIN result and a
human operator is expected to provide a manual override via a hand gesture.

It does NOT:
  - perform waste classification
  - load or run the MobileNetV2 model
  - replace or augment the classifier in any way

MEDIAPIPE DEPENDENCY STATUS
----------------------------
MediaPipe is the intended runtime library for hand landmark detection.

In the verified project environment (WSL2, Ubuntu 24.04, Python 3.12.3,
TensorFlow 2.21.0, NumPy 2.5.3), MediaPipe 0.10.x cannot be installed
without downgrading NumPy (2.5.3 → 1.26.4) and protobuf (7.36.2 → 4.25.9),
which would break TensorFlow 2.21.0.

Therefore:
  - MediaPipe is imported lazily at runtime, not at module import time.
  - If MediaPipe is unavailable, GestureDetector raises
    MediaPipeNotAvailableError on initialisation.
  - All unit tests use an injected mock (see tests/test_gesture.py).
  - Real-time gesture detection requires a Windows environment with
    MediaPipe installed separately from the TF inference environment.

GESTURE MAPPING
---------------
The mapping follows the canonical 6-class order established by the project:

    Fingers shown  →  Class ID  →  Class Name
    1              →  0         →  cardboard
    2              →  1         →  glass
    3              →  2         →  metal
    4              →  3         →  paper
    5              →  4         →  plastic
    6              →  5         →  trash

DEBOUNCING
----------
Raw finger-count estimates from a single frame are often noisy. The detector
requires a stable gesture to persist for at minimum `stable_duration_s` seconds
over a sliding window before reporting a confirmed result. This prevents
accidental hand movements from triggering an unintended override.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants — canonical 6-class mapping (must match 02_train_models.py)
# ---------------------------------------------------------------------------

GESTURE_CLASS_NAMES: List[str] = [
    "cardboard",  # 0 — 1 finger
    "glass",      # 1 — 2 fingers
    "metal",      # 2 — 3 fingers
    "paper",      # 3 — 4 fingers
    "plastic",    # 4 — 5 fingers
    "trash",      # 5 — 6 fingers (open hand / all fingers)
]

# Map: fingers shown (1-6) → class ID (0-5)
FINGER_COUNT_TO_CLASS_ID: Dict[int, int] = {
    1: 0,  # cardboard
    2: 1,  # glass
    3: 2,  # metal
    4: 3,  # paper
    5: 4,  # plastic
    6: 5,  # trash
}

# Inverse map: class ID → fingers required
CLASS_ID_TO_FINGER_COUNT: Dict[int, int] = {v: k for k, v in FINGER_COUNT_TO_CLASS_ID.items()}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MediaPipeNotAvailableError(RuntimeError):
    """Raised when MediaPipe is not installed in the current environment."""


class GestureDetectionError(RuntimeError):
    """Raised when gesture detection fails due to an unrecoverable error."""


# ---------------------------------------------------------------------------
# Result dataclass (plain dict for Python 3.12 compatibility without dataclasses)
# ---------------------------------------------------------------------------

def _make_gesture_result(
    *,
    valid: bool,
    finger_count: Optional[int],
    class_id: Optional[int],
    class_name: Optional[str],
    confidence: float,
    hands_detected: int,
    reason: str,
) -> Dict:
    """Returns a standardised gesture result dictionary."""
    return {
        "valid": valid,
        "finger_count": finger_count,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "hands_detected": hands_detected,
        "reason": reason,
    }


INVALID_RESULT = _make_gesture_result(
    valid=False,
    finger_count=None,
    class_id=None,
    class_name=None,
    confidence=0.0,
    hands_detected=0,
    reason="no_result",
)


# ---------------------------------------------------------------------------
# Finger counting helper (pure Python, MediaPipe landmark independent)
# ---------------------------------------------------------------------------

def count_extended_fingers(hand_landmarks, handedness: str = "Right") -> int:
    """
    Counts the number of extended fingers from MediaPipe hand landmark data.

    Uses the landmark indices from MediaPipe's 21-point hand model:
      - Fingertip landmarks: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
      - MCP (knuckle) landmarks: 2 (thumb), 5 (index), 9 (middle), 13 (ring), 17 (pinky)

    Thumb extension: compared against the MCP x-coordinate (mirrored for left hand).
    Other fingers: tip y-coordinate < pip y-coordinate (tip is above pip when extended).

    Parameters
    ----------
    hand_landmarks : MediaPipe NormalizedLandmarkList
        Landmarks from mp.solutions.hands.Hands results.
    handedness : str
        "Right" or "Left" — affects thumb extension direction logic.

    Returns
    -------
    int : number of extended fingers in range [0, 5].
          Thumb counts as 1; maximum return value is 5.
          For "6 fingers" (trash), the supervisor maps open-hand (5) + wrist gesture
          OR the caller may pass finger_count=6 directly via mock in tests.
    """
    lm = hand_landmarks.landmark

    # Fingertip and pip (proximal interphalangeal) landmark indices
    # Index:  [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    count = 0

    # Thumb: compare tip x vs IP joint x (flipped for left hand)
    if handedness == "Right":
        if lm[tip_ids[0]].x < lm[pip_ids[0]].x:
            count += 1
    else:
        if lm[tip_ids[0]].x > lm[pip_ids[0]].x:
            count += 1

    # Remaining four fingers: tip y < pip y means finger is up
    for tip_id, pip_id in zip(tip_ids[1:], pip_ids[1:]):
        if lm[tip_id].y < lm[pip_id].y:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Debounce buffer
# ---------------------------------------------------------------------------

class _DebounceBuffer:
    """
    Sliding-window temporal debounce for raw finger counts.

    Requires the same finger count to appear consistently for at least
    `stable_duration_s` seconds before it is considered stable.
    """

    def __init__(self, stable_duration_s: float = 1.0) -> None:
        self.stable_duration_s = stable_duration_s
        self._window: deque = deque()  # deque of (timestamp, finger_count)

    def push(self, finger_count: int, timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.monotonic()
        self._window.append((now, finger_count))
        # Evict entries older than stable_duration_s
        cutoff = now - self.stable_duration_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def stable_count(self) -> Optional[int]:
        """
        Returns the finger count if ALL entries in the current window agree,
        and the window spans at least stable_duration_s seconds.
        Returns None if no stable count is available.
        """
        if len(self._window) < 2:
            return None
        counts = [c for _, c in self._window]
        if len(set(counts)) == 1:
            span = self._window[-1][0] - self._window[0][0]
            if span >= self.stable_duration_s:
                return counts[0]
        return None

    def reset(self) -> None:
        self._window.clear()


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    Hand-gesture detector using MediaPipe Hands.

    Provides a simple interface that maps a detected finger count to a
    waste class ID/name. Requires MediaPipe to be installed at runtime.

    If MediaPipe is not available, raises MediaPipeNotAvailableError on
    construction. This allows the calling application to degrade gracefully
    (e.g., skip gesture mode, display a warning) rather than crashing.

    Parameters
    ----------
    stable_duration_s : float
        Seconds the same finger count must be held before reporting.
    min_detection_confidence : float
        MediaPipe hand-detection confidence threshold (0.0–1.0).
    min_tracking_confidence : float
        MediaPipe hand-tracking confidence threshold (0.0–1.0).
    max_hands : int
        Maximum number of hands to detect (default 1).
    class_names : sequence of str, optional
        Class name labels. Defaults to canonical GESTURE_CLASS_NAMES.
    _mp_hands_mock : object, optional
        Inject a mock MediaPipe Hands solution for unit testing.
        When provided, MediaPipe is NOT imported.
    """

    def __init__(
        self,
        stable_duration_s: float = 1.0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_hands: int = 1,
        class_names: Optional[Sequence[str]] = None,
        _mp_hands_mock=None,
    ) -> None:
        self.class_names: List[str] = (
            list(class_names) if class_names is not None else list(GESTURE_CLASS_NAMES)
        )
        self.num_classes = len(self.class_names)
        self._debounce = _DebounceBuffer(stable_duration_s=stable_duration_s)

        if _mp_hands_mock is not None:
            # Unit-test injection path — MediaPipe is NOT imported
            self._hands = _mp_hands_mock
            self._using_mock = True
        else:
            # Real runtime path — try to import MediaPipe
            try:
                import mediapipe as mp  # type: ignore
                self._mp = mp
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._using_mock = False
            except ImportError as exc:
                raise MediaPipeNotAvailableError(
                    "MediaPipe is not installed in the current environment. "
                    "Install it separately (not in the TF/Keras venv to avoid "
                    "NumPy/protobuf conflicts) or use _mp_hands_mock for testing. "
                    f"Original error: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_rgb,
        timestamp: Optional[float] = None,
    ) -> Dict:
        """
        Processes one RGB frame and returns a gesture result dict.

        Parameters
        ----------
        frame_rgb : numpy.ndarray or mock-compatible object
            RGB image frame (HxWx3, uint8).
        timestamp : float, optional
            Monotonic timestamp. If None, uses time.monotonic().

        Returns
        -------
        dict with keys:
            valid        : bool — True if a stable, mappable gesture was detected
            finger_count : int or None
            class_id     : int or None (0–5)
            class_name   : str or None
            confidence   : float (0.0–1.0; landmark detection confidence)
            hands_detected : int (0 or 1)
            reason       : str — description of the result
        """
        if frame_rgb is None:
            return _make_gesture_result(
                valid=False, finger_count=None, class_id=None,
                class_name=None, confidence=0.0, hands_detected=0,
                reason="null_frame",
            )

        try:
            results = self._hands.process(frame_rgb)
        except Exception as exc:
            raise GestureDetectionError(
                f"MediaPipe Hands.process() failed: {exc}"
            ) from exc

        # No hands detected
        if not results.multi_hand_landmarks:
            self._debounce.reset()
            return _make_gesture_result(
                valid=False, finger_count=None, class_id=None,
                class_name=None, confidence=0.0, hands_detected=0,
                reason="no_hand_detected",
            )

        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        hands_detected = len(results.multi_hand_landmarks)

        # Determine handedness label
        handedness_label = "Right"
        if (
            results.multi_handedness
            and results.multi_handedness[0].classification
        ):
            handedness_label = results.multi_handedness[0].classification[0].label

        # Count fingers
        try:
            finger_count = count_extended_fingers(hand_landmarks, handedness_label)
        except Exception as exc:
            raise GestureDetectionError(
                f"Finger counting failed: {exc}"
            ) from exc

        # Push to debounce buffer
        self._debounce.push(finger_count, timestamp=timestamp)
        stable = self._debounce.stable_count()

        if stable is None:
            return _make_gesture_result(
                valid=False, finger_count=finger_count, class_id=None,
                class_name=None, confidence=0.0, hands_detected=hands_detected,
                reason="unstable_gesture",
            )

        # Map finger count to class
        class_id = FINGER_COUNT_TO_CLASS_ID.get(stable)
        if class_id is None:
            return _make_gesture_result(
                valid=False, finger_count=stable, class_id=None,
                class_name=None, confidence=0.0, hands_detected=hands_detected,
                reason=f"unmapped_finger_count_{stable}",
            )

        class_name = self.class_names[class_id]
        return _make_gesture_result(
            valid=True,
            finger_count=stable,
            class_id=class_id,
            class_name=class_name,
            confidence=1.0,  # binary: stable gesture = full confidence
            hands_detected=hands_detected,
            reason="stable_gesture_detected",
        )

    def process_finger_count(self, finger_count: int) -> Dict:
        """
        Maps a raw finger count directly to a gesture result, bypassing
        MediaPipe and debouncing. Intended for unit testing and direct
        injection when finger_count is already known.

        Parameters
        ----------
        finger_count : int

        Returns
        -------
        dict — gesture result
        """
        if not isinstance(finger_count, int):
            raise TypeError(
                f"finger_count must be int, got {type(finger_count).__name__}."
            )

        class_id = FINGER_COUNT_TO_CLASS_ID.get(finger_count)
        if class_id is None:
            return _make_gesture_result(
                valid=False, finger_count=finger_count, class_id=None,
                class_name=None, confidence=0.0, hands_detected=1,
                reason=f"unmapped_finger_count_{finger_count}",
            )

        return _make_gesture_result(
            valid=True,
            finger_count=finger_count,
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=1.0,
            hands_detected=1,
            reason="direct_finger_count",
        )

    def reset_debounce(self) -> None:
        """Clears the debounce buffer. Call when a new waste object arrives."""
        self._debounce.reset()

    def close(self) -> None:
        """Releases MediaPipe resources if not using a mock."""
        if not self._using_mock and hasattr(self._hands, "close"):
            self._hands.close()
