"""
Deterministic diverter gate for the digital twin sorting simulator.

The sorting gate maps a received class decision to one of six bins and
routes the waste object accordingly. It implements a simple three-state
machine:

    IDLE → DIVERTING → RESETTING → IDLE

DESIGN PRINCIPLES
-----------------
- Deterministic: given the same class_id and dt sequence, the output
  is always identical.
- No randomness.
- The gate never reinterprets a decision — it routes based on the
  class_id it receives.
- No Pygame, TensorFlow, or MediaPipe dependency.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional

from src.simulator.waste_object import CLASS_NAMES, NUM_CLASSES, WasteObject


# ---------------------------------------------------------------------------
# Bin definitions
# ---------------------------------------------------------------------------

# Bin index is identical to class_id (0–5)
BIN_NAMES: List[str] = list(CLASS_NAMES)
NUM_BINS: int = NUM_CLASSES

# Canonical routing: class_id → bin_id (identity mapping)
CLASS_TO_BIN: Dict[int, int] = {i: i for i in range(NUM_CLASSES)}


# ---------------------------------------------------------------------------
# Gate state
# ---------------------------------------------------------------------------

class GateState(Enum):
    IDLE = auto()        # waiting for a decision
    DIVERTING = auto()   # actively routing an object
    RESETTING = auto()   # returning to neutral after divert


# ---------------------------------------------------------------------------
# Bin counter
# ---------------------------------------------------------------------------

class BinCounter:
    """Tracks objects received by one bin."""

    def __init__(self, class_id: int) -> None:
        self.class_id = class_id
        self.class_name = CLASS_NAMES[class_id]
        self.received: int = 0

    def increment(self) -> None:
        self.received += 1

    def reset(self) -> None:
        self.received = 0

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "received": self.received,
        }


# ---------------------------------------------------------------------------
# SortingGate
# ---------------------------------------------------------------------------

class SortingGate:
    """
    Diverter gate that routes waste objects to one of six bins.

    Parameters
    ----------
    divert_duration_s : float
        How long (in simulation seconds) the gate stays in DIVERTING state.
    reset_duration_s : float
        How long (in simulation seconds) the gate takes to reset to IDLE.
    gate_x : float
        X-position of the gate (for renderer use; not used in physics).
    gate_y : float
        Y-position of the gate (for renderer use).
    """

    def __init__(
        self,
        divert_duration_s: float = 0.5,
        reset_duration_s: float = 0.3,
        gate_x: float = 600.0,
        gate_y: float = 300.0,
    ) -> None:
        if divert_duration_s <= 0:
            raise ValueError(f"divert_duration_s must be positive, got {divert_duration_s}.")
        if reset_duration_s <= 0:
            raise ValueError(f"reset_duration_s must be positive, got {reset_duration_s}.")

        self.divert_duration_s = float(divert_duration_s)
        self.reset_duration_s = float(reset_duration_s)
        self.gate_x = float(gate_x)
        self.gate_y = float(gate_y)

        # State machine
        self.state: GateState = GateState.IDLE
        self._pending_class_id: Optional[int] = None
        self._state_timer: float = 0.0  # time remaining in current state

        # Bin counters — one per class
        self.bins: Dict[int, BinCounter] = {
            i: BinCounter(i) for i in range(NUM_BINS)
        }

        # Objects sorted this session
        self.sorted_objects: List[WasteObject] = []
        self.total_sorted: int = 0
        self.total_fallen: int = 0

    # ------------------------------------------------------------------
    # Decision reception
    # ------------------------------------------------------------------

    def receive_decision(self, class_id: int) -> bool:
        """
        Provides the gate with a routing decision.

        The gate accepts the decision only if it is currently IDLE.
        If the gate is already DIVERTING or RESETTING, the decision is
        rejected (returns False) — the simulator should queue it or wait.

        Parameters
        ----------
        class_id : int  (0–5)

        Returns
        -------
        bool — True if the decision was accepted.
        """
        if not isinstance(class_id, int) or class_id not in CLASS_TO_BIN:
            raise ValueError(
                f"class_id must be int in [0, {NUM_CLASSES - 1}], got {class_id!r}."
            )
        if self.state == GateState.IDLE:
            self._pending_class_id = class_id
            self.state = GateState.DIVERTING
            self._state_timer = self.divert_duration_s
            return True
        return False  # gate busy

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def step(self, dt: float, objects_at_gate: List[WasteObject]) -> List[WasteObject]:
        """
        Advances gate state by one fixed timestep.

        Parameters
        ----------
        dt : float
            Timestep in seconds.
        objects_at_gate : list of WasteObject
            Objects that have just reached the gate trigger position.
            The gate will attempt to sort them if DIVERTING.

        Returns
        -------
        sorted_now : list of WasteObject
            Objects that were successfully sorted in this step.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")

        sorted_now: List[WasteObject] = []

        # --- Route objects arriving at the gate ---
        if self.state == GateState.DIVERTING and self._pending_class_id is not None:
            bin_id = CLASS_TO_BIN[self._pending_class_id]
            for obj in objects_at_gate:
                if obj.is_active:
                    obj.mark_sorted(bin_id)
                    self.bins[bin_id].increment()
                    self.sorted_objects.append(obj)
                    self.total_sorted += 1
                    sorted_now.append(obj)

        # --- Advance state timer ---
        if self.state in (GateState.DIVERTING, GateState.RESETTING):
            self._state_timer -= dt
            if self._state_timer <= 0:
                if self.state == GateState.DIVERTING:
                    self.state = GateState.RESETTING
                    self._state_timer = self.reset_duration_s
                else:  # RESETTING → IDLE
                    self.state = GateState.IDLE
                    self._pending_class_id = None
                    self._state_timer = 0.0

        return sorted_now

    def register_fallen(self, obj: WasteObject) -> None:
        """Records that an object fell off the conveyor unsorted."""
        self.total_fallen += 1

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Resets gate to IDLE and clears all counters."""
        self.state = GateState.IDLE
        self._pending_class_id = None
        self._state_timer = 0.0
        self.sorted_objects.clear()
        self.total_sorted = 0
        self.total_fallen = 0
        for bc in self.bins.values():
            bc.reset()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_bin_counts(self) -> Dict[str, int]:
        """Returns {class_name: received_count} for all bins."""
        return {bc.class_name: bc.received for bc in self.bins.values()}

    def get_bin_state(self) -> List[dict]:
        """Returns a list of bin counter dicts for serialisation."""
        return [bc.to_dict() for bc in self.bins.values()]

    @property
    def is_idle(self) -> bool:
        return self.state == GateState.IDLE

    @property
    def pending_class_id(self) -> Optional[int]:
        return self._pending_class_id
