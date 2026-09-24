"""
Deterministic conveyor belt for the digital twin sorting simulator.

The conveyor is responsible for:
- Moving all active waste objects at a configurable speed
- Detecting when an object reaches the gate trigger zone
- Detecting when an object has passed the end of the belt (fallen off)

DESIGN PRINCIPLES
-----------------
- Fixed-timestep physics: state advances only via step(dt) calls.
- No wall-clock timing dependency.
- No randomness.
- Pause/resume and reset are supported.
- No Pygame or TensorFlow dependency.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple

from src.simulator.waste_object import WasteObject


# ---------------------------------------------------------------------------
# Conveyor state
# ---------------------------------------------------------------------------

class ConveyorState(Enum):
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


# ---------------------------------------------------------------------------
# Conveyor
# ---------------------------------------------------------------------------

class Conveyor:
    """
    Moves waste objects along a 1-D track from left to right.

    Parameters
    ----------
    belt_start_x : float
        X-coordinate of the left end of the belt (object spawn zone).
    belt_end_x : float
        X-coordinate of the right end (fall-off threshold).
    gate_trigger_x : float
        X-coordinate at which the sorting gate is activated.
    speed : float
        Default object velocity (units/second). Must be positive.
    """

    def __init__(
        self,
        belt_start_x: float = 50.0,
        belt_end_x: float = 950.0,
        gate_trigger_x: float = 600.0,
        speed: float = 150.0,
    ) -> None:
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}.")
        if belt_end_x <= belt_start_x:
            raise ValueError("belt_end_x must be greater than belt_start_x.")
        if not (belt_start_x <= gate_trigger_x <= belt_end_x):
            raise ValueError("gate_trigger_x must be within [belt_start_x, belt_end_x].")

        self.belt_start_x = float(belt_start_x)
        self.belt_end_x = float(belt_end_x)
        self.gate_trigger_x = float(gate_trigger_x)
        self.speed = float(speed)

        self.state: ConveyorState = ConveyorState.RUNNING
        self._objects: List[WasteObject] = []

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(self, obj: WasteObject) -> None:
        """Adds a waste object to the conveyor's tracking list."""
        self._objects.append(obj)

    def remove_object(self, obj: WasteObject) -> None:
        """Removes a waste object from tracking (after sorting or falling)."""
        if obj in self._objects:
            self._objects.remove(obj)

    @property
    def active_objects(self) -> List[WasteObject]:
        """Returns all currently tracked objects (active or completed)."""
        return list(self._objects)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def step(self, dt: float) -> Tuple[List[WasteObject], List[WasteObject]]:
        """
        Advances the conveyor one fixed timestep.

        Parameters
        ----------
        dt : float
            Timestep in seconds. Must be positive.

        Returns
        -------
        at_gate : list of WasteObject
            Objects that have reached or crossed gate_trigger_x this step
            and are still MOVING (gate has not acted on them yet).
        fallen : list of WasteObject
            Objects that have reached or crossed belt_end_x and were still
            MOVING (they are marked FALLEN by this method).
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")

        at_gate: List[WasteObject] = []
        fallen: List[WasteObject] = []

        if self.state != ConveyorState.RUNNING:
            return at_gate, fallen

        for obj in list(self._objects):
            if not obj.is_active:
                continue

            prev_x = obj.x
            obj.step(dt)

            # Check fall-off
            if obj.x >= self.belt_end_x:
                obj.mark_fallen()
                fallen.append(obj)
                continue

            # Check gate trigger (crossed this step or was already past)
            if prev_x < self.gate_trigger_x <= obj.x:
                at_gate.append(obj)

        return at_gate, fallen

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self.state = ConveyorState.PAUSED

    def resume(self) -> None:
        self.state = ConveyorState.RUNNING

    def stop(self) -> None:
        self.state = ConveyorState.STOPPED

    def reset(self) -> None:
        """Clears all objects and returns to RUNNING state."""
        self._objects.clear()
        self.state = ConveyorState.RUNNING

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self.state == ConveyorState.RUNNING

    def object_count(self) -> int:
        return len(self._objects)
