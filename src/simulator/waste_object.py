"""
Waste object for the digital twin sorting simulator.

Represents a single waste item moving along the conveyor belt.

RESPONSIBILITIES
----------------
- Store object identity (ID, class ID, class name)
- Track position and velocity
- Track lifecycle state (MOVING → SORTED | FALLEN)
- Provide deterministic position update given a fixed dt

This module has no dependencies on TensorFlow, MediaPipe, Pygame,
or the UDP communication layer.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical class mapping — must match all other project modules
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "cardboard",  # 0
    "glass",      # 1
    "metal",      # 2
    "paper",      # 3
    "plastic",    # 4
    "trash",      # 5
]

# Visual colours for each class (RGB) — used by renderer if Pygame available
CLASS_COLOURS = {
    0: (139, 90, 43),    # cardboard — brown
    1: (100, 180, 220),  # glass — light blue
    2: (180, 180, 180),  # metal — silver
    3: (220, 210, 160),  # paper — cream
    4: (60, 180, 75),    # plastic — green
    5: (128, 128, 128),  # trash — dark grey
}

NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Object state
# ---------------------------------------------------------------------------

class ObjectState(Enum):
    MOVING = auto()   # travelling along the conveyor
    SORTED = auto()   # successfully routed to a bin
    FALLEN = auto()   # reached end of conveyor without being sorted (error)


# ---------------------------------------------------------------------------
# WasteObject
# ---------------------------------------------------------------------------

class WasteObject:
    """
    Represents one waste item on the conveyor belt.

    Parameters
    ----------
    object_id : int
        Unique identifier for this object.
    class_id : int
        Class index (0–5).
    spawn_x : float
        Initial x-coordinate on the conveyor (pixels or simulation units).
    spawn_y : float
        Initial y-coordinate (fixed to conveyor height).
    velocity_x : float
        Horizontal velocity (units/second). Positive = moving right.
    sim_time : float
        Simulation time at spawn (seconds).
    """

    def __init__(
        self,
        object_id: int,
        class_id: int,
        spawn_x: float = 0.0,
        spawn_y: float = 300.0,
        velocity_x: float = 100.0,
        sim_time: float = 0.0,
    ) -> None:
        if not isinstance(class_id, int) or not (0 <= class_id < NUM_CLASSES):
            raise ValueError(
                f"class_id must be int in [0, {NUM_CLASSES - 1}], got {class_id!r}."
            )

        self.object_id: int = object_id
        self.class_id: int = class_id
        self.class_name: str = CLASS_NAMES[class_id]
        self.colour: tuple = CLASS_COLOURS[class_id]

        self.x: float = float(spawn_x)
        self.y: float = float(spawn_y)
        self.velocity_x: float = float(velocity_x)

        self.spawn_time: float = float(sim_time)
        self.state: ObjectState = ObjectState.MOVING
        self.sorted_to_bin: Optional[int] = None  # class_id of destination bin

    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """
        Advances object position by one fixed timestep.

        Parameters
        ----------
        dt : float
            Timestep in seconds. Must be positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if self.state == ObjectState.MOVING:
            self.x += self.velocity_x * dt

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_sorted(self, bin_class_id: int) -> None:
        """Mark this object as successfully sorted into a bin."""
        self.state = ObjectState.SORTED
        self.sorted_to_bin = bin_class_id

    def mark_fallen(self) -> None:
        """Mark this object as having fallen off the conveyor unsorted."""
        self.state = ObjectState.FALLEN

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.state == ObjectState.MOVING

    @property
    def is_completed(self) -> bool:
        return self.state in (ObjectState.SORTED, ObjectState.FALLEN)

    def __repr__(self) -> str:
        return (
            f"WasteObject(id={self.object_id}, class={self.class_name!r}, "
            f"x={self.x:.1f}, state={self.state.name})"
        )
