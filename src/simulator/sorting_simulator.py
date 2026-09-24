"""
SortingSimulator — top-level coordinator for the digital twin.

Coordinates:
  - Conveyor belt physics
  - Waste object lifecycle
  - Sorting gate state machine
  - Simulation clock
  - Decision intake (from UDP messages or direct API calls)

DESIGN PRINCIPLES
-----------------
- Fully headless: can operate without Pygame or any display.
- Deterministic: given same inputs and dt sequence → same final state.
- Clean API: reset(), spawn_object(), receive_decision(), step(), get_state().
- No TensorFlow, MediaPipe, or networking dependency.

UDP INTEGRATION NOTE
---------------------
This simulator can receive validated decision dicts from UdpReceiver.
It does NOT open sockets itself. The integration loop (Phase 7) will
feed decisions from UdpReceiver into receive_decision().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.simulator.conveyor import Conveyor, ConveyorState
from src.simulator.sorting_gate import CLASS_TO_BIN, GateState, SortingGate
from src.simulator.waste_object import (
    CLASS_NAMES,
    NUM_CLASSES,
    ObjectState,
    WasteObject,
)

# ---------------------------------------------------------------------------
# Simulator configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_CONVEYOR_SPEED: float = 150.0    # px/s
DEFAULT_GATE_X: float = 600.0
DEFAULT_CONVEYOR_Y: float = 300.0
DEFAULT_BELT_START: float = 50.0
DEFAULT_BELT_END: float = 950.0
DEFAULT_DIVERT_DURATION: float = 0.5
DEFAULT_RESET_DURATION: float = 0.3


class SortingSimulator:
    """
    Coordinates the full waste-sorting digital twin simulation.

    Parameters
    ----------
    conveyor_speed : float
        Object velocity (units/second). From configs/config.yaml.
    gate_x : float
        X-position of the sorting gate.
    belt_y : float
        Y-position of the conveyor belt.
    belt_start_x : float
        Left end of the belt.
    belt_end_x : float
        Right end of the belt (fall-off threshold).
    divert_duration_s : float
        Gate divert hold time (seconds).
    reset_duration_s : float
        Gate reset time (seconds).
    """

    def __init__(
        self,
        conveyor_speed: float = DEFAULT_CONVEYOR_SPEED,
        gate_x: float = DEFAULT_GATE_X,
        belt_y: float = DEFAULT_CONVEYOR_Y,
        belt_start_x: float = DEFAULT_BELT_START,
        belt_end_x: float = DEFAULT_BELT_END,
        divert_duration_s: float = DEFAULT_DIVERT_DURATION,
        reset_duration_s: float = DEFAULT_RESET_DURATION,
    ) -> None:
        self._conveyor_speed = conveyor_speed
        self._gate_x = gate_x
        self._belt_y = belt_y
        self._belt_start_x = belt_start_x
        self._belt_end_x = belt_end_x
        self._divert_duration_s = divert_duration_s
        self._reset_duration_s = reset_duration_s

        self._next_object_id: int = 0
        self._sim_time: float = 0.0

        # Decision queue: decisions received before an object arrives at the gate
        self._decision_queue: List[int] = []

        # Last received decision metadata (for state reporting)
        self._last_decision: Optional[Dict[str, Any]] = None

        self._init_components()

    def _init_components(self) -> None:
        """Creates fresh conveyor and gate instances."""
        self.conveyor = Conveyor(
            belt_start_x=self._belt_start_x,
            belt_end_x=self._belt_end_x,
            gate_trigger_x=self._gate_x,
            speed=self._conveyor_speed,
        )
        self.gate = SortingGate(
            divert_duration_s=self._divert_duration_s,
            reset_duration_s=self._reset_duration_s,
            gate_x=self._gate_x,
            gate_y=self._belt_y,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Resets the simulator to its initial state.

        Clears all objects, resets the gate and conveyor, zeroes the
        simulation clock and decision queue.
        """
        self._next_object_id = 0
        self._sim_time = 0.0
        self._decision_queue.clear()
        self._last_decision = None
        self._init_components()

    def spawn_object(self, class_id: int) -> WasteObject:
        """
        Spawns a new waste object at the start of the conveyor.

        Parameters
        ----------
        class_id : int (0–5)

        Returns
        -------
        WasteObject — the newly created object.

        Raises
        ------
        ValueError if class_id is out of range.
        """
        obj = WasteObject(
            object_id=self._next_object_id,
            class_id=class_id,
            spawn_x=self._belt_start_x,
            spawn_y=self._belt_y,
            velocity_x=self._conveyor_speed,
            sim_time=self._sim_time,
        )
        self._next_object_id += 1
        self.conveyor.add_object(obj)
        return obj

    def receive_decision(self, message: Dict[str, Any]) -> bool:
        """
        Accepts a validated sorting-decision message and queues the routing.

        Can accept:
          - A full UDP message dict (from UdpReceiver/build_message)
          - A minimal dict with at least 'class_id' key
          - An unresolved message (class_id present but source='unresolved')

        Unresolved messages (source='unresolved') with no usable class_id
        are silently discarded — no gate action is taken.

        Parameters
        ----------
        message : dict
            Must contain 'class_id' (int, 0–5).
            Optionally 'source', 'decision', 'confidence', 'entropy', 'override'.

        Returns
        -------
        bool — True if a routing decision was queued.
        """
        if not isinstance(message, dict):
            return False

        source = message.get("source", "classifier")
        # Unresolved decisions carry no routing information
        if source == "unresolved":
            return False

        class_id = message.get("class_id")
        if not isinstance(class_id, int) or not (0 <= class_id < NUM_CLASSES):
            return False

        self._decision_queue.append(class_id)
        self._last_decision = dict(message)
        return True

    def step(self, dt: float) -> Dict[str, Any]:
        """
        Advances the simulation by one fixed timestep.

        Order of operations per step:
          1. Advance conveyor → collect objects at gate and fallen objects
          2. Feed queued decision to gate (if gate is IDLE and queue has items)
          3. Advance gate → sort objects at gate
          4. Remove completed (sorted/fallen) objects from conveyor tracking
          5. Advance simulation clock

        Parameters
        ----------
        dt : float
            Timestep in seconds.

        Returns
        -------
        dict — step result summary:
            at_gate    : list of object IDs reaching the gate this step
            sorted_now : list of object IDs sorted this step
            fallen     : list of object IDs fallen this step
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")

        # 1. Conveyor step
        at_gate, fallen = self.conveyor.step(dt)

        # 2. Feed queued decision to gate when objects arrive at gate
        if at_gate and self.gate.is_idle and self._decision_queue:
            class_id = self._decision_queue.pop(0)
            self.gate.receive_decision(class_id)

        # 3. Gate step — sort objects at gate
        sorted_now = self.gate.step(dt, at_gate)

        # 4. Register fallen objects
        for obj in fallen:
            self.gate.register_fallen(obj)

        # 5. Remove completed objects from conveyor
        for obj in self.conveyor.active_objects:
            if obj.is_completed:
                self.conveyor.remove_object(obj)

        # 6. Advance sim clock
        self._sim_time += dt

        return {
            "at_gate": [o.object_id for o in at_gate],
            "sorted_now": [o.object_id for o in sorted_now],
            "fallen": [o.object_id for o in fallen],
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a snapshot of the current simulator state.

        Returns
        -------
        dict with keys:
            sim_time         : float — elapsed simulation time (seconds)
            conveyor_state   : str
            gate_state       : str
            gate_pending     : int or None — class_id queued at gate
            active_objects   : list of object state dicts
            decision_queue   : list of int — pending class_ids
            last_decision    : dict or None
            bin_counts       : dict {class_name: count}
            total_sorted     : int
            total_fallen     : int
        """
        active_objs = [
            {
                "object_id": o.object_id,
                "class_id": o.class_id,
                "class_name": o.class_name,
                "x": o.x,
                "y": o.y,
                "state": o.state.name,
                "sorted_to_bin": o.sorted_to_bin,
            }
            for o in self.conveyor.active_objects
        ]

        return {
            "sim_time": self._sim_time,
            "conveyor_state": self.conveyor.state.name,
            "gate_state": self.gate.state.name,
            "gate_pending": self.gate.pending_class_id,
            "active_objects": active_objs,
            "decision_queue": list(self._decision_queue),
            "last_decision": self._last_decision,
            "bin_counts": self.gate.get_bin_counts(),
            "total_sorted": self.gate.total_sorted,
            "total_fallen": self.gate.total_fallen,
        }
