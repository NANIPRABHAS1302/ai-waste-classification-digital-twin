"""
Digital Twin Sorting Simulator Application.

Orchestrates:
1. UDP receiver listening on localhost:5005
2. SortingSimulator physics and 6-bin diverter state machine
3. Pygame graphical renderer (with headless fallback)
4. Telemetry logging for all simulation actions (sorts, fall-offs)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from src.communication.udp_receiver import STATUS_OK, UdpReceiver
from src.simulator.renderer import PygameRenderer
from src.simulator.sorting_simulator import SortingSimulator
from src.telemetry.telemetry_logger import TelemetryLogger, TelemetryRecord


class SimulatorApp:
    """
    Coordinates UDP reception, digital twin physics stepping, rendering,
    and telemetry logging.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5005,
        conveyor_speed: float = 200.0,
        gate_position: float = 600.0,
        gate_duration: float = 0.5,
        telemetry_csv: Optional[str] = "results/telemetry/telemetry.csv",
        headless: bool = False,
        fps: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.fps = fps
        self.headless = headless
        self.receiver = UdpReceiver(host=self.host, port=self.port, timeout_s=0.005)
        self._sock = None
        self._init_socket()
        self.simulator = SortingSimulator(
            conveyor_speed=conveyor_speed,
            gate_x=gate_position,
            divert_duration_s=gate_duration,
        )
        self.renderer = PygameRenderer(headless=self.headless, fps=self.fps)
        self.logger = TelemetryLogger(csv_path=telemetry_csv)

        self._running = False
        self._object_counter = 0

    def _init_socket(self) -> None:
        """Initializes a persistent UDP socket for high-rate polling."""
        import socket
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(0.001)
            self._sock.bind((self.host, self.port))
        except Exception:
            self._sock = None

    def step_once(self, dt: float = 0.016) -> dict:
        """
        Executes one polling and physics cycle:
        1. Check for incoming UDP decision
        2. If decision arrived, spawn object and queue decision
        3. Step simulator physics
        4. Log any sort or fall events to telemetry
        5. Render display frame
        """
        # Poll UDP
        if self._sock is not None:
            udp_res = self.receiver.receive_one_on_socket(self._sock)
        else:
            udp_res = self.receiver.receive_one()
        if udp_res.get("status") == STATUS_OK and udp_res.get("message"):
            msg = udp_res["message"]
            class_id = msg.get("class_id")
            if class_id is not None:
                # Spawn corresponding item on conveyor
                obj = self.simulator.spawn_object(class_id)
                self.simulator.receive_decision(msg)

                # Log spawn event
                self.logger.log(
                    TelemetryRecord(
                        event_type="simulator_spawn",
                        object_id=obj.object_id,
                        class_id=obj.class_id,
                        class_name=obj.class_name,
                        simulation_state=self.simulator.conveyor.state.name,
                    )
                )

        # Step physics
        step_res = self.simulator.step(dt)

        # Log sort events
        for sorted_id in step_res.get("sorted_now", []):
            self.logger.log(
                TelemetryRecord(
                    event_type="simulator_sort",
                    object_id=sorted_id,
                    sorting_result="SORTED",
                    simulation_state=self.simulator.conveyor.state.name,
                )
            )

        # Log fall events
        for fallen_id in step_res.get("fallen", []):
            self.logger.log(
                TelemetryRecord(
                    event_type="simulator_fall",
                    object_id=fallen_id,
                    sorting_result="FALLEN",
                    simulation_state=self.simulator.conveyor.state.name,
                )
            )

        # Render display
        if self.renderer.initialized:
            self.renderer.render(self.simulator)

        return step_res

    def run_loop(self, max_seconds: Optional[float] = None) -> None:
        """Runs the simulator event loop."""
        self._running = True
        start_time = time.time()

        try:
            while self._running:
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    break

                if not self.renderer.handle_events():
                    break

                dt = self.renderer.tick()
                self.step_once(dt)
        finally:
            self.close()

    def close(self) -> None:
        """Cleans up simulator and renderer resources."""
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self.renderer.close()
        self.logger.close()


def main():
    parser = argparse.ArgumentParser(description="Digital Twin Waste Sorting Simulator App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="UDP host")
    parser.add_argument("--port", type=int, default=5005, help="UDP port")
    parser.add_argument("--speed", type=float, default=200.0, help="Conveyor speed (px/s)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--duration", type=float, default=None, help="Max run duration in seconds")
    args = parser.parse_args()

    app = SimulatorApp(
        host=args.host,
        port=args.port,
        conveyor_speed=args.speed,
        headless=args.headless,
    )
    app.run_loop(max_seconds=args.duration)


if __name__ == "__main__":
    main()
