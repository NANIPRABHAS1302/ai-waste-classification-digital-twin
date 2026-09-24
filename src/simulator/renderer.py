"""
Renderer module for the digital twin sorting simulator.

Provides visual rendering of:
- Conveyor belt and movement markers
- Sorting diverter gate and its state (IDLE / DIVERTING / RESETTING)
- Six sorting bins with class labels, fill levels, and color-coded bins
- Waste items with class color, object ID, and text labels
- Heads-Up Display (HUD):
    - Conveyor status & simulation time
    - Gate state & pending routing
    - Last received decision (source, confidence, entropy)
    - Total sorted / fallen counts and sorting accuracy rate

DESIGN PRINCIPLES
-----------------
- Modular & optional: the physics engine in SortingSimulator runs completely
  independently of this renderer.
- Headless support: If Pygame is not available or if running in headless mode,
  PygameRenderer gracefully handles initialization or provides a no-op / headless
  mode without crashing.
- Deterministic rendering: does not mutate physics state.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore
    PYGAME_AVAILABLE = False

from src.simulator.conveyor import ConveyorState
from src.simulator.sorting_gate import BIN_NAMES, NUM_BINS, GateState
from src.simulator.sorting_simulator import SortingSimulator
from src.simulator.waste_object import CLASS_COLOURS, CLASS_NAMES, WasteObject

# ---------------------------------------------------------------------------
# Visual Styling Constants
# ---------------------------------------------------------------------------

WINDOW_WIDTH: int = 1000
WINDOW_HEIGHT: int = 600

# Palette (RGB)
COLOR_BG: Tuple[int, int, int] = (24, 26, 32)            # Dark charcoal
COLOR_PANEL_BG: Tuple[int, int, int] = (34, 38, 46)      # Slate grey panel
COLOR_TEXT_MAIN: Tuple[int, int, int] = (235, 240, 245)  # Soft white
COLOR_TEXT_MUTED: Tuple[int, int, int] = (150, 160, 175) # Dim text
COLOR_ACCENT: Tuple[int, int, int] = (64, 150, 255)      # Blue highlight

COLOR_CONVEYOR_BELT: Tuple[int, int, int] = (45, 50, 60)
COLOR_CONVEYOR_ROLLER: Tuple[int, int, int] = (80, 85, 95)
COLOR_CONVEYOR_BORDER: Tuple[int, int, int] = (100, 110, 125)

COLOR_GATE_IDLE: Tuple[int, int, int] = (90, 100, 115)       # Neutral grey
COLOR_GATE_DIVERTING: Tuple[int, int, int] = (46, 204, 113)  # Active emerald green
COLOR_GATE_RESETTING: Tuple[int, int, int] = (230, 126, 34)  # Amber returning

COLOR_BIN_BORDER: Tuple[int, int, int] = (60, 65, 75)
COLOR_BIN_FILL_BG: Tuple[int, int, int] = (28, 30, 36)

# Geometry
CONVEYOR_HEIGHT: int = 40
OBJECT_RADIUS: int = 14


class PygameRenderer:
    """
    Renders the waste-sorting digital twin using Pygame.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    headless : bool
        If True, initializes using the headless dummy video driver (useful in CI/WSL).
    fps : int
        Target frame rate cap for tick().
    """

    def __init__(
        self,
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
        headless: bool = False,
        fps: int = 60,
    ) -> None:
        self.width = width
        self.height = height
        self.headless = headless
        self.fps = fps

        self.initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_large = None

        if PYGAME_AVAILABLE:
            self._init_pygame()

    def _init_pygame(self) -> None:
        """Initializes Pygame display, fonts, and clock."""
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        try:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Waste Sorting — Digital Twin Simulator")
            self.clock = pygame.time.Clock()

            # Initialize standard system fonts
            self.font = pygame.font.SysFont("Arial", 14)
            self.font_small = pygame.font.SysFont("Arial", 12)
            self.font_large = pygame.font.SysFont("Arial", 18, bold=True)
            self.initialized = True
        except Exception:
            # Fall back safely if video driver fails
            self.initialized = False
            self.screen = None

    def render(self, simulator: SortingSimulator) -> bool:
        """
        Renders the complete simulation state to the screen.

        Parameters
        ----------
        simulator : SortingSimulator
            The simulator instance providing state to render.

        Returns
        -------
        bool
            True if rendering succeeded, False if Pygame is uninitialized.
        """
        if not self.initialized or self.screen is None:
            return False

        # Clear background
        self.screen.fill(COLOR_BG)

        state = simulator.get_state()

        # Render sections
        self._render_hud(state)
        self._render_conveyor(simulator)
        self._render_gate(simulator)
        self._render_bins(simulator)
        self._render_objects(simulator)

        # Update display
        pygame.display.flip()
        return True

    def _render_hud(self, state: Dict[str, Any]) -> None:
        """Renders the top dashboard HUD."""
        # Top panel background
        panel_rect = pygame.Rect(10, 10, self.width - 20, 110)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_CONVEYOR_BORDER, panel_rect, width=1, border_radius=8)

        # Title
        title_surf = self.font_large.render("DIGITAL TWIN SORTING SIMULATOR", True, COLOR_ACCENT)
        self.screen.blit(title_surf, (25, 20))

        # Simulation telemetry
        sim_time = state.get("sim_time", 0.0)
        conv_state = state.get("conveyor_state", "RUNNING")
        gate_state = state.get("gate_state", "IDLE")
        total_sorted = state.get("total_sorted", 0)
        total_fallen = state.get("total_fallen", 0)
        total_processed = total_sorted + total_fallen
        accuracy = (total_sorted / total_processed * 100.0) if total_processed > 0 else 100.0

        col1_text = (
            f"Sim Time: {sim_time:6.2f}s | "
            f"Conveyor: {conv_state} | "
            f"Gate: {gate_state} | "
            f"Sorted: {total_sorted} | "
            f"Fallen: {total_fallen} | "
            f"Efficiency: {accuracy:5.1f}%"
        )
        self.screen.blit(self.font.render(col1_text, True, COLOR_TEXT_MAIN), (25, 48))

        # Last received decision info
        last_dec = state.get("last_decision")
        if last_dec:
            src = last_dec.get("source", "classifier")
            c_name = last_dec.get("class_name", "none")
            conf = last_dec.get("confidence", 0.0)
            ent = last_dec.get("entropy", 0.0)
            override = last_dec.get("override", False)

            dec_str = (
                f"Latest Decision: [{c_name.upper()}] (Source: {src}) | "
                f"Confidence: {conf:4.2f} | Entropy: {ent:4.2f} | "
                f"Override: {'YES' if override else 'NO'}"
            )
            dec_color = (46, 204, 113) if src == "classifier" else (241, 196, 15)
        else:
            dec_str = "Latest Decision: [WAITING FOR UDP / MANUAL DECISION]"
            dec_color = COLOR_TEXT_MUTED

        self.screen.blit(self.font_small.render(dec_str, True, dec_color), (25, 75))

        # Instructions / footer in HUD
        ctrl_str = "Status: Online | Protocol: Localhost UDP JSON (Port 5005) | Deterministic Physics: Fixed dt"
        self.screen.blit(self.font_small.render(ctrl_str, True, COLOR_TEXT_MUTED), (25, 95))

    def _render_conveyor(self, simulator: SortingSimulator) -> None:
        """Renders the conveyor belt track and rollers."""
        conv = simulator.conveyor
        start_x = int(conv.belt_start_x)
        end_x = int(conv.belt_end_x)
        belt_y = int(simulator._belt_y)

        # Belt track
        belt_rect = pygame.Rect(
            start_x,
            belt_y - CONVEYOR_HEIGHT // 2,
            end_x - start_x,
            CONVEYOR_HEIGHT,
        )
        pygame.draw.rect(self.screen, COLOR_CONVEYOR_BELT, belt_rect)
        pygame.draw.rect(self.screen, COLOR_CONVEYOR_BORDER, belt_rect, width=2)

        # Belt direction arrows / tick marks
        step_px = 60
        for x in range(start_x + 30, end_x - 20, step_px):
            pygame.draw.circle(self.screen, COLOR_CONVEYOR_ROLLER, (x, belt_y), 4)

        # Labels
        lbl_start = self.font_small.render("SPAWN", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_start, (start_x, belt_y + CONVEYOR_HEIGHT // 2 + 5))

        lbl_end = self.font_small.render("FALL-OFF", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_end, (end_x - 40, belt_y + CONVEYOR_HEIGHT // 2 + 5))

    def _render_gate(self, simulator: SortingSimulator) -> None:
        """Renders the diverter sorting gate."""
        gate = simulator.gate
        gate_x = int(gate.gate_x)
        gate_y = int(gate.gate_y)

        # Pick gate color according to state
        if gate.state == GateState.DIVERTING:
            gate_col = COLOR_GATE_DIVERTING
            state_text = f"DIVERTING -> BIN {gate.pending_class_id}"
        elif gate.state == GateState.RESETTING:
            gate_col = COLOR_GATE_RESETTING
            state_text = "RESETTING"
        else:
            gate_col = COLOR_GATE_IDLE
            state_text = "IDLE (WAITING)"

        # Vertical actuator arm
        pygame.draw.line(
            self.screen,
            gate_col,
            (gate_x, gate_y - CONVEYOR_HEIGHT // 2 - 25),
            (gate_x, gate_y + CONVEYOR_HEIGHT // 2 + 10),
            width=5,
        )

        # Gate housing box
        box_w, box_h = 36, 20
        box_rect = pygame.Rect(
            gate_x - box_w // 2,
            gate_y - CONVEYOR_HEIGHT // 2 - 40,
            box_w,
            box_h,
        )
        pygame.draw.rect(self.screen, gate_col, box_rect, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_TEXT_MAIN, box_rect, width=1, border_radius=4)

        # Label above gate
        txt_surf = self.font_small.render(f"GATE: {state_text}", True, gate_col)
        self.screen.blit(txt_surf, (gate_x - txt_surf.get_width() // 2, gate_y - CONVEYOR_HEIGHT // 2 - 58))

    def _render_bins(self, simulator: SortingSimulator) -> None:
        """Renders the 6 sorting bins at the bottom of the display."""
        gate = simulator.gate
        bin_counts = gate.get_bin_counts()

        total_bins = NUM_BINS
        bin_margin = 15
        total_available_w = self.width - 40
        bin_w = (total_available_w - (total_bins - 1) * bin_margin) // total_bins
        bin_h = 130
        bin_y = self.height - bin_h - 20
        start_x = 20

        for i, class_name in enumerate(BIN_NAMES):
            bx = start_x + i * (bin_w + bin_margin)
            rect = pygame.Rect(bx, bin_y, bin_w, bin_h)

            # Bin background and border
            pygame.draw.rect(self.screen, COLOR_BIN_FILL_BG, rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_BIN_BORDER, rect, width=2, border_radius=6)

            # Class color banner
            banner_rect = pygame.Rect(bx, bin_y, bin_w, 18)
            banner_col = CLASS_COLOURS.get(i, (100, 100, 100))
            pygame.draw.rect(self.screen, banner_col, banner_rect, border_top_left_radius=6, border_top_right_radius=6)

            # Bin ID and name
            lbl_name = self.font_small.render(f"[{i}] {class_name[:8].upper()}", True, (255, 255, 255))
            self.screen.blit(lbl_name, (bx + 5, bin_y + 2))

            # Received counter
            cnt = bin_counts.get(class_name, 0)
            cnt_surf = self.font_large.render(str(cnt), True, COLOR_TEXT_MAIN)
            self.screen.blit(
                cnt_surf,
                (bx + bin_w // 2 - cnt_surf.get_width() // 2, bin_y + bin_h // 2 - 15),
            )

            # Items label
            lbl_items = self.font_small.render("items", True, COLOR_TEXT_MUTED)
            self.screen.blit(
                lbl_items,
                (bx + bin_w // 2 - lbl_items.get_width() // 2, bin_y + bin_h // 2 + 10),
            )

            # Highlight bin if gate is diverting into it
            if gate.state == GateState.DIVERTING and gate.pending_class_id == i:
                pygame.draw.rect(self.screen, COLOR_GATE_DIVERTING, rect, width=3, border_radius=6)

    def _render_objects(self, simulator: SortingSimulator) -> None:
        """Renders waste objects currently on the conveyor."""
        for obj in simulator.conveyor.active_objects:
            if not obj.is_active:
                continue

            ox = int(obj.x)
            oy = int(obj.y)

            # Waste item circle with class color
            col = obj.colour
            pygame.draw.circle(self.screen, col, (ox, oy), OBJECT_RADIUS)
            pygame.draw.circle(self.screen, COLOR_TEXT_MAIN, (ox, oy), OBJECT_RADIUS, width=1)

            # ID label inside or next to circle
            id_surf = self.font_small.render(str(obj.object_id), True, (255, 255, 255))
            self.screen.blit(
                id_surf,
                (ox - id_surf.get_width() // 2, oy - id_surf.get_height() // 2),
            )

            # Class label above the item
            lbl = self.font_small.render(obj.class_name, True, COLOR_TEXT_MAIN)
            self.screen.blit(lbl, (ox - lbl.get_width() // 2, oy - OBJECT_RADIUS - 16))

    def tick(self) -> float:
        """
        Limits frame rate and returns seconds since last call.

        Returns
        -------
        float
            Delta time in seconds.
        """
        if self.clock is not None:
            return self.clock.tick(self.fps) / 1000.0
        return 1.0 / self.fps

    def handle_events(self) -> bool:
        """
        Processes standard Pygame events (e.g. window close).

        Returns
        -------
        bool
            False if QUIT event was detected, True otherwise.
        """
        if not self.initialized:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        return True

    def close(self) -> None:
        """Shuts down Pygame display resources cleanly."""
        if self.initialized and pygame is not None:
            try:
                pygame.quit()
            except Exception:
                pass
            self.initialized = False
            self.screen = None
