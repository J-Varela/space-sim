from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Any, Tuple

from space_sim.core.frames import Vector3
from space_sim.simulation.scenario import Scenario


class System(Protocol):
    """
    Plugin interface for simulation systems.
    Each system runs per tick and can write to the log.
    """
    name: str

    def on_step(self, t_s: float, scenario: Scenario, log: "SimulationLog") -> None:
        ...


@dataclass
class SimulationLog:
    """
    Stores outputs from a simulation run.
    Keep it simple and serializable.
    """
    # Positions: sat_id -> list of (t, r_eci)
    sat_positions_eci_km: Dict[str, List[Tuple[float, Vector3]]] = field(default_factory=dict)

    # Visibility: (gs_id, sat_id) -> list of (t, visible_bool)
    visibility: Dict[Tuple[str, str], List[Tuple[float, bool]]] = field(default_factory=dict)

    # Free-form events later
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record_position(self, sat_id: str, t_s: float, r_eci: Vector3) -> None:
        self.sat_positions_eci_km.setdefault(sat_id, []).append((t_s, r_eci))

    def record_visibility(self, gs_id: str, sat_id: str, t_s: float, visible: bool) -> None:
        key = (gs_id, sat_id)
        self.visibility.setdefault(key, []).append((t_s, visible))


@dataclass
class Engine:
    """
    Fixed-step simulation engine.
    Deterministic replay: given same scenario + dt + start/end => same output.
    """
    dt_s: float
    systems: List[System] = field(default_factory=list)

    def run(self, scenario: Scenario, t_start_s: float, t_end_s: float) -> SimulationLog:
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive.")
        if t_end_s < t_start_s:
            raise ValueError("t_end_s must be >= t_start_s.")

        log = SimulationLog()
        t = t_start_s

        # Tick loop
        # Note: inclusive end if it lands exactly; otherwise last tick < end
        while t <= t_end_s + 1e-9:
            # Run systems (each system decides what to record)
            for sys in self.systems:
                sys.on_step(t, scenario, log)

            t += self.dt_s

        return log
