from __future__ import annotations

from dataclasses import dataclass

from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import SimulationLog


@dataclass
class StateRecorderSystem:
    name: str = "state_recorder"

    def on_step(self, t_s: float, scenario: Scenario, log: SimulationLog) -> None:
        for sat in scenario.satellite_list():
            r, _v = sat.state_eci_at(t_s)
            log.record_position(sat.sat_id, t_s, r)
