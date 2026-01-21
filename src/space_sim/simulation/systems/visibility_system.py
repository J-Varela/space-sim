from __future__ import annotations

from dataclasses import dataclass

from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import SimulationLog

@dataclass
class VisibilitySystem: 
    name: str = "visibility"

    def on_step(self, t_s: float, scenario: Scenario, log: SimulationLog) -> None:
        sats = scenario.satellite_list()
        gss = scenario.ground_station_list()

        for gs in gss:
            for sat in sats: 
                visible = gs.can_see(sat, t_s)
                log.record_visibility(gs.station_id, sat.sat_id, t_s, visible)