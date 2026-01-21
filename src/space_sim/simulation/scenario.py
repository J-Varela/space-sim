from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation

@dataclass
class Scenario:
    """
    Container for all objects in a simulation run.
    Keep this pure: just data + lookup, no stepping logic. 
    """
    name: str 
    satellites: Dict[str, Satellite] = field(default_factory=dict)
    ground_stations: Dict[str, GroundStation] = field(default_factory=dict)

    def add_satellite(self, sat: Satellite) -> None: 
        if sat.sat_id in self.satellites:
            raise ValueError(f"Duplicate satellite ID: {sat.sat_id}")
        self.satellites[sat.sat_id] = sat
    
    def add_ground_station(self,gs: GroundStation) -> None: 
        if gs.station_id in self.ground_stations:
            raise ValueError(f"Duplicate ground station ID: {gs.station_id}")
        self.ground_stations[gs.station_id] = gs
    
    def satellite_list(self) -> List[Satellite]: 
        return list(self.satellites.values())
    
    def ground_station_list(self) -> List[GroundStation]:
        return list(self.ground_stations.values())