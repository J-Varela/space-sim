from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from space_sim.core.frames import (
    Vector3,
    geodetic_to_ecef_km,
    ecef_to_eci_km,
)
from space_sim.physics.visibility import is_visible, compute_access_windows
from space_sim.objects.satellite import Satellite


@dataclass
class GroundStation:
    station_id: str
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 10.0

    def __post_init__(self):
        if not (-90.0 <= self.lat_deg <= 90.0):
            raise ValueError(f"Latitude must be in range [-90, 90] degrees. Got: {self.lat_deg}")
        if not (-180.0 <= self.lon_deg <= 180.0):
            raise ValueError(f"Longitude must be in range [-180, 180] degrees. Got: {self.lon_deg}")
        if self.alt_km < 0:
            raise ValueError(f"Altitude must be non-negative. Got: {self.alt_km}")
        if not (0.0 <= self.min_elevation_deg <= 90.0):
            raise ValueError(f"Minimum elevation must be in range [0, 90] degrees. Got: {self.min_elevation_deg}")
        if not self.station_id.strip():
            raise ValueError("Station ID cannot be empty or whitespace.")
        if not self.name.strip():
            raise ValueError("Station name cannot be empty or whitespace.")

    def position_eci_km(self, t_s: float) -> Vector3:
        lat = math.radians(self.lat_deg)
        lon = math.radians(self.lon_deg)
        r_ecef = geodetic_to_ecef_km(lat, lon, self.alt_km)
        return ecef_to_eci_km(r_ecef, t_s)

    def can_see(self, sat: Satellite, t_s: float) -> bool:
        r_gs = self.position_eci_km(t_s)
        r_sat, _v_sat = sat.state_eci_at(t_s)
        return is_visible(r_gs, r_sat, min_elevation_deg=self.min_elevation_deg)

    def access_windows(self, sat: Satellite, times_s: List[float]) -> List[Tuple[float, float]]:
        flags = [self.can_see(sat, t) for t in times_s]
        return compute_access_windows(times_s, flags)
