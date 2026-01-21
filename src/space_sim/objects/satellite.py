from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

from space_sim.physics.orbit import OrbitalElements, coe_to_rv_eci
from space_sim.core.frames import Vector3


@dataclass
class Satellite:
    """
    A domain object representing a satellite.
    Phase 1: purely kinematic (orbit-driven) state from Keplerian propagation.
    """
    sat_id: str
    name: str
    elements: OrbitalElements

    # Cached state (optional; useful once you do step-based simulation)
    last_t_s: Optional[float] = None
    last_r_eci_km: Optional[Vector3] = None
    last_v_eci_km_s: Optional[Vector3] = None

    def state_eci_at(self, t_s: float) -> Tuple[Vector3, Vector3]:
        """
        Returns ECI position/velocity at time t_s (seconds since scenario epoch).
        """
        r_eci, v_eci = coe_to_rv_eci(self.elements, t_s)

        # Cache it (handy later for engine stepping / debugging)
        self.last_t_s = t_s
        self.last_r_eci_km = r_eci
        self.last_v_eci_km_s = v_eci

        return r_eci, v_eci
