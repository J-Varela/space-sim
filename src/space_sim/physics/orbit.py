# src/space_sim/orbit.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from space_sim.core.constants import MU_EARTH_KM3_S2
from space_sim.physics.gravity import solve_keplers_equation, wrap_to_2pi
from space_sim.core.frames import perifocal_to_eci, Vector3


@dataclass(frozen=True)
class OrbitalElements:
    """
    Classical Orbital Elements (COEs) for an elliptic orbit.

    Units:
        a_km: semi-major axis in km
        e: eccentricity (0<=e<1)
        inc_rad: inclination in radians
        raan_rad: right ascension of ascending node in radians
        argp_rad: argument of periapsis in radians
        M0_rad: mean anomaly at epoch (t=0) in radians
    """
    a_km: float
    e: float
    inc_rad: float
    raan_rad: float
    argp_rad: float
    M0_rad: float

    def __post_init__(self):
        if self.a_km <= 0:
            raise ValueError("Semi-major axis must be positive.")
        if not (0.0 <= self.e < 1.0):
            raise ValueError("This Phase-1 model supports elliptic orbits only (0 <= e < 1).")
        if not (0.0 <= self.inc_rad <= math.pi):
            raise ValueError(f"Inclination must be in range [0, π] radians. Got: {self.inc_rad}")
        if not math.isfinite(self.raan_rad):
            raise ValueError(f"RAAN must be finite. Got: {self.raan_rad}")
        if not math.isfinite(self.argp_rad):
            raise ValueError(f"Argument of periapsis must be finite. Got: {self.argp_rad}")
        if not math.isfinite(self.M0_rad):
            raise ValueError(f"Mean anomaly must be finite. Got: {self.M0_rad}")


def mean_motion_rad_s(a_km: float, mu_km3_s2: float = MU_EARTH_KM3_S2) -> float:
    """n = sqrt(mu / a^3)."""
    return math.sqrt(mu_km3_s2 / (a_km ** 3))


def coe_to_rv_eci(elements: OrbitalElements, t_s: float, mu_km3_s2: float = MU_EARTH_KM3_S2) -> Tuple[Vector3, Vector3]:
    """
    Convert orbital elements at epoch + t to ECI position and velocity (km, km/s).
    Two-body Keplerian propagation using mean anomaly.

    Returns:
        r_eci (km), v_eci (km/s)
    """
    a = elements.a_km
    e = elements.e
    inc = elements.inc_rad
    raan = elements.raan_rad
    argp = elements.argp_rad

    n = mean_motion_rad_s(a, mu_km3_s2)
    M = wrap_to_2pi(elements.M0_rad + n * t_s)

    E = solve_keplers_equation(M, e)

    # True anomaly ν from eccentric anomaly E
    sin_v = (math.sqrt(1.0 - e * e) * math.sin(E)) / (1.0 - e * math.cos(E))
    cos_v = (math.cos(E) - e) / (1.0 - e * math.cos(E))
    nu = math.atan2(sin_v, cos_v)

    # Distance r
    r_km = a * (1.0 - e * math.cos(E))

    # Position in PQW
    r_pqw: Vector3 = (r_km * math.cos(nu), r_km * math.sin(nu), 0.0)

    # Velocity in PQW
    # p = a(1-e^2)
    p = a * (1.0 - e * e)
    h = math.sqrt(mu_km3_s2 * p)  # specific angular momentum
    v_pqw: Vector3 = (
        -mu_km3_s2 / h * math.sin(nu),
        mu_km3_s2 / h * (e + math.cos(nu)),
        0.0,
    )

    # Rotate PQW -> ECI
    r_eci, v_eci = perifocal_to_eci(r_pqw, v_pqw, raan, inc, argp)
    return r_eci, v_eci


def propagate(elements: OrbitalElements, times_s: List[float], mu_km3_s2: float = MU_EARTH_KM3_S2) -> List[Tuple[float, Vector3, Vector3]]:
    """
    Propagate an orbit across a list of time stamps (seconds since epoch).
    Returns list of (t, r_eci, v_eci).
    """
    out: List[Tuple[float, Vector3, Vector3]] = []
    for t in times_s:
        r, v = coe_to_rv_eci(elements, t, mu_km3_s2)
        out.append((t, r, v))
    return out
