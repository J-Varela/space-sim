from __future__ import annotations

import math
from typing import Tuple, List, Optional

from space_sim.core.constants import R_EARTH_KM
from space_sim.core.frames import Vector3, dot, sub, norm


def elevation_angle_rad(r_gs_eci: Vector3, r_sat_eci: Vector3) -> float:
    """
    Elevation angle above the local horizon at the ground station.
    Uses ground-station "up" direction ~= radial vector (spherical Earth model).
    """
    rho = sub(r_sat_eci, r_gs_eci)          # line-of-sight vector from GS to SAT
    gs_up = r_gs_eci                        # radial outward vector
    # elevation = 90 - angle(rho, up)
    # sin(elev) = dot(rho_hat, up_hat)
    rho_hat = (rho[0]/norm(rho), rho[1]/norm(rho), rho[2]/norm(rho))
    up_hat = (gs_up[0]/norm(gs_up), gs_up[1]/norm(gs_up), gs_up[2]/norm(gs_up))
    s = dot(rho_hat, up_hat)
    # clamp for numeric stability
    s = max(-1.0, min(1.0, s))
    return math.asin(s)


def is_visible(
    r_gs_eci: Vector3,
    r_sat_eci: Vector3,
    min_elevation_deg: float = 10.0,
) -> bool:
    """
    Visible if satellite is above min elevation (horizon mask).
    Earth occultation is implicitly handled by elevation check in this spherical model.
    """
    elev = elevation_angle_rad(r_gs_eci, r_sat_eci)
    return elev >= math.radians(min_elevation_deg)


def compute_access_windows(
    times_s: List[float],
    visible_flags: List[bool],
) -> List[Tuple[float, float]]:
    """
    Convert a boolean visibility time series into access windows [t_start, t_end].
    Assumes times_s is sorted and evenly-ish spaced (works best with uniform dt).
    """
    if len(times_s) != len(visible_flags):
        raise ValueError("times_s and visible_flags must be same length.")

    windows: List[Tuple[float, float]] = []
    in_pass = False
    t_start: Optional[float] = None

    for t, vis in zip(times_s, visible_flags):
        if vis and not in_pass:
            in_pass = True
            t_start = t
        elif (not vis) and in_pass:
            in_pass = False
            windows.append((t_start if t_start is not None else times_s[0], t))
            t_start = None

    if in_pass and t_start is not None:
        windows.append((t_start, times_s[-1]))

    return windows
