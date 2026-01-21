from __future__ import annotations

import math
from typing import Tuple

from space_sim.core.constants import R_EARTH_KM, OMEGA_EARTH_RAD_S

Vector3 = Tuple[float, float, float]


def rot3(angle_rad: float, v: Vector3) -> Vector3:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = v
    return (c * x - s * y, s * x + c * y, z)


def rot1(angle_rad: float, v: Vector3) -> Vector3:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = v
    return (x, c * y - s * z, s * y + c * z)


def geodetic_to_ecef_km(lat_rad: float, lon_rad: float, alt_km: float) -> Vector3:
    """
    Spherical Earth approximation (good enough for Phase 2).
    """
    r = R_EARTH_KM + alt_km
    clat = math.cos(lat_rad)
    slat = math.sin(lat_rad)
    clon = math.cos(lon_rad)
    slon = math.sin(lon_rad)

    x = r * clat * clon
    y = r * clat * slon
    z = r * slat
    return (x, y, z)


def ecef_to_latlon_deg(r_ecef_km: Vector3) -> tuple[float, float]:
    """
    Spherical Earth approximation: ECEF -> geocentric lat/lon (deg)
    Returns lon wrapped to [-180, 180).
    """
    x, y, z = r_ecef_km
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0:
        raise ValueError("Zero ECEF vector.")
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def ecef_to_eci_km(r_ecef: Vector3, t_s: float) -> Vector3:
    """
    Very simple Earth rotation model: ECI = R3( +omega*t ) * ECEF
    Assumes ECEF and ECI axes aligned at t=0.
    """
    theta = OMEGA_EARTH_RAD_S * t_s
    return rot3(theta, r_ecef)


def eci_to_ecef_km(r_eci: Vector3, t_s: float) -> Vector3:
    """
    Inverse rotation: ECEF = R3( -omega*t ) * ECI
    """
    theta = -OMEGA_EARTH_RAD_S * t_s
    return rot3(theta, r_eci)


def dot(a: Vector3, b: Vector3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def norm(a: Vector3) -> float:
    return math.sqrt(dot(a, a))


def perifocal_to_eci(r_pqw: Vector3, v_pqw: Vector3, raan_rad: float, inc_rad: float, argp_rad: float) -> Tuple[Vector3, Vector3]:
    """
    Convert position and velocity from perifocal (PQW) frame to ECI frame.
    
    Args:
        r_pqw: Position vector in PQW frame (km)
        v_pqw: Velocity vector in PQW frame (km/s)
        raan_rad: Right ascension of ascending node (radians)
        inc_rad: Inclination (radians)
        argp_rad: Argument of periapsis (radians)
    
    Returns:
        (r_eci, v_eci): Position and velocity in ECI frame
    """
    # Rotation sequence: R3(-argp) * R1(-inc) * R3(-raan)
    # Applied in reverse order: first RAAN, then inclination, then argument of periapsis
    
    # First rotation: R3(-raan)
    r_temp = rot3(-raan_rad, r_pqw)
    v_temp = rot3(-raan_rad, v_pqw)
    
    # Second rotation: R1(-inc)
    r_temp = rot1(-inc_rad, r_temp)
    v_temp = rot1(-inc_rad, v_temp)
    
    # Third rotation: R3(-argp)
    r_eci = rot3(-argp_rad, r_temp)
    v_eci = rot3(-argp_rad, v_temp)
    
    return r_eci, v_eci
