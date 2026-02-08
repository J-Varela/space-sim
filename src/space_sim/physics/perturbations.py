"""
Orbital perturbation models for high-fidelity orbit propagation.

Includes:
- J2 gravitational harmonics (Earth oblateness)
- Atmospheric drag
- Solar radiation pressure
- Third-body perturbations (Sun, Moon)
"""

from __future__ import annotations

import math
from typing import Tuple
from space_sim.core.constants import MU_EARTH_KM3_S2, R_EARTH_KM
from space_sim.core.frames import Vector3


# Earth gravity model constants (WGS-84)
J2_EARTH: float = 1.08263e-3  # Second zonal harmonic coefficient
J3_EARTH: float = -2.53266e-6
J4_EARTH: float = -1.61962e-6

# Atmospheric drag parameters
RHO_0_KG_M3: float = 1.225  # Sea level density
H_KM: float = 8.5  # Scale height


def j2_perturbation_accel(r_eci: Vector3, mu: float = MU_EARTH_KM3_S2, 
                          r_earth: float = R_EARTH_KM, j2: float = J2_EARTH) -> Vector3:
    """
    Calculate acceleration due to J2 perturbation (Earth oblateness).
    
    The J2 perturbation is the most significant non-spherical gravitational effect,
    caused by Earth's equatorial bulge.
    
    Args:
        r_eci: Position vector in ECI frame (km)
        mu: Gravitational parameter (km³/s²)
        r_earth: Earth radius (km)
        j2: J2 harmonic coefficient
        
    Returns:
        Acceleration vector due to J2 (km/s²)
    """
    x, y, z = r_eci
    r = math.sqrt(x*x + y*y + z*z)
    
    if r < r_earth:
        # Below Earth surface
        return (0.0, 0.0, 0.0)
    
    # Common factor
    factor = (3.0 / 2.0) * j2 * mu * (r_earth / r)**2 / r**2
    
    # Z component factor
    z2_r2 = (z / r)**2
    
    # Acceleration components
    ax = factor * x / r * (5.0 * z2_r2 - 1.0)
    ay = factor * y / r * (5.0 * z2_r2 - 1.0)
    az = factor * z / r * (5.0 * z2_r2 - 3.0)
    
    return (ax, ay, az)


def j3_perturbation_accel(r_eci: Vector3, mu: float = MU_EARTH_KM3_S2,
                          r_earth: float = R_EARTH_KM, j3: float = J3_EARTH) -> Vector3:
    """
    Calculate acceleration due to J3 perturbation.
    
    Args:
        r_eci: Position vector in ECI frame (km)
        mu: Gravitational parameter (km³/s²)
        r_earth: Earth radius (km)
        j3: J3 harmonic coefficient
        
    Returns:
        Acceleration vector due to J3 (km/s²)
    """
    x, y, z = r_eci
    r = math.sqrt(x*x + y*y + z*z)
    
    if r < r_earth:
        return (0.0, 0.0, 0.0)
    
    factor = (1.0 / 2.0) * j3 * mu * (r_earth / r)**3 / r**2
    z_r = z / r
    z2_r2 = z_r * z_r
    
    ax = factor * x / r * (35.0 * z_r * z2_r2 - 15.0 * z_r)
    ay = factor * y / r * (35.0 * z_r * z2_r2 - 15.0 * z_r)
    az = factor * (35.0 * z2_r2 * z2_r2 - 30.0 * z2_r2 + 3.0)
    
    return (ax, ay, az)


def atmospheric_drag_accel(r_eci: Vector3, v_eci: Vector3, 
                          cd: float = 2.2, area_m2: float = 10.0, 
                          mass_kg: float = 1000.0) -> Vector3:
    """
    Calculate acceleration due to atmospheric drag.
    
    Uses exponential atmospheric density model.
    
    Args:
        r_eci: Position vector in ECI frame (km)
        v_eci: Velocity vector in ECI frame (km/s)
        cd: Drag coefficient (dimensionless)
        area_m2: Cross-sectional area (m²)
        mass_kg: Satellite mass (kg)
        
    Returns:
        Acceleration vector due to drag (km/s²)
    """
    x, y, z = r_eci
    r = math.sqrt(x*x + y*y + z*z)
    altitude_km = r - R_EARTH_KM
    
    # Only significant below ~1000 km
    if altitude_km > 1000.0 or altitude_km < 0.0:
        return (0.0, 0.0, 0.0)
    
    # Exponential atmospheric density model
    rho = RHO_0_KG_M3 * math.exp(-altitude_km / H_KM)  # kg/m³
    
    # Convert to km units: kg/m³ -> kg/km³
    rho_km = rho * 1e9
    
    # Velocity magnitude
    vx, vy, vz = v_eci
    v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    
    if v_mag < 1e-10:
        return (0.0, 0.0, 0.0)
    
    # Drag force: F = -0.5 * rho * Cd * A * v² * v_hat
    # Acceleration: a = F/m (convert area from m² to km²)
    area_km2 = area_m2 / 1e6
    coeff = -0.5 * rho_km * cd * area_km2 * v_mag / mass_kg
    
    ax = coeff * vx
    ay = coeff * vy
    az = coeff * vz
    
    return (ax, ay, az)


def solar_radiation_pressure_accel(r_eci: Vector3, r_sun_eci: Vector3,
                                    cr: float = 1.5, area_m2: float = 10.0,
                                    mass_kg: float = 1000.0,
                                    in_eclipse: bool = False) -> Vector3:
    """
    Calculate acceleration due to solar radiation pressure.
    
    Args:
        r_eci: Satellite position in ECI frame (km)
        r_sun_eci: Sun position in ECI frame (km)
        cr: Radiation pressure coefficient (dimensionless)
        area_m2: Cross-sectional area exposed to Sun (m²)
        mass_kg: Satellite mass (kg)
        in_eclipse: Whether satellite is in Earth's shadow
        
    Returns:
        Acceleration vector due to SRP (km/s²)
    """
    if in_eclipse:
        return (0.0, 0.0, 0.0)
    
    # Solar radiation pressure at 1 AU: P = 4.56e-6 N/m²
    P_SRP = 4.56e-6  # N/m²
    
    # Vector from satellite to Sun
    dx = r_sun_eci[0] - r_eci[0]
    dy = r_sun_eci[1] - r_eci[1]
    dz = r_sun_eci[2] - r_eci[2]
    
    r_sun_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    if r_sun_dist < 1e-10:
        return (0.0, 0.0, 0.0)
    
    # Unit vector toward Sun
    u_sun = (dx / r_sun_dist, dy / r_sun_dist, dz / r_sun_dist)
    
    # Acceleration: a = P * Cr * A / m
    # Convert area to km²: m² to km² is divide by 1e6
    # Convert pressure: N/m² = kg/(m·s²) = kg/(1000km·s²) for km units
    # Result needs to be in km/s²
    
    # Force = P * Cr * A  [N = kg·m/s²]
    # Accel = Force / mass [m/s²]
    # Convert to km/s²: divide by 1000
    accel_mag = (P_SRP * cr * area_m2 / mass_kg) / 1000.0  # km/s²
    
    ax = accel_mag * u_sun[0]
    ay = accel_mag * u_sun[1]
    az = accel_mag * u_sun[2]
    
    return (ax, ay, az)


def third_body_accel(r_sat_eci: Vector3, r_body_eci: Vector3, 
                    mu_body: float) -> Vector3:
    """
    Calculate third-body gravitational perturbation (Sun or Moon).
    
    Args:
        r_sat_eci: Satellite position in ECI frame (km)
        r_body_eci: Third body position in ECI frame (km)
        mu_body: Gravitational parameter of third body (km³/s²)
        
    Returns:
        Acceleration vector (km/s²)
    """
    # Vector from satellite to body
    dx_sb = r_body_eci[0] - r_sat_eci[0]
    dy_sb = r_body_eci[1] - r_sat_eci[1]
    dz_sb = r_body_eci[2] - r_sat_eci[2]
    
    r_sb = math.sqrt(dx_sb*dx_sb + dy_sb*dy_sb + dz_sb*dz_sb)
    
    # Vector from Earth to body
    r_eb = math.sqrt(r_body_eci[0]**2 + r_body_eci[1]**2 + r_body_eci[2]**2)
    
    if r_sb < 1.0 or r_eb < 1.0:
        return (0.0, 0.0, 0.0)
    
    # Third-body acceleration
    # a = mu_body * [r_sb/|r_sb|³ - r_eb/|r_eb|³]
    ax = mu_body * (dx_sb / r_sb**3 - r_body_eci[0] / r_eb**3)
    ay = mu_body * (dy_sb / r_sb**3 - r_body_eci[1] / r_eb**3)
    az = mu_body * (dz_sb / r_sb**3 - r_body_eci[2] / r_eb**3)
    
    return (ax, ay, az)


# Sun and Moon gravitational parameters (km³/s²)
MU_SUN = 1.32712440018e11
MU_MOON = 4.9028e3


def total_perturbation_accel(r_eci: Vector3, v_eci: Vector3,
                             include_j2: bool = True,
                             include_j3: bool = False,
                             include_drag: bool = False,
                             include_srp: bool = False,
                             include_sun: bool = False,
                             include_moon: bool = False,
                             sat_mass_kg: float = 1000.0,
                             sat_area_m2: float = 10.0,
                             sat_cd: float = 2.2,
                             sat_cr: float = 1.5,
                             r_sun_eci: Vector3 = (149598023.0, 0.0, 0.0),
                             r_moon_eci: Vector3 = (384400.0, 0.0, 0.0),
                             in_eclipse: bool = False) -> Vector3:
    """
    Calculate total perturbation acceleration from selected sources.
    
    Args:
        r_eci: Position vector (km)
        v_eci: Velocity vector (km/s)
        include_j2: Include J2 perturbation
        include_j3: Include J3 perturbation
        include_drag: Include atmospheric drag
        include_srp: Include solar radiation pressure
        include_sun: Include Sun third-body effects
        include_moon: Include Moon third-body effects
        sat_mass_kg: Satellite mass (kg)
        sat_area_m2: Satellite cross-sectional area (m²)
        sat_cd: Drag coefficient
        sat_cr: Radiation pressure coefficient
        r_sun_eci: Sun position in ECI (km)
        r_moon_eci: Moon position in ECI (km)
        in_eclipse: Is satellite in Earth's shadow
        
    Returns:
        Total perturbation acceleration (km/s²)
    """
    ax, ay, az = 0.0, 0.0, 0.0
    
    if include_j2:
        a_j2 = j2_perturbation_accel(r_eci)
        ax += a_j2[0]
        ay += a_j2[1]
        az += a_j2[2]
    
    if include_j3:
        a_j3 = j3_perturbation_accel(r_eci)
        ax += a_j3[0]
        ay += a_j3[1]
        az += a_j3[2]
    
    if include_drag:
        a_drag = atmospheric_drag_accel(r_eci, v_eci, sat_cd, sat_area_m2, sat_mass_kg)
        ax += a_drag[0]
        ay += a_drag[1]
        az += a_drag[2]
    
    if include_srp:
        a_srp = solar_radiation_pressure_accel(r_eci, r_sun_eci, sat_cr, 
                                              sat_area_m2, sat_mass_kg, in_eclipse)
        ax += a_srp[0]
        ay += a_srp[1]
        az += a_srp[2]
    
    if include_sun:
        a_sun = third_body_accel(r_eci, r_sun_eci, MU_SUN)
        ax += a_sun[0]
        ay += a_sun[1]
        az += a_sun[2]
    
    if include_moon:
        a_moon = third_body_accel(r_eci, r_moon_eci, MU_MOON)
        ax += a_moon[0]
        ay += a_moon[1]
        az += a_moon[2]
    
    return (ax, ay, az)
