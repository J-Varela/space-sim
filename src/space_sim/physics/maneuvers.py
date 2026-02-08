"""
Orbital maneuver planning and delta-V calculations.

Includes:
- Hohmann transfers
- Bi-elliptic transfers
- Plane change maneuvers
- Rendezvous calculations
- Delta-V budgets
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List
from space_sim.core.frames import Vector3
from space_sim.core.constants import MU_EARTH_KM3_S2
from space_sim.physics.orbit import OrbitalElements


@dataclass
class ManeuverResult:
    """Result of a maneuver calculation."""
    delta_v_km_s: float  # Required delta-V magnitude
    delta_v_vector: Vector3  # Delta-V vector (km/s)
    time_s: float  # Time of maneuver (s)
    description: str  # Maneuver description


def circular_velocity(r_km: float, mu: float = MU_EARTH_KM3_S2) -> float:
    """
    Calculate circular orbital velocity at given radius.
    
    Args:
        r_km: Orbital radius from Earth center (km)
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        Velocity in km/s
    """
    return math.sqrt(mu / r_km)


def hohmann_transfer(r1_km: float, r2_km: float, 
                    mu: float = MU_EARTH_KM3_S2) -> Tuple[float, float, float]:
    """
    Calculate Hohmann transfer between two circular orbits.
    
    Args:
        r1_km: Initial circular orbit radius (km)
        r2_km: Final circular orbit radius (km)
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        (delta_v1, delta_v2, transfer_time) in (km/s, km/s, seconds)
    """
    # Semi-major axis of transfer ellipse
    a_transfer = (r1_km + r2_km) / 2.0
    
    # Velocities
    v1_circular = circular_velocity(r1_km, mu)
    v2_circular = circular_velocity(r2_km, mu)
    
    # Transfer orbit velocities at periapsis and apoapsis
    v1_transfer = math.sqrt(mu * (2.0/r1_km - 1.0/a_transfer))
    v2_transfer = math.sqrt(mu * (2.0/r2_km - 1.0/a_transfer))
    
    # Delta-V for each burn
    delta_v1 = abs(v1_transfer - v1_circular)
    delta_v2 = abs(v2_circular - v2_transfer)
    
    # Transfer time (half period of transfer ellipse)
    transfer_time = math.pi * math.sqrt(a_transfer**3 / mu)
    
    return (delta_v1, delta_v2, transfer_time)


def bi_elliptic_transfer(r1_km: float, r2_km: float, rb_km: float,
                        mu: float = MU_EARTH_KM3_S2) -> Tuple[float, float, float, float]:
    """
    Calculate bi-elliptic transfer (can be more efficient than Hohmann for large ratio).
    
    Args:
        r1_km: Initial circular orbit radius (km)
        r2_km: Final circular orbit radius (km)
        rb_km: Intermediate apoapsis radius (km), must be > max(r1, r2)
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        (delta_v1, delta_v2, delta_v3, total_time) in (km/s, km/s, km/s, seconds)
    """
    # First transfer ellipse: r1 to rb
    a1 = (r1_km + rb_km) / 2.0
    v1_circ = circular_velocity(r1_km, mu)
    v1_transfer = math.sqrt(mu * (2.0/r1_km - 1.0/a1))
    delta_v1 = abs(v1_transfer - v1_circ)
    
    # At rb (apoapsis of first ellipse)
    v_rb1 = math.sqrt(mu * (2.0/rb_km - 1.0/a1))
    
    # Second transfer ellipse: rb to r2
    a2 = (rb_km + r2_km) / 2.0
    v_rb2 = math.sqrt(mu * (2.0/rb_km - 1.0/a2))
    delta_v2 = abs(v_rb2 - v_rb1)
    
    # At r2 (periapsis of second ellipse)
    v2_transfer = math.sqrt(mu * (2.0/r2_km - 1.0/a2))
    v2_circ = circular_velocity(r2_km, mu)
    delta_v3 = abs(v2_circ - v2_transfer)
    
    # Total time
    time1 = math.pi * math.sqrt(a1**3 / mu)
    time2 = math.pi * math.sqrt(a2**3 / mu)
    total_time = time1 + time2
    
    return (delta_v1, delta_v2, delta_v3, total_time)


def plane_change_delta_v(v_km_s: float, angle_rad: float) -> float:
    """
    Calculate delta-V required for a plane change maneuver.
    
    Args:
        v_km_s: Orbital velocity (km/s)
        angle_rad: Plane change angle (radians)
        
    Returns:
        Delta-V magnitude (km/s)
    """
    return 2.0 * v_km_s * math.sin(angle_rad / 2.0)


def combined_maneuver(v_km_s: float, delta_v_tangent: float, 
                     delta_v_normal: float) -> float:
    """
    Calculate delta-V for combined tangent and normal burn.
    
    Args:
        v_km_s: Current velocity (km/s)
        delta_v_tangent: Tangential delta-V component (km/s)
        delta_v_normal: Normal delta-V component (km/s)
        
    Returns:
        Total delta-V magnitude (km/s)
    """
    v_new_tangent = v_km_s + delta_v_tangent
    return math.sqrt(v_new_tangent**2 + delta_v_normal**2) - v_km_s


def apoapsis_raising_burn(r_km: float, target_apoapsis_km: float,
                         mu: float = MU_EARTH_KM3_S2) -> float:
    """
    Calculate delta-V to raise apoapsis from circular orbit.
    
    Args:
        r_km: Current circular orbit radius (km)
        target_apoapsis_km: Desired apoapsis radius (km)
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        Delta-V magnitude (km/s)
    """
    v_circular = circular_velocity(r_km, mu)
    
    # Semi-major axis of new ellipse
    a_new = (r_km + target_apoapsis_km) / 2.0
    
    # Velocity at periapsis of new orbit
    v_new = math.sqrt(mu * (2.0/r_km - 1.0/a_new))
    
    return v_new - v_circular


def periapsis_lowering_burn(r_km: float, target_periapsis_km: float,
                           mu: float = MU_EARTH_KM3_S2) -> float:
    """
    Calculate delta-V to lower periapsis from circular orbit.
    
    Args:
        r_km: Current circular orbit radius (km)
        target_periapsis_km: Desired periapsis radius (km)
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        Delta-V magnitude (km/s, negative for retrograde burn)
    """
    v_circular = circular_velocity(r_km, mu)
    
    # Semi-major axis of new ellipse
    a_new = (r_km + target_periapsis_km) / 2.0
    
    # Velocity at apoapsis of new orbit
    v_new = math.sqrt(mu * (2.0/r_km - 1.0/a_new))
    
    return v_new - v_circular  # Will be negative


@dataclass
class DeltaVBudget:
    """Track delta-V budget for a mission."""
    launch_to_orbit: float = 0.0  # km/s
    orbit_raising: float = 0.0
    plane_changes: float = 0.0
    rendezvous: float = 0.0
    station_keeping: float = 0.0
    deorbit: float = 0.0
    margin: float = 0.0  # Reserve
    
    def total_delta_v(self) -> float:
        """Calculate total delta-V budget."""
        return (self.launch_to_orbit + self.orbit_raising + 
                self.plane_changes + self.rendezvous + 
                self.station_keeping + self.deorbit + self.margin)
    
    def add_maneuver(self, category: str, delta_v: float):
        """Add a maneuver to the budget."""
        if category == "orbit_raising":
            self.orbit_raising += delta_v
        elif category == "plane_change":
            self.plane_changes += delta_v
        elif category == "rendezvous":
            self.rendezvous += delta_v
        elif category == "station_keeping":
            self.station_keeping += delta_v
        elif category == "deorbit":
            self.deorbit += delta_v


def lambert_solver_simple(r1: Vector3, r2: Vector3, tof_s: float,
                         mu: float = MU_EARTH_KM3_S2,
                         prograde: bool = True) -> Tuple[Vector3, Vector3]:
    """
    Simplified Lambert's problem solver for rendezvous.
    
    Finds the velocity vectors for a transfer between two positions
    in a given time of flight.
    
    Args:
        r1: Initial position vector (km)
        r2: Final position vector (km)
        tof_s: Time of flight (seconds)
        mu: Gravitational parameter (km³/s²)
        prograde: True for short way, False for long way
        
    Returns:
        (v1, v2) - velocity vectors at r1 and r2 (km/s)
    """
    # This is a simplified version - full Lambert solver is complex
    # Using Lagrange coefficients approximation
    
    r1_mag = math.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    r2_mag = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)
    
    # Cosine of transfer angle
    cos_dtheta = ((r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]) / 
                  (r1_mag * r2_mag))
    
    if not prograde:
        cos_dtheta = -cos_dtheta
    
    # Semi-parameter (approximation for parabolic transfer)
    A = math.sqrt(r1_mag * r2_mag * (1.0 + cos_dtheta))
    
    if abs(A) < 1e-10:
        # 180 degree transfer, not solvable with this method
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    
    # Lagrange coefficients
    c = math.sqrt(r1_mag**2 + r2_mag**2 - 2.0*r1_mag*r2_mag*cos_dtheta)
    s = (r1_mag + r2_mag + c) / 2.0
    
    # Semi-major axis (approximation)
    a = s / 2.0
    alpha = 2.0 * math.asin(math.sqrt(s / (2.0 * a)))
    beta = 2.0 * math.asin(math.sqrt((s - c) / (2.0 * a)))
    
    # Time of flight check
    tof_check = math.sqrt(a**3 / mu) * (alpha - beta - (math.sin(alpha) - math.sin(beta)))
    
    # Scale semi-major axis
    if abs(tof_check) > 1e-10:
        a_actual = a * (tof_s / tof_check)**(2.0/3.0)
    else:
        a_actual = a
    
    # Lagrange f and g functions
    f = 1.0 - r2_mag / a_actual * (1.0 - cos_dtheta)
    g = r1_mag * r2_mag * math.sqrt(1.0 - cos_dtheta) / math.sqrt(mu * a_actual)
    
    # Velocity at r1
    v1 = ((r2[0] - f*r1[0]) / g,
          (r2[1] - f*r1[1]) / g,
          (r2[2] - f*r1[2]) / g)
    
    # f_dot
    f_dot = math.sqrt(mu * a_actual) / (r1_mag * r2_mag) * math.sqrt(1.0 - cos_dtheta) * (
        (1.0 - cos_dtheta) / a_actual - 1.0/r1_mag - 1.0/r2_mag
    )
    
    # Velocity at r2
    v2 = ((f_dot*r2[0] - r1[0]) / g,
          (f_dot*r2[1] - r1[1]) / g,
          (f_dot*r2[2] - r1[2]) / g)
    
    return (v1, v2)


def compute_maneuver_sequence(initial_orbit: OrbitalElements,
                             final_orbit: OrbitalElements,
                             mu: float = MU_EARTH_KM3_S2) -> List[ManeuverResult]:
    """
    Compute a sequence of maneuvers to transfer between two orbits.
    
    Simplified approach: Hohmann transfer for altitude, then plane change.
    
    Args:
        initial_orbit: Starting orbital elements
        final_orbit: Target orbital elements
        mu: Gravitational parameter
        
    Returns:
        List of maneuver results
    """
    maneuvers = []
    
    # Altitude change (Hohmann transfer)
    if abs(initial_orbit.a_km - final_orbit.a_km) > 1.0:  # More than 1 km difference
        dv1, dv2, t_transfer = hohmann_transfer(initial_orbit.a_km, final_orbit.a_km, mu)
        
        maneuvers.append(ManeuverResult(
            delta_v_km_s=dv1,
            delta_v_vector=(dv1, 0.0, 0.0),  # Simplified
            time_s=0.0,
            description=f"Burn 1: Hohmann transfer insertion ({dv1:.3f} km/s)"
        ))
        
        maneuvers.append(ManeuverResult(
            delta_v_km_s=dv2,
            delta_v_vector=(dv2, 0.0, 0.0),  # Simplified
            time_s=t_transfer,
            description=f"Burn 2: Hohmann transfer circularization ({dv2:.3f} km/s)"
        ))
    
    # Plane change
    inc_change = abs(final_orbit.inc_rad - initial_orbit.inc_rad)
    if inc_change > 0.001:  # More than ~0.06 degrees
        v_final = circular_velocity(final_orbit.a_km, mu)
        dv_plane = plane_change_delta_v(v_final, inc_change)
        
        maneuvers.append(ManeuverResult(
            delta_v_km_s=dv_plane,
            delta_v_vector=(0.0, 0.0, dv_plane),  # Simplified
            time_s=0.0,  # Can be combined with circularization
            description=f"Plane change: {math.degrees(inc_change):.2f}° ({dv_plane:.3f} km/s)"
        ))
    
    return maneuvers
