"""
High-fidelity numerical propagators using Runge-Kutta integration.

Includes perturbation models for accurate orbit propagation.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple, Optional
from space_sim.core.frames import Vector3
from space_sim.core.constants import MU_EARTH_KM3_S2
from space_sim.physics.perturbations import total_perturbation_accel


def vector_add(v1: Vector3, v2: Vector3) -> Vector3:
    """Add two 3D vectors."""
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


def vector_scale(v: Vector3, scalar: float) -> Vector3:
    """Multiply vector by scalar."""
    return (v[0] * scalar, v[1] * scalar, v[2] * scalar)


def orbital_dynamics(r: Vector3, v: Vector3, 
                     mu: float = MU_EARTH_KM3_S2,
                     perturb_accel: Optional[Vector3] = None) -> Tuple[Vector3, Vector3]:
    """
    Compute the derivatives for orbital motion: dr/dt = v, dv/dt = -mu*r/|r|³ + perturbations.
    
    Args:
        r: Position vector (km)
        v: Velocity vector (km/s)
        mu: Gravitational parameter (km³/s²)
        perturb_accel: Additional perturbation acceleration (km/s²)
        
    Returns:
        (dr/dt, dv/dt)
    """
    # Position derivative
    dr_dt = v
    
    # Gravitational acceleration
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    
    if r_mag < 1.0:
        # Avoid singularity
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    
    grav_accel = (-mu / r_mag**3 * r[0], 
                  -mu / r_mag**3 * r[1], 
                  -mu / r_mag**3 * r[2])
    
    # Add perturbations
    if perturb_accel:
        dv_dt = vector_add(grav_accel, perturb_accel)
    else:
        dv_dt = grav_accel
    
    return (dr_dt, dv_dt)


def rk4_step(r: Vector3, v: Vector3, dt: float,
             mu: float = MU_EARTH_KM3_S2,
             compute_perturb: Optional[Callable[[Vector3, Vector3], Vector3]] = None) -> Tuple[Vector3, Vector3]:
    """
    Single Runge-Kutta 4th order integration step.
    
    Args:
        r: Position vector (km)
        v: Velocity vector (km/s)
        dt: Time step (s)
        mu: Gravitational parameter (km³/s²)
        compute_perturb: Function to compute perturbation acceleration
        
    Returns:
        (r_new, v_new) at time t + dt
    """
    # k1
    perturb1 = compute_perturb(r, v) if compute_perturb else None
    k1_r, k1_v = orbital_dynamics(r, v, mu, perturb1)
    
    # k2
    r2 = vector_add(r, vector_scale(k1_r, dt / 2.0))
    v2 = vector_add(v, vector_scale(k1_v, dt / 2.0))
    perturb2 = compute_perturb(r2, v2) if compute_perturb else None
    k2_r, k2_v = orbital_dynamics(r2, v2, mu, perturb2)
    
    # k3
    r3 = vector_add(r, vector_scale(k2_r, dt / 2.0))
    v3 = vector_add(v, vector_scale(k2_v, dt / 2.0))
    perturb3 = compute_perturb(r3, v3) if compute_perturb else None
    k3_r, k3_v = orbital_dynamics(r3, v3, mu, perturb3)
    
    # k4
    r4 = vector_add(r, vector_scale(k3_r, dt))
    v4 = vector_add(v, vector_scale(k3_v, dt))
    perturb4 = compute_perturb(r4, v4) if compute_perturb else None
    k4_r, k4_v = orbital_dynamics(r4, v4, mu, perturb4)
    
    # Combine
    r_new = vector_add(r, vector_scale(
        vector_add(vector_add(k1_r, vector_scale(k2_r, 2.0)),
                  vector_add(vector_scale(k3_r, 2.0), k4_r)),
        dt / 6.0
    ))
    
    v_new = vector_add(v, vector_scale(
        vector_add(vector_add(k1_v, vector_scale(k2_v, 2.0)),
                  vector_add(vector_scale(k3_v, 2.0), k4_v)),
        dt / 6.0
    ))
    
    return (r_new, v_new)


def propagate_with_perturbations(r0: Vector3, v0: Vector3, 
                                 t_start: float, t_end: float, 
                                 dt: float = 60.0,
                                 include_j2: bool = True,
                                 include_j3: bool = False,
                                 include_drag: bool = False,
                                 sat_mass_kg: float = 1000.0,
                                 sat_area_m2: float = 10.0,
                                 sat_cd: float = 2.2,
                                 mu: float = MU_EARTH_KM3_S2) -> list[Tuple[float, Vector3, Vector3]]:
    """
    Propagate orbit with perturbations using RK4 integration.
    
    Args:
        r0: Initial position (km)
        v0: Initial velocity (km/s)
        t_start: Start time (s)
        t_end: End time (s)
        dt: Integration time step (s)
        include_j2: Include J2 perturbation
        include_j3: Include J3 perturbation
        include_drag: Include atmospheric drag
        sat_mass_kg: Satellite mass (kg)
        sat_area_m2: Cross-sectional area (m²)
        sat_cd: Drag coefficient
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        List of (time, position, velocity) tuples
    """
    trajectory = []
    
    r = r0
    v = v0
    t = t_start
    
    trajectory.append((t, r, v))
    
    # Perturbation function closure
    def compute_perturb(pos: Vector3, vel: Vector3) -> Vector3:
        return total_perturbation_accel(
            pos, vel,
            include_j2=include_j2,
            include_j3=include_j3,
            include_drag=include_drag,
            include_srp=False,
            include_sun=False,
            include_moon=False,
            sat_mass_kg=sat_mass_kg,
            sat_area_m2=sat_area_m2,
            sat_cd=sat_cd
        )
    
    # Integration loop
    while t < t_end:
        step_size = min(dt, t_end - t)
        r, v = rk4_step(r, v, step_size, mu, compute_perturb)
        t += step_size
        trajectory.append((t, r, v))
    
    return trajectory


class NumericalPropagator:
    """
    High-fidelity numerical orbit propagator with perturbations.
    """
    
    def __init__(self, 
                 include_j2: bool = True,
                 include_j3: bool = False,
                 include_drag: bool = False,
                 dt: float = 60.0,
                 sat_mass_kg: float = 1000.0,
                 sat_area_m2: float = 10.0,
                 sat_cd: float = 2.2):
        """
        Initialize propagator with perturbation settings.
        
        Args:
            include_j2: Include J2 perturbation (Earth oblateness)
            include_j3: Include J3 perturbation
            include_drag: Include atmospheric drag
            dt: Integration time step (seconds)
            sat_mass_kg: Satellite mass (kg)
            sat_area_m2: Cross-sectional area (m²)
            sat_cd: Drag coefficient
        """
        self.include_j2 = include_j2
        self.include_j3 = include_j3
        self.include_drag = include_drag
        self.dt = dt
        self.sat_mass_kg = sat_mass_kg
        self.sat_area_m2 = sat_area_m2
        self.sat_cd = sat_cd
    
    def propagate(self, r0: Vector3, v0: Vector3, 
                 duration: float) -> list[Tuple[float, Vector3, Vector3]]:
        """
        Propagate from initial state for given duration.
        
        Args:
            r0: Initial position in ECI (km)
            v0: Initial velocity in ECI (km/s)
            duration: Propagation duration (s)
            
        Returns:
            List of (time, position, velocity) tuples
        """
        return propagate_with_perturbations(
            r0, v0, 0.0, duration, self.dt,
            self.include_j2, self.include_j3, self.include_drag,
            self.sat_mass_kg, self.sat_area_m2, self.sat_cd
        )
