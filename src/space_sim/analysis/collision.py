"""
Collision detection and avoidance analysis.

Includes:
- Close approach detection
- Conjunction analysis
- Probability of collision
- Miss distance calculations
- Collision risk assessment
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from space_sim.core.frames import Vector3


@dataclass
class Conjunction:
    """Information about a close approach event."""
    time_s: float  # Time of closest approach (s)
    miss_distance_km: float  # Minimum separation distance (km)
    relative_velocity_km_s: float  # Relative velocity magnitude (km/s)
    position_primary: Vector3  # Position of primary object at TCA
    position_secondary: Vector3  # Position of secondary object at TCA
    probability_of_collision: float = 0.0  # Pc if calculated
    risk_level: str = "unknown"  # "high", "medium", "low"


def relative_distance(r1: Vector3, r2: Vector3) -> float:
    """
    Calculate distance between two position vectors.
    
    Args:
        r1: First position vector (km)
        r2: Second position vector (km)
        
    Returns:
        Distance in km
    """
    dx = r2[0] - r1[0]
    dy = r2[1] - r1[1]
    dz = r2[2] - r1[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def relative_velocity_magnitude(v1: Vector3, v2: Vector3) -> float:
    """
    Calculate relative velocity magnitude.
    
    Args:
        v1: First velocity vector (km/s)
        v2: Second velocity vector (km/s)
        
    Returns:
        Relative velocity magnitude in km/s
    """
    dvx = v2[0] - v1[0]
    dvy = v2[1] - v1[1]
    dvz = v2[2] - v1[2]
    return math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)


def find_closest_approach(trajectory1: List[Tuple[float, Vector3, Vector3]],
                         trajectory2: List[Tuple[float, Vector3, Vector3]],
                         threshold_km: float = 10.0) -> Optional[Conjunction]:
    """
    Find the closest approach between two trajectories.
    
    Args:
        trajectory1: List of (time, position, velocity) for object 1
        trajectory2: List of (time, position, velocity) for object 2
        threshold_km: Only report if closer than this distance
        
    Returns:
        Conjunction object if close approach found, None otherwise
    """
    if not trajectory1 or not trajectory2:
        return None
    
    min_distance = float('inf')
    tca_time = 0.0
    tca_pos1 = (0.0, 0.0, 0.0)
    tca_pos2 = (0.0, 0.0, 0.0)
    tca_vel1 = (0.0, 0.0, 0.0)
    tca_vel2 = (0.0, 0.0, 0.0)
    
    # Find common time points
    # Assumes both trajectories span similar time range
    for t1, r1, v1 in trajectory1:
        # Find closest time point in trajectory2
        for t2, r2, v2 in trajectory2:
            if abs(t1 - t2) < 1.0:  # Within 1 second
                distance = relative_distance(r1, r2)
                
                if distance < min_distance:
                    min_distance = distance
                    tca_time = t1
                    tca_pos1 = r1
                    tca_pos2 = r2
                    tca_vel1 = v1
                    tca_vel2 = v2
    
    if min_distance > threshold_km:
        return None
    
    rel_vel = relative_velocity_magnitude(tca_vel1, tca_vel2)
    
    # Assess risk level
    if min_distance < 1.0:
        risk = "high"
    elif min_distance < 5.0:
        risk = "medium"
    else:
        risk = "low"
    
    return Conjunction(
        time_s=tca_time,
        miss_distance_km=min_distance,
        relative_velocity_km_s=rel_vel,
        position_primary=tca_pos1,
        position_secondary=tca_pos2,
        risk_level=risk
    )


def probability_of_collision_2d(miss_distance_km: float,
                               combined_radius_km: float,
                               position_uncertainty_km: float) -> float:
    """
    Calculate probability of collision using 2D circular model.
    
    Simplified Pc calculation assuming circular cross-section.
    
    Args:
        miss_distance_km: Predicted miss distance (km)
        combined_radius_km: Sum of object radii (km)
        position_uncertainty_km: 1-sigma position uncertainty (km)
        
    Returns:
        Probability of collision (0-1)
    """
    if position_uncertainty_km < 1e-10:
        # No uncertainty, deterministic
        return 1.0 if miss_distance_km <= combined_radius_km else 0.0
    
    # Ratio of miss distance to uncertainty
    x = miss_distance_km / position_uncertainty_km
    
    # Hard body radius in sigma units
    r_sigma = combined_radius_km / position_uncertainty_km
    
    # 2D circular collision probability
    if x > 10.0 * r_sigma:
        return 0.0  # Negligible probability
    
    # Approximate using complementary error function
    # Pc ≈ exp(-x²/2) for small r
    pc = math.exp(-(x**2) / (2.0 * r_sigma**2))
    
    return min(pc, 1.0)


def conjunction_analysis(r1: Vector3, v1: Vector3,
                        r2: Vector3, v2: Vector3,
                        combined_radius_km: float = 0.01,
                        position_uncertainty_km: float = 1.0) -> dict:
    """
    Perform detailed conjunction analysis at a given time.
    
    Args:
        r1: Position of object 1 (km)
        v1: Velocity of object 1 (km/s)
        r2: Position of object 2 (km)
        v2: Velocity of object 2 (km/s)
        combined_radius_km: Sum of object hard body radii (km)
        position_uncertainty_km: Position uncertainty (km)
        
    Returns:
        Dictionary with conjunction metrics
    """
    # Relative position and velocity
    dr = (r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2])
    dv = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    
    # Current separation
    current_separation = math.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
    
    # Relative velocity magnitude
    rel_vel = math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
    
    # Time to closest approach (assuming linear motion)
    if rel_vel > 1e-10:
        # dot(dr, dv)
        dr_dot_dv = dr[0]*dv[0] + dr[1]*dv[1] + dr[2]*dv[2]
        time_to_tca = -dr_dot_dv / (rel_vel**2)
    else:
        time_to_tca = 0.0
    
    # Position at TCA
    if time_to_tca > 0:
        r1_tca = (r1[0] + v1[0]*time_to_tca,
                  r1[1] + v1[1]*time_to_tca,
                  r1[2] + v1[2]*time_to_tca)
        r2_tca = (r2[0] + v2[0]*time_to_tca,
                  r2[1] + v2[1]*time_to_tca,
                  r2[2] + v2[2]*time_to_tca)
        miss_distance = relative_distance(r1_tca, r2_tca)
    else:
        miss_distance = current_separation
    
    # Probability of collision
    pc = probability_of_collision_2d(miss_distance, combined_radius_km,
                                    position_uncertainty_km)
    
    # Radial and tangential components
    if current_separation > 1e-10:
        radial_vel = (dr[0]*dv[0] + dr[1]*dv[1] + dr[2]*dv[2]) / current_separation
    else:
        radial_vel = 0.0
    
    tangential_vel = math.sqrt(max(0.0, rel_vel**2 - radial_vel**2))
    
    return {
        "current_separation_km": current_separation,
        "miss_distance_km": miss_distance,
        "time_to_tca_s": time_to_tca,
        "relative_velocity_km_s": rel_vel,
        "radial_velocity_km_s": radial_vel,
        "tangential_velocity_km_s": tangential_vel,
        "probability_of_collision": pc,
        "collision_risk": "high" if pc > 1e-4 else "medium" if pc > 1e-6 else "low"
    }


def screen_catalog_for_conjunctions(target_trajectory: List[Tuple[float, Vector3, Vector3]],
                                   catalog_trajectories: List[List[Tuple[float, Vector3, Vector3]]],
                                   threshold_km: float = 5.0) -> List[Conjunction]:
    """
    Screen a catalog of objects for potential conjunctions with target.
    
    Args:
        target_trajectory: Trajectory of target satellite
        catalog_trajectories: List of trajectories for catalog objects
        threshold_km: Alert threshold distance (km)
        
    Returns:
        List of Conjunction objects for close approaches
    """
    conjunctions = []
    
    for i, catalog_traj in enumerate(catalog_trajectories):
        conj = find_closest_approach(target_trajectory, catalog_traj, threshold_km)
        if conj:
            conjunctions.append(conj)
    
    # Sort by miss distance
    conjunctions.sort(key=lambda c: c.miss_distance_km)
    
    return conjunctions


@dataclass
class CollisionAvoidanceManeuver:
    """Recommended maneuver to avoid collision."""
    delta_v_km_s: float  # Magnitude of maneuver
    delta_v_vector: Vector3  # Direction and magnitude
    maneuver_time_s: float  # When to execute maneuver
    resulting_miss_distance_km: float  # Miss distance after maneuver
    description: str  # Human-readable description


def compute_avoidance_maneuver(conjunction: Conjunction,
                              maneuver_lead_time_s: float = 3600.0,
                              desired_miss_km: float = 10.0) -> CollisionAvoidanceManeuver:
    """
    Compute a simple collision avoidance maneuver.
    
    Args:
        conjunction: The conjunction to avoid
        maneuver_lead_time_s: How long before TCA to maneuver (seconds)
        desired_miss_km: Target miss distance after maneuver (km)
        
    Returns:
        Recommended collision avoidance maneuver
    """
    # Required displacement at TCA
    required_displacement = desired_miss_km - conjunction.miss_distance_km
    
    if required_displacement <= 0:
        # Already safe
        return CollisionAvoidanceManeuver(
            delta_v_km_s=0.0,
            delta_v_vector=(0.0, 0.0, 0.0),
            maneuver_time_s=conjunction.time_s - maneuver_lead_time_s,
            resulting_miss_distance_km=conjunction.miss_distance_km,
            description="No maneuver required"
        )
    
    # Simple out-of-plane maneuver (most efficient)
    # Δv = displacement / time_to_tca
    maneuver_time = conjunction.time_s - maneuver_lead_time_s
    
    if maneuver_lead_time_s > 0:
        delta_v_magnitude = required_displacement / maneuver_lead_time_s
    else:
        delta_v_magnitude = required_displacement / 60.0  # Minimum 1 minute
    
    # Direction perpendicular to relative velocity (out of plane)
    # Simplified: use normal direction
    delta_v_vector = (0.0, 0.0, delta_v_magnitude)
    
    return CollisionAvoidanceManeuver(
        delta_v_km_s=delta_v_magnitude,
        delta_v_vector=delta_v_vector,
        maneuver_time_s=maneuver_time,
        resulting_miss_distance_km=desired_miss_km,
        description=f"Out-of-plane CAM: {delta_v_magnitude*1000:.1f} m/s, "
                   f"{maneuver_lead_time_s/3600:.1f} hours before TCA"
    )


def is_collision_imminent(current_distance_km: float,
                         relative_velocity_km_s: float,
                         warning_time_s: float = 3600.0) -> bool:
    """
    Determine if a collision is imminent based on current state.
    
    Args:
        current_distance_km: Current separation (km)
        relative_velocity_km_s: Relative velocity (km/s)
        warning_time_s: Warning threshold time (seconds)
        
    Returns:
        True if collision possible within warning time
    """
    if relative_velocity_km_s < 1e-10:
        return False
    
    # Time to collision if on collision course
    time_to_contact = current_distance_km / relative_velocity_km_s
    
    return time_to_contact < warning_time_s
