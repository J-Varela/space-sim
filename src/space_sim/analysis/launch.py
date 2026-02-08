"""
Launch window calculations and mission planning tools.

Includes:
- Launch azimuth calculations
- Launch window determination
- Ground track prediction
- Target plane intersection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from space_sim.core.constants import OMEGA_EARTH_RAD_S, R_EARTH_KM
from space_sim.core.frames import Vector3


@dataclass
class LaunchSite:
    """Launch site location."""
    name: str
    latitude_deg: float  # Geodetic latitude (degrees)
    longitude_deg: float  # Longitude (degrees, East positive)
    altitude_km: float = 0.0  # Altitude above sea level (km)


@dataclass
class LaunchWindow:
    """Information about a launch opportunity."""
    open_time_s: float  # Window opens (seconds from epoch)
    close_time_s: float  # Window closes (seconds)
    duration_s: float  # Window duration (seconds)
    launch_azimuth_deg: float  # Required launch azimuth (degrees from North)
    inclination_deg: float  # Resulting inclination (degrees)
    description: str  # Human-readable description


def launch_azimuth(site_latitude_deg: float, target_inclination_deg: float,
                  ascending: bool = True) -> Optional[float]:
    """
    Calculate launch azimuth for a given target inclination.
    
    Args:
        site_latitude_deg: Launch site latitude (degrees)
        target_inclination_deg: Desired orbital inclination (degrees)
        ascending: True for ascending node, False for descending
        
    Returns:
        Launch azimuth in degrees from North (0-360), or None if not achievable
    """
    lat_rad = math.radians(site_latitude_deg)
    inc_rad = math.radians(target_inclination_deg)
    
    # Check if inclination is achievable from this latitude
    if abs(site_latitude_deg) > target_inclination_deg:
        return None  # Cannot achieve inclination less than latitude
    
    # Launch azimuth formula
    # sin(Az) = cos(i) / cos(lat)
    cos_i = math.cos(inc_rad)
    cos_lat = math.cos(lat_rad)
    
    if abs(cos_lat) < 1e-10:
        # At poles
        if ascending:
            return 0.0  # North
        else:
            return 180.0  # South
    
    sin_az = cos_i / cos_lat
    
    # Check if achievable
    if abs(sin_az) > 1.0:
        return None
    
    az_rad = math.asin(sin_az)
    
    # Convert to azimuth from North
    if ascending:
        # Eastward launch
        if site_latitude_deg >= 0:
            # Northern hemisphere
            azimuth_deg = 90.0 - math.degrees(az_rad)
        else:
            # Southern hemisphere
            azimuth_deg = 90.0 + math.degrees(az_rad)
    else:
        # Westward launch (retrograde)
        if site_latitude_deg >= 0:
            azimuth_deg = 270.0 + math.degrees(az_rad)
        else:
            azimuth_deg = 270.0 - math.degrees(az_rad)
    
    # Normalize to 0-360
    azimuth_deg = azimuth_deg % 360.0
    
    return azimuth_deg


def plane_intersection_time(site_longitude_deg: float, target_raan_deg: float,
                           current_time_s: float = 0.0) -> float:
    """
    Calculate when the launch site rotates into the target orbital plane.
    
    Args:
        site_longitude_deg: Launch site longitude (degrees East)
        target_raan_deg: Target Right Ascension of Ascending Node (degrees)
        current_time_s: Current time (seconds from epoch)
        
    Returns:
        Time when site aligns with orbital plane (seconds from epoch)
    """
    # Current longitude accounts for Earth rotation
    earth_rotation_deg = math.degrees(OMEGA_EARTH_RAD_S * current_time_s)
    current_longitude = (site_longitude_deg + earth_rotation_deg) % 360.0
    
    # Angular difference to target RAAN
    delta_longitude = (target_raan_deg - current_longitude) % 360.0
    
    # Time to rotate to alignment
    rotation_rate_deg_s = math.degrees(OMEGA_EARTH_RAD_S)
    time_to_alignment = delta_longitude / rotation_rate_deg_s
    
    return current_time_s + time_to_alignment


def calculate_launch_windows(site: LaunchSite, 
                            target_inclination_deg: float,
                            target_raan_deg: Optional[float] = None,
                            search_duration_s: float = 86400.0,
                            start_time_s: float = 0.0) -> List[LaunchWindow]:
    """
    Calculate launch windows for a given target orbit.
    
    Args:
        site: Launch site location
        target_inclination_deg: Target orbital inclination (degrees)
        target_raan_deg: Target RAAN (None for any RAAN)
        search_duration_s: Time period to search (seconds)
        start_time_s: Start of search period (seconds from epoch)
        
    Returns:
        List of launch windows
    """
    windows = []
    
    # Calculate launch azimuth
    az_ascending = launch_azimuth(site.latitude_deg, target_inclination_deg, True)
    az_descending = launch_azimuth(site.latitude_deg, target_inclination_deg, False)
    
    if az_ascending is None and az_descending is None:
        # Inclination not achievable
        return windows
    
    if target_raan_deg is None:
        # Any RAAN acceptable - launch anytime
        # Two opportunities per day (ascending and descending)
        
        # One sidereal day
        sidereal_day_s = 86164.0905
        half_day = sidereal_day_s / 2.0
        
        if az_ascending is not None:
            opportunities = int(search_duration_s / half_day) + 1
            for i in range(opportunities):
                window_time = start_time_s + i * half_day
                if window_time <= start_time_s + search_duration_s:
                    windows.append(LaunchWindow(
                        open_time_s=window_time,
                        close_time_s=window_time + 600.0,  # 10 minute window
                        duration_s=600.0,
                        launch_azimuth_deg=az_ascending,
                        inclination_deg=target_inclination_deg,
                        description=f"Ascending node launch at {az_ascending:.1f}°"
                    ))
        
        if az_descending is not None:
            opportunities = int(search_duration_s / half_day) + 1
            for i in range(opportunities):
                window_time = start_time_s + i * half_day + half_day/2.0
                if window_time <= start_time_s + search_duration_s:
                    windows.append(LaunchWindow(
                        open_time_s=window_time,
                        close_time_s=window_time + 600.0,
                        duration_s=600.0,
                        launch_azimuth_deg=az_descending,
                        inclination_deg=target_inclination_deg,
                        description=f"Descending node launch at {az_descending:.1f}°"
                    ))
    else:
        # Specific RAAN required
        # Find when site rotates into orbital plane
        
        sidereal_day_s = 86164.0905
        
        # Check for ascending node launches
        if az_ascending is not None:
            current_time = start_time_s
            while current_time < start_time_s + search_duration_s:
                window_time = plane_intersection_time(site.longitude_deg, 
                                                     target_raan_deg, current_time)
                
                if start_time_s <= window_time <= start_time_s + search_duration_s:
                    windows.append(LaunchWindow(
                        open_time_s=window_time - 300.0,  # 5 min before
                        close_time_s=window_time + 300.0,  # 5 min after
                        duration_s=600.0,
                        launch_azimuth_deg=az_ascending,
                        inclination_deg=target_inclination_deg,
                        description=f"Ascending RAAN={target_raan_deg:.1f}° at Az={az_ascending:.1f}°"
                    ))
                
                current_time += sidereal_day_s
        
        # Check for descending node launches (RAAN + 180°)
        if az_descending is not None:
            target_raan_descending = (target_raan_deg + 180.0) % 360.0
            current_time = start_time_s
            while current_time < start_time_s + search_duration_s:
                window_time = plane_intersection_time(site.longitude_deg,
                                                     target_raan_descending, current_time)
                
                if start_time_s <= window_time <= start_time_s + search_duration_s:
                    windows.append(LaunchWindow(
                        open_time_s=window_time - 300.0,
                        close_time_s=window_time + 300.0,
                        duration_s=600.0,
                        launch_azimuth_deg=az_descending,
                        inclination_deg=target_inclination_deg,
                        description=f"Descending RAAN={target_raan_descending:.1f}° at Az={az_descending:.1f}°"
                    ))
                
                current_time += sidereal_day_s
    
    # Sort by time
    windows.sort(key=lambda w: w.open_time_s)
    
    return windows


def ground_track_longitude(site_longitude_deg: float, elapsed_time_s: float) -> float:
    """
    Calculate ground track longitude after Earth rotation.
    
    Args:
        site_longitude_deg: Initial longitude (degrees East)
        elapsed_time_s: Time elapsed (seconds)
        
    Returns:
        New longitude (degrees East, 0-360)
    """
    rotation_deg = math.degrees(OMEGA_EARTH_RAD_S * elapsed_time_s)
    new_longitude = (site_longitude_deg - rotation_deg) % 360.0
    return new_longitude


def orbit_ground_track(inclination_rad: float, raan_rad: float,
                      argp_rad: float, true_anomaly_rad: float,
                      elapsed_time_s: float) -> Tuple[float, float]:
    """
    Calculate ground track latitude and longitude for an orbital position.
    
    Args:
        inclination_rad: Orbital inclination (radians)
        raan_rad: RAAN (radians)
        argp_rad: Argument of perigee (radians)
        true_anomaly_rad: True anomaly (radians)
        elapsed_time_s: Time since epoch (seconds)
        
    Returns:
        (latitude_deg, longitude_deg) tuple
    """
    # Argument of latitude
    u = argp_rad + true_anomaly_rad
    
    # Latitude
    lat_rad = math.asin(math.sin(inclination_rad) * math.sin(u))
    lat_deg = math.degrees(lat_rad)
    
    # Longitude (inertial)
    lon_inertial_rad = raan_rad + math.atan2(
        math.cos(inclination_rad) * math.sin(u),
        math.cos(u)
    )
    
    # Account for Earth rotation
    earth_rotation_rad = OMEGA_EARTH_RAD_S * elapsed_time_s
    lon_rad = lon_inertial_rad - earth_rotation_rad
    lon_deg = (math.degrees(lon_rad) + 360.0) % 360.0
    
    # Convert to -180 to 180
    if lon_deg > 180.0:
        lon_deg -= 360.0
    
    return (lat_deg, lon_deg)


# Common launch sites
LAUNCH_SITES = {
    "Kennedy Space Center": LaunchSite("Kennedy Space Center", 28.573, -80.649),
    "Cape Canaveral": LaunchSite("Cape Canaveral", 28.562, -80.577),
    "Vandenberg": LaunchSite("Vandenberg SFB", 34.632, -120.611),
    "Baikonur": LaunchSite("Baikonur Cosmodrome", 45.965, 63.305),
    "Kourou": LaunchSite("Guiana Space Centre", 5.239, -52.768),
    "Jiuquan": LaunchSite("Jiuquan Satellite Launch Center", 40.958, 100.292),
    "Tanegashima": LaunchSite("Tanegashima Space Center", 30.391, 130.975),
    "Plesetsk": LaunchSite("Plesetsk Cosmodrome", 62.958, 40.577),
}


def example_launch_window_calculation():
    """Example usage of launch window calculations."""
    site = LAUNCH_SITES["Kennedy Space Center"]
    
    print(f"Launch Site: {site.name}")
    print(f"Location: {site.latitude_deg:.3f}°N, {abs(site.longitude_deg):.3f}°W")
    print()
    
    # ISS-like orbit (51.6° inclination)
    target_inc = 51.6
    
    print(f"Target Inclination: {target_inc}°")
    
    # Calculate azimuths
    az_asc = launch_azimuth(site.latitude_deg, target_inc, True)
    az_desc = launch_azimuth(site.latitude_deg, target_inc, False)
    
    print(f"Launch Azimuth (Ascending): {az_asc:.2f}° from North")
    print(f"Launch Azimuth (Descending): {az_desc:.2f}° from North")
    print()
    
    # Calculate windows for next 3 days
    windows = calculate_launch_windows(site, target_inc, search_duration_s=3*86400.0)
    
    print(f"Launch Windows (next 3 days): {len(windows)}")
    for i, window in enumerate(windows[:10]):  # Show first 10
        hours = window.open_time_s / 3600.0
        print(f"  {i+1}. T+{hours:.2f}h: {window.description}")


if __name__ == "__main__":
    example_launch_window_calculation()
