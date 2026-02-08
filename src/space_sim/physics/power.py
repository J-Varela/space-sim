"""
Spacecraft power system modeling.

Includes:
- Solar panel power generation
- Battery charge/discharge
- Eclipse calculations
- Power budget management
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from space_sim.core.frames import Vector3
from space_sim.core.constants import R_EARTH_KM


# Solar flux at 1 AU (W/m²)
SOLAR_FLUX_1AU = 1361.0


@dataclass
class SolarPanel:
    """Solar panel specifications."""
    area_m2: float  # Panel area
    efficiency: float  # Conversion efficiency (0-1)
    sun_angle_deg: float = 0.0  # Angle from sun vector (degrees)
    degradation: float = 1.0  # Degradation factor (1.0 = new, <1.0 = aged)
    
    def power_output_w(self, eclipse: bool = False, 
                      sun_direction: Optional[Vector3] = None,
                      panel_normal: Optional[Vector3] = None) -> float:
        """
        Calculate solar panel power output.
        
        Args:
            eclipse: Whether satellite is in Earth's shadow
            sun_direction: Unit vector pointing to Sun
            panel_normal: Unit vector normal to panel surface
            
        Returns:
            Power output in Watts
        """
        if eclipse:
            return 0.0
        
        # Cosine loss if angles provided
        cos_angle = 1.0
        if sun_direction and panel_normal:
            # Dot product of unit vectors
            dot = (sun_direction[0] * panel_normal[0] + 
                   sun_direction[1] * panel_normal[1] + 
                   sun_direction[2] * panel_normal[2])
            cos_angle = max(0.0, dot)  # Only positive angles contribute
        
        # Power = Solar Flux × Area × Efficiency × cos(angle) × degradation
        power = (SOLAR_FLUX_1AU * self.area_m2 * self.efficiency * 
                cos_angle * self.degradation)
        
        return power


@dataclass
class Battery:
    """Battery specifications and state."""
    capacity_wh: float  # Total capacity in Watt-hours
    current_charge_wh: float  # Current charge level
    charge_efficiency: float = 0.95  # Charging efficiency
    discharge_efficiency: float = 0.95  # Discharging efficiency
    max_charge_rate_w: float = 1000.0  # Maximum charge rate (W)
    max_discharge_rate_w: float = 1000.0  # Maximum discharge rate (W)
    depth_of_discharge_limit: float = 0.2  # Minimum state of charge (0-1)
    
    def state_of_charge(self) -> float:
        """Return battery state of charge (0-1)."""
        return self.current_charge_wh / self.capacity_wh
    
    def charge(self, power_w: float, dt_s: float) -> float:
        """
        Charge battery with available power.
        
        Args:
            power_w: Available charging power (W)
            dt_s: Time step (seconds)
            
        Returns:
            Actual power consumed for charging (W)
        """
        if self.current_charge_wh >= self.capacity_wh:
            return 0.0  # Battery full
        
        # Limit charge rate
        actual_power = min(power_w, self.max_charge_rate_w)
        
        # Energy added (considering efficiency)
        energy_wh = (actual_power * dt_s / 3600.0) * self.charge_efficiency
        
        # Don't overcharge
        available_capacity = self.capacity_wh - self.current_charge_wh
        energy_added = min(energy_wh, available_capacity)
        
        self.current_charge_wh += energy_added
        
        # Return actual power consumed from source
        return energy_added / (dt_s / 3600.0) / self.charge_efficiency
    
    def discharge(self, power_demand_w: float, dt_s: float) -> float:
        """
        Discharge battery to meet power demand.
        
        Args:
            power_demand_w: Required power (W)
            dt_s: Time step (seconds)
            
        Returns:
            Actual power delivered (may be less than demand if depleted)
        """
        min_charge = self.capacity_wh * self.depth_of_discharge_limit
        available_charge = self.current_charge_wh - min_charge
        
        if available_charge <= 0:
            return 0.0  # Battery at minimum charge
        
        # Limit discharge rate
        max_deliverable_power = min(power_demand_w, self.max_discharge_rate_w)
        
        # Energy needed
        energy_needed_wh = max_deliverable_power * dt_s / 3600.0
        
        # Account for discharge efficiency
        energy_from_battery = energy_needed_wh / self.discharge_efficiency
        
        # Don't over-discharge
        actual_energy = min(energy_from_battery, available_charge)
        
        self.current_charge_wh -= actual_energy
        
        # Return actual power delivered
        return actual_energy * self.discharge_efficiency / (dt_s / 3600.0)


def is_in_eclipse(r_sat_eci: Vector3, r_sun_eci: Vector3) -> bool:
    """
    Determine if satellite is in Earth's shadow (umbra).
    
    Uses simple cylindrical shadow model.
    
    Args:
        r_sat_eci: Satellite position in ECI (km)
        r_sun_eci: Sun position in ECI (km)
        
    Returns:
        True if in eclipse, False otherwise
    """
    # Vector from Sun to satellite
    dx = r_sat_eci[0] - r_sun_eci[0]
    dy = r_sat_eci[1] - r_sun_eci[1]
    dz = r_sat_eci[2] - r_sun_eci[2]
    
    # Normalize sun direction
    sun_dist = math.sqrt(r_sun_eci[0]**2 + r_sun_eci[1]**2 + r_sun_eci[2]**2)
    if sun_dist < 1e-10:
        return False
    
    sun_dir = (r_sun_eci[0]/sun_dist, r_sun_eci[1]/sun_dist, r_sun_eci[2]/sun_dist)
    
    # Check if satellite is on night side of Earth
    sat_sun_dot = (r_sat_eci[0]*sun_dir[0] + r_sat_eci[1]*sun_dir[1] + 
                   r_sat_eci[2]*sun_dir[2])
    
    if sat_sun_dot > 0:
        return False  # Satellite on day side
    
    # Project satellite position onto plane perpendicular to sun direction
    # Distance from shadow axis
    proj_along_sun = sat_sun_dot
    perp_x = r_sat_eci[0] - proj_along_sun * sun_dir[0]
    perp_y = r_sat_eci[1] - proj_along_sun * sun_dir[1]
    perp_z = r_sat_eci[2] - proj_along_sun * sun_dir[2]
    
    perp_dist = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
    
    # In eclipse if within Earth's shadow cylinder
    return perp_dist < R_EARTH_KM


@dataclass
class PowerBudget:
    """Track power consumption by subsystem."""
    payload_w: float = 0.0
    communication_w: float = 0.0
    attitude_control_w: float = 0.0
    thermal_w: float = 0.0
    avionics_w: float = 10.0  # Always-on baseline
    
    def total_consumption_w(self) -> float:
        """Return total power consumption."""
        return (self.payload_w + self.communication_w + 
                self.attitude_control_w + self.thermal_w + self.avionics_w)


@dataclass
class PowerSystem:
    """
    Complete power system for a spacecraft.
    """
    solar_panel: SolarPanel
    battery: Battery
    power_budget: PowerBudget
    
    def step(self, dt_s: float, r_sat_eci: Vector3, 
            r_sun_eci: Vector3,
            panel_normal: Optional[Vector3] = None) -> dict:
        """
        Update power system state for one time step.
        
        Args:
            dt_s: Time step (seconds)
            r_sat_eci: Satellite position in ECI (km)
            r_sun_eci: Sun position in ECI (km)
            panel_normal: Solar panel normal vector (None = always optimal)
            
        Returns:
            Dictionary with power system status
        """
        # Check eclipse
        eclipse = is_in_eclipse(r_sat_eci, r_sun_eci)
        
        # Compute sun direction
        dx = r_sun_eci[0] - r_sat_eci[0]
        dy = r_sun_eci[1] - r_sat_eci[1]
        dz = r_sun_eci[2] - r_sat_eci[2]
        sun_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if sun_dist > 1e-10:
            sun_dir = (dx/sun_dist, dy/sun_dist, dz/sun_dist)
        else:
            sun_dir = (1.0, 0.0, 0.0)
        
        # Solar power generation
        solar_power = self.solar_panel.power_output_w(eclipse, sun_dir, panel_normal)
        
        # Power consumption
        power_demand = self.power_budget.total_consumption_w()
        
        # Power balance
        net_power = solar_power - power_demand
        
        if net_power > 0:
            # Excess power, charge battery
            charged_power = self.battery.charge(net_power, dt_s)
            available_power = solar_power
            battery_power = -charged_power  # Negative = charging
            power_deficit = 0.0
        else:
            # Deficit, discharge battery
            deficit = abs(net_power)
            delivered_power = self.battery.discharge(deficit, dt_s)
            available_power = solar_power + delivered_power
            battery_power = delivered_power  # Positive = discharging
            power_deficit = deficit - delivered_power  # Unmet demand
        
        return {
            "eclipse": eclipse,
            "solar_power_w": solar_power,
            "power_demand_w": power_demand,
            "battery_soc": self.battery.state_of_charge(),
            "battery_charge_wh": self.battery.current_charge_wh,
            "battery_power_w": battery_power,
            "available_power_w": available_power,
            "power_deficit_w": power_deficit,
        }


def simple_sun_position_eci(julian_day: float) -> Vector3:
    """
    Simple approximate Sun position in ECI frame.
    
    Args:
        julian_day: Julian day number
        
    Returns:
        Sun position vector in ECI (km)
    """
    # Simplified model - Sun in ecliptic plane
    # Days since J2000.0
    d = julian_day - 2451545.0
    
    # Mean longitude
    L = 280.460 + 0.9856474 * d
    L_rad = math.radians(L % 360.0)
    
    # Mean anomaly
    g = 357.528 + 0.9856003 * d
    g_rad = math.radians(g % 360.0)
    
    # Ecliptic longitude
    lambda_sun = L_rad + math.radians(1.915) * math.sin(g_rad)
    
    # Distance to Sun (AU)
    r_au = 1.00014 - 0.01671 * math.cos(g_rad)
    r_km = r_au * 149598023.0  # Convert AU to km
    
    # Position in ecliptic coordinates (J2000)
    # For simplicity, ignore obliquity (23.44°) - use ECI directly
    x = r_km * math.cos(lambda_sun)
    y = r_km * math.sin(lambda_sun)
    z = 0.0
    
    return (x, y, z)
