"""
Communication link budget analysis.

Includes:
- Link budget calculations
- Signal-to-noise ratio
- Data rate estimation
- Doppler shift
- Antenna gain patterns
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from space_sim.core.frames import Vector3


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
SPEED_OF_LIGHT_KM_S = 299792.458  # km/s


@dataclass
class Transmitter:
    """Transmitter specifications."""
    power_w: float  # Transmit power (Watts)
    frequency_hz: float  # Carrier frequency (Hz)
    antenna_gain_dbi: float = 0.0  # Antenna gain (dBi)
    losses_db: float = 2.0  # System losses (cables, connectors, etc.)
    
    def eirp_dbw(self) -> float:
        """Effective Isotropic Radiated Power in dBW."""
        power_dbw = 10.0 * math.log10(self.power_w)
        return power_dbw + self.antenna_gain_dbi - self.losses_db


@dataclass
class Receiver:
    """Receiver specifications."""
    antenna_gain_dbi: float = 0.0  # Receive antenna gain (dBi)
    system_noise_temp_k: float = 290.0  # System noise temperature (K)
    losses_db: float = 2.0  # System losses
    
    def g_over_t_db(self) -> float:
        """Figure of merit G/T in dB/K."""
        return self.antenna_gain_dbi - 10.0 * math.log10(self.system_noise_temp_k) - self.losses_db


def free_space_path_loss_db(distance_km: float, frequency_hz: float) -> float:
    """
    Calculate free space path loss.
    
    Args:
        distance_km: Distance between transmitter and receiver (km)
        frequency_hz: Carrier frequency (Hz)
        
    Returns:
        Path loss in dB
    """
    if distance_km <= 0:
        return 0.0
    
    # Convert distance to meters
    distance_m = distance_km * 1000.0
    
    # Wavelength
    wavelength_m = SPEED_OF_LIGHT_KM_S * 1000.0 / frequency_hz
    
    # FSPL = 20*log10(4*pi*d/lambda)
    fspl = 20.0 * math.log10((4.0 * math.pi * distance_m) / wavelength_m)
    
    return fspl


def atmospheric_loss_db(elevation_deg: float, frequency_ghz: float) -> float:
    """
    Approximate atmospheric attenuation.
    
    Args:
        elevation_deg: Elevation angle above horizon (degrees)
        frequency_ghz: Frequency in GHz
        
    Returns:
        Atmospheric loss in dB
    """
    if elevation_deg < 0:
        return 999.0  # Below horizon
    
    if elevation_deg >= 90:
        zenith_loss = 0.1 * frequency_ghz  # Simplified model
        return zenith_loss
    
    # Approximate atmospheric path length factor
    zenith_loss = 0.1 * frequency_ghz
    path_length_factor = 1.0 / math.sin(math.radians(max(5.0, elevation_deg)))
    
    return zenith_loss * path_length_factor


@dataclass
class LinkBudget:
    """Complete link budget calculation."""
    transmitter: Transmitter
    receiver: Receiver
    distance_km: float
    elevation_deg: float = 90.0  # Elevation angle for atmospheric loss
    additional_losses_db: float = 0.0  # Rain, polarization, etc.
    
    def received_power_dbw(self) -> float:
        """Calculate received power in dBW."""
        eirp = self.transmitter.eirp_dbw()
        fspl = free_space_path_loss_db(self.distance_km, self.transmitter.frequency_hz)
        atm_loss = atmospheric_loss_db(self.elevation_deg, 
                                       self.transmitter.frequency_hz / 1e9)
        
        rx_power = (eirp + self.receiver.antenna_gain_dbi - 
                   fspl - atm_loss - self.additional_losses_db - 
                   self.receiver.losses_db)
        
        return rx_power
    
    def carrier_to_noise_ratio_db(self, bandwidth_hz: float = 1.0) -> float:
        """
        Calculate carrier-to-noise ratio (C/N).
        
        Args:
            bandwidth_hz: Receiver bandwidth (Hz)
            
        Returns:
            C/N ratio in dB
        """
        rx_power = self.received_power_dbw()
        
        # Noise power: N = k * T * B
        noise_power_w = BOLTZMANN_CONSTANT * self.receiver.system_noise_temp_k * bandwidth_hz
        noise_power_dbw = 10.0 * math.log10(noise_power_w)
        
        return rx_power - noise_power_dbw
    
    def eb_n0_db(self, data_rate_bps: float) -> float:
        """
        Calculate Eb/N0 (energy per bit to noise spectral density).
        
        Args:
            data_rate_bps: Data rate in bits per second
            
        Returns:
            Eb/N0 in dB
        """
        cn_ratio = self.carrier_to_noise_ratio_db(data_rate_bps)
        return cn_ratio
    
    def max_data_rate_bps(self, required_eb_n0_db: float = 10.0) -> float:
        """
        Calculate maximum achievable data rate.
        
        Args:
            required_eb_n0_db: Required Eb/N0 for desired BER
            
        Returns:
            Maximum data rate in bps
        """
        rx_power_dbw = self.received_power_dbw()
        rx_power_w = 10.0 ** (rx_power_dbw / 10.0)
        
        # N0 = k * T
        n0 = BOLTZMANN_CONSTANT * self.receiver.system_noise_temp_k
        
        # Eb/N0 = (C/R) / N0, where R is data rate
        # R = C / (Eb/N0 * N0)
        eb_n0_linear = 10.0 ** (required_eb_n0_db / 10.0)
        
        max_rate = rx_power_w / (eb_n0_linear * n0)
        
        return max_rate
    
    def link_margin_db(self, required_eb_n0_db: float, data_rate_bps: float) -> float:
        """
        Calculate link margin.
        
        Args:
            required_eb_n0_db: Required Eb/N0 for link closure
            data_rate_bps: Operating data rate
            
        Returns:
            Link margin in dB (positive = good)
        """
        actual_eb_n0 = self.eb_n0_db(data_rate_bps)
        return actual_eb_n0 - required_eb_n0_db


def doppler_shift_hz(frequency_hz: float, relative_velocity_km_s: float) -> float:
    """
    Calculate Doppler frequency shift.
    
    Args:
        frequency_hz: Transmitted frequency (Hz)
        relative_velocity_km_s: Relative velocity (positive = approaching)
        
    Returns:
        Doppler shift in Hz
    """
    # Doppler shift: Δf = f * (v/c)
    shift = frequency_hz * (relative_velocity_km_s / SPEED_OF_LIGHT_KM_S)
    return shift


def radial_velocity(r_sat: Vector3, v_sat: Vector3, r_gs: Vector3) -> float:
    """
    Calculate radial velocity between satellite and ground station.
    
    Args:
        r_sat: Satellite position (km)
        v_sat: Satellite velocity (km/s)
        r_gs: Ground station position (km)
        
    Returns:
        Radial velocity in km/s (positive = approaching, negative = receding)
    """
    # Vector from ground station to satellite
    dx = r_sat[0] - r_gs[0]
    dy = r_sat[1] - r_gs[1]
    dz = r_sat[2] - r_gs[2]
    
    range_km = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    if range_km < 1e-10:
        return 0.0
    
    # Unit vector from GS to satellite
    u_range = (dx/range_km, dy/range_km, dz/range_km)
    
    # Radial velocity = velocity dot unit_range
    v_radial = v_sat[0]*u_range[0] + v_sat[1]*u_range[1] + v_sat[2]*u_range[2]
    
    return v_radial


@dataclass
class AntennaPattern:
    """Simple antenna gain pattern."""
    max_gain_dbi: float  # Peak gain
    beamwidth_deg: float  # 3dB beamwidth
    
    def gain_at_angle(self, off_axis_deg: float) -> float:
        """
        Calculate gain at off-axis angle.
        
        Uses simple Gaussian approximation.
        
        Args:
            off_axis_deg: Angle from boresight (degrees)
            
        Returns:
            Gain in dBi
        """
        if off_axis_deg > 90:
            return -30.0  # Back lobe
        
        # Gaussian approximation
        # 3dB at beamwidth/2
        sigma = self.beamwidth_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        loss_db = -12.0 * (off_axis_deg / self.beamwidth_deg) ** 2
        
        return self.max_gain_dbi + max(loss_db, -30.0)


def compute_link_at_position(r_sat_eci: Vector3, v_sat_eci: Vector3,
                            r_gs_eci: Vector3,
                            tx_power_w: float = 10.0,
                            frequency_ghz: float = 2.4,
                            tx_gain_dbi: float = 3.0,
                            rx_gain_dbi: float = 15.0) -> dict:
    """
    Compute complete link budget for satellite-ground station link.
    
    Args:
        r_sat_eci: Satellite position in ECI (km)
        v_sat_eci: Satellite velocity in ECI (km/s)
        r_gs_eci: Ground station position in ECI (km)
        tx_power_w: Transmit power (W)
        frequency_ghz: Frequency (GHz)
        tx_gain_dbi: Transmit antenna gain (dBi)
        rx_gain_dbi: Receive antenna gain (dBi)
        
    Returns:
        Dictionary with link parameters
    """
    # Calculate range
    dx = r_sat_eci[0] - r_gs_eci[0]
    dy = r_sat_eci[1] - r_gs_eci[1]
    dz = r_sat_eci[2] - r_gs_eci[2]
    range_km = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Setup link budget
    tx = Transmitter(
        power_w=tx_power_w,
        frequency_hz=frequency_ghz * 1e9,
        antenna_gain_dbi=tx_gain_dbi
    )
    
    rx = Receiver(
        antenna_gain_dbi=rx_gain_dbi,
        system_noise_temp_k=290.0
    )
    
    link = LinkBudget(
        transmitter=tx,
        receiver=rx,
        distance_km=range_km,
        elevation_deg=45.0  # Simplified
    )
    
    # Doppler
    v_rad = radial_velocity(r_sat_eci, v_sat_eci, r_gs_eci)
    doppler_hz = doppler_shift_hz(tx.frequency_hz, v_rad)
    
    # Calculate link parameters
    data_rate_bps = 1e6  # 1 Mbps
    
    return {
        "range_km": range_km,
        "received_power_dbw": link.received_power_dbw(),
        "cn_ratio_db": link.carrier_to_noise_ratio_db(data_rate_bps),
        "eb_n0_db": link.eb_n0_db(data_rate_bps),
        "link_margin_db": link.link_margin_db(10.0, data_rate_bps),
        "max_data_rate_mbps": link.max_data_rate_bps(10.0) / 1e6,
        "doppler_shift_hz": doppler_hz,
        "radial_velocity_km_s": v_rad,
    }
