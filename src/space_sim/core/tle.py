"""
Two-Line Element (TLE) parsing and conversion.

Allows importing real satellite data from TLE format and converting
to orbital elements for simulation.

TLE Format:
Line 0 (optional): Satellite name
Line 1: Catalog number, epoch, ballistic coefficient, etc.
Line 2: Inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from space_sim.physics.orbit import OrbitalElements
from space_sim.core.constants import MU_EARTH_KM3_S2


@dataclass
class TLE:
    """Two-Line Element set representation."""
    line0: str  # Satellite name (optional)
    line1: str  # First line of TLE
    line2: str  # Second line of TLE
    
    # Parsed fields from Line 1
    catalog_number: int = 0
    classification: str = "U"
    international_designator: str = ""
    epoch_year: int = 0
    epoch_day: float = 0.0
    mean_motion_derivative: float = 0.0  # rev/day²
    mean_motion_second_derivative: float = 0.0  # rev/day³
    bstar: float = 0.0  # Drag term
    element_set_number: int = 0
    
    # Parsed fields from Line 2
    inclination_deg: float = 0.0
    raan_deg: float = 0.0
    eccentricity: float = 0.0
    argument_of_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    mean_motion_rev_day: float = 0.0
    revolution_number: int = 0


def parse_tle(line0: str, line1: str, line2: str) -> TLE:
    """
    Parse a two-line element set.
    
    Args:
        line0: Satellite name (can be empty)
        line1: First line of TLE (69 characters)
        line2: Second line of TLE (69 characters)
        
    Returns:
        Parsed TLE object
    """
    tle = TLE(line0=line0.strip(), line1=line1.strip(), line2=line2.strip())
    
    # Validate line numbers
    if not line1.startswith('1 '):
        raise ValueError("Line 1 must start with '1 '")
    if not line2.startswith('2 '):
        raise ValueError("Line 2 must start with '2 '")
    
    # Parse Line 1
    try:
        tle.catalog_number = int(line1[2:7].strip())
        tle.classification = line1[7].strip()
        tle.international_designator = line1[9:17].strip()
        
        # Epoch
        epoch_year_str = line1[18:20]
        tle.epoch_year = int(epoch_year_str)
        # Convert 2-digit year to 4-digit (assumes 1957-2056 range)
        if tle.epoch_year < 57:
            tle.epoch_year += 2000
        else:
            tle.epoch_year += 1900
        
        tle.epoch_day = float(line1[20:32].strip())
        
        tle.mean_motion_derivative = float(line1[33:43].strip())
        
        # Second derivative (stored as mantissa with assumed decimal)
        second_deriv_str = line1[44:52].strip()
        if second_deriv_str:
            tle.mean_motion_second_derivative = parse_exponential_notation(second_deriv_str)
        
        # BSTAR drag term
        bstar_str = line1[53:61].strip()
        if bstar_str:
            tle.bstar = parse_exponential_notation(bstar_str)
        
        tle.element_set_number = int(line1[64:68].strip())
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing TLE line 1: {e}")
    
    # Parse Line 2
    try:
        # Verify catalog number matches
        catalog_check = int(line2[2:7].strip())
        if catalog_check != tle.catalog_number:
            raise ValueError("Catalog number mismatch between lines")
        
        tle.inclination_deg = float(line2[8:16].strip())
        tle.raan_deg = float(line2[17:25].strip())
        
        # Eccentricity (stored without leading decimal point)
        ecc_str = line2[26:33].strip()
        tle.eccentricity = float("0." + ecc_str)
        
        tle.argument_of_perigee_deg = float(line2[34:42].strip())
        tle.mean_anomaly_deg = float(line2[43:51].strip())
        tle.mean_motion_rev_day = float(line2[52:63].strip())
        tle.revolution_number = int(line2[63:68].strip())
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing TLE line 2: {e}")
    
    return tle


def parse_exponential_notation(s: str) -> float:
    """
    Parse TLE exponential notation (e.g., '+12345-3' means 0.12345e-3).
    
    Args:
        s: String in TLE exponential format
        
    Returns:
        Float value
    """
    s = s.strip()
    if not s or s == '+00000-0' or s == '-00000-0' or s == '00000-0':
        return 0.0
    
    # Handle various formats
    if len(s) < 6:
        return 0.0
    
    # Format: ±XXXXX±Y means ±0.XXXXX × 10^±Y
    sign = 1.0 if s[0] != '-' else -1.0
    mantissa_str = s[1:6]
    
    # Check if we have exponent part
    if len(s) > 6:
        exp_sign_char = s[6] if len(s) > 6 else '+'
        exp_str = s[7] if len(s) > 7 else '0'
    else:
        exp_sign_char = '+'
        exp_str = '0'
    
    try:
        mantissa = float(mantissa_str) / 100000.0
        exponent = int(exp_str)
    except ValueError:
        return 0.0
    
    if exp_sign_char == '-':
        exponent = -exponent
    
    return sign * mantissa * (10.0 ** exponent)


def tle_to_orbital_elements(tle: TLE, mu: float = MU_EARTH_KM3_S2) -> OrbitalElements:
    """
    Convert TLE to orbital elements for simulator.
    
    Args:
        tle: Parsed TLE object
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        OrbitalElements for use in simulation
    """
    # Convert mean motion (rev/day) to mean motion (rad/s)
    n_rev_day = tle.mean_motion_rev_day
    n_rad_s = n_rev_day * (2.0 * math.pi) / 86400.0  # Convert rev/day to rad/s
    
    # Compute semi-major axis from mean motion: n = sqrt(mu/a³)
    # a = (mu/n²)^(1/3)
    a_km = (mu / (n_rad_s ** 2)) ** (1.0 / 3.0)
    
    # Convert angles to radians
    inc_rad = math.radians(tle.inclination_deg)
    raan_rad = math.radians(tle.raan_deg)
    argp_rad = math.radians(tle.argument_of_perigee_deg)
    M0_rad = math.radians(tle.mean_anomaly_deg)
    
    return OrbitalElements(
        a_km=a_km,
        e=tle.eccentricity,
        inc_rad=inc_rad,
        raan_rad=raan_rad,
        argp_rad=argp_rad,
        M0_rad=M0_rad
    )


def load_tle_from_file(filepath: str) -> list[TLE]:
    """
    Load TLE data from a file.
    
    Supports formats:
    - 3-line TLE (name + 2 lines)
    - 2-line TLE (just the 2 data lines)
    
    Args:
        filepath: Path to TLE file
        
    Returns:
        List of parsed TLE objects
    """
    with open(filepath, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    
    tles = []
    i = 0
    
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
        
        # Check if this is a 3-line or 2-line format
        if i + 1 < len(lines) and lines[i + 1].startswith('1 '):
            # 3-line format
            line0 = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2] if i + 2 < len(lines) else ""
            
            if line2.startswith('2 '):
                tle = parse_tle(line0, line1, line2)
                tles.append(tle)
                i += 3
            else:
                i += 1
        elif lines[i].startswith('1 '):
            # 2-line format
            line0 = ""
            line1 = lines[i]
            line2 = lines[i + 1] if i + 1 < len(lines) else ""
            
            if line2.startswith('2 '):
                tle = parse_tle(line0, line1, line2)
                tles.append(tle)
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return tles


def tle_from_string(tle_string: str) -> TLE:
    """
    Parse TLE from a multi-line string.
    
    Args:
        tle_string: String containing TLE data (2 or 3 lines)
        
    Returns:
        Parsed TLE object
    """
    lines = [line.strip() for line in tle_string.strip().split('\n') if line.strip()]
    
    if len(lines) == 2:
        return parse_tle("", lines[0], lines[1])
    elif len(lines) == 3:
        return parse_tle(lines[0], lines[1], lines[2])
    else:
        raise ValueError("TLE string must contain 2 or 3 lines")


def format_tle(tle: TLE) -> str:
    """
    Format TLE object back to standard 3-line string format.
    
    Args:
        tle: TLE object
        
    Returns:
        Formatted TLE string
    """
    output = []
    
    if tle.line0:
        output.append(tle.line0)
    
    output.append(tle.line1)
    output.append(tle.line2)
    
    return '\n'.join(output)


# Example TLEs for testing
EXAMPLE_ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   21275.51020370  .00003026  00000-0  63146-4 0  9993
2 25544  51.6454 297.5612 0003681  73.8901  43.4185 15.48957534303374"""

EXAMPLE_STARLINK_TLE = """STARLINK-1007
1 44713U 19074A   21275.50886752  .00001156  00000-0  93328-4 0  9998
2 44713  53.0534 123.4578 0001387  87.6543 272.4623 15.06380957106897"""


def example_tle_loading():
    """Demonstrate TLE loading and conversion."""
    # Parse ISS TLE
    iss_tle = tle_from_string(EXAMPLE_ISS_TLE)
    print(f"Satellite: {iss_tle.line0}")
    print(f"Catalog #: {iss_tle.catalog_number}")
    print(f"Inclination: {iss_tle.inclination_deg}°")
    print(f"Eccentricity: {iss_tle.eccentricity}")
    
    # Convert to orbital elements
    iss_elements = tle_to_orbital_elements(iss_tle)
    print(f"\nOrbital Elements:")
    print(f"  Semi-major axis: {iss_elements.a_km:.2f} km")
    print(f"  Eccentricity: {iss_elements.e:.6f}")
    print(f"  Inclination: {math.degrees(iss_elements.inc_rad):.4f}°")
    
    return iss_elements


if __name__ == "__main__":
    example_tle_loading()
