import math
import pytest

from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements


def test_ground_station_validates_latitude():
    with pytest.raises(ValueError, match="Latitude must be in range"):
        GroundStation("GS-1", "Test", lat_deg=95.0, lon_deg=0.0)
    
    with pytest.raises(ValueError, match="Latitude must be in range"):
        GroundStation("GS-1", "Test", lat_deg=-95.0, lon_deg=0.0)


def test_ground_station_validates_longitude():
    with pytest.raises(ValueError, match="Longitude must be in range"):
        GroundStation("GS-1", "Test", lat_deg=0.0, lon_deg=185.0)
    
    with pytest.raises(ValueError, match="Longitude must be in range"):
        GroundStation("GS-1", "Test", lat_deg=0.0, lon_deg=-185.0)


def test_ground_station_validates_altitude():
    with pytest.raises(ValueError, match="Altitude must be non-negative"):
        GroundStation("GS-1", "Test", lat_deg=0.0, lon_deg=0.0, alt_km=-1.0)


def test_ground_station_validates_elevation():
    with pytest.raises(ValueError, match="Minimum elevation must be in range"):
        GroundStation("GS-1", "Test", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=95.0)
    
    with pytest.raises(ValueError, match="Minimum elevation must be in range"):
        GroundStation("GS-1", "Test", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=-5.0)


def test_ground_station_validates_station_id():
    with pytest.raises(ValueError, match="Station ID cannot be empty"):
        GroundStation("", "Test", lat_deg=0.0, lon_deg=0.0)
    
    with pytest.raises(ValueError, match="Station ID cannot be empty"):
        GroundStation("   ", "Test", lat_deg=0.0, lon_deg=0.0)


def test_ground_station_validates_name():
    with pytest.raises(ValueError, match="Station name cannot be empty"):
        GroundStation("GS-1", "", lat_deg=0.0, lon_deg=0.0)
    
    with pytest.raises(ValueError, match="Station name cannot be empty"):
        GroundStation("GS-1", "   ", lat_deg=0.0, lon_deg=0.0)


def test_ground_station_accepts_valid_values():
    # Should not raise
    gs = GroundStation("GS-1", "Test Station", lat_deg=38.8339, lon_deg=-104.8214, alt_km=1.9, min_elevation_deg=10.0)
    assert gs.station_id == "GS-1"
    assert gs.name == "Test Station"


def test_orbital_elements_validates_inclination():
    with pytest.raises(ValueError, match="Inclination must be in range"):
        OrbitalElements(a_km=7000.0, e=0.01, inc_rad=-0.1, raan_rad=0.0, argp_rad=0.0, M0_rad=0.0)
    
    with pytest.raises(ValueError, match="Inclination must be in range"):
        OrbitalElements(a_km=7000.0, e=0.01, inc_rad=3.5, raan_rad=0.0, argp_rad=0.0, M0_rad=0.0)


def test_orbital_elements_validates_raan():
    with pytest.raises(ValueError, match="RAAN must be finite"):
        OrbitalElements(a_km=7000.0, e=0.01, inc_rad=1.0, raan_rad=math.inf, argp_rad=0.0, M0_rad=0.0)
    
    with pytest.raises(ValueError, match="RAAN must be finite"):
        OrbitalElements(a_km=7000.0, e=0.01, inc_rad=1.0, raan_rad=math.nan, argp_rad=0.0, M0_rad=0.0)


def test_orbital_elements_validates_argp():
    with pytest.raises(ValueError, match="Argument of periapsis must be finite"):
        OrbitalElements(a_km=7000.0, e=0.01, inc_rad=1.0, raan_rad=0.0, argp_rad=math.inf, M0_rad=0.0)


def test_orbital_elements_validates_mean_anomaly():
    with pytest.raises(ValueError, match="Mean anomaly must be finite"):
        OrbitalElements(a_km=7000.0, e=0.01, inc_rad=1.0, raan_rad=0.0, argp_rad=0.0, M0_rad=math.nan)


def test_orbital_elements_accepts_valid_values():
    # Should not raise
    elements = OrbitalElements(
        a_km=7000.0,
        e=0.001,
        inc_rad=math.radians(51.6),
        raan_rad=math.radians(30.0),
        argp_rad=math.radians(40.0),
        M0_rad=0.0,
    )
    assert elements.a_km == 7000.0
    assert elements.e == 0.001
