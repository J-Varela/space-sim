"""
Tests for coordinate frame transformations.
"""
import math
import pytest

from space_sim.core.frames import (
    rot1, rot3, 
    geodetic_to_ecef_km, ecef_to_latlon_deg,
    ecef_to_eci_km, eci_to_ecef_km,
    dot, sub, norm
)
from space_sim.core.constants import R_EARTH_KM, OMEGA_EARTH_RAD_S


class TestVectorOperations:
    def test_dot_product(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        assert dot(a, b) == 1*4 + 2*5 + 3*6
        assert dot(a, b) == 32.0

    def test_subtraction(self):
        a = (5.0, 7.0, 9.0)
        b = (2.0, 3.0, 4.0)
        result = sub(a, b)
        assert result == (3.0, 4.0, 5.0)

    def test_norm(self):
        v = (3.0, 4.0, 0.0)
        assert norm(v) == 5.0
        
        v2 = (1.0, 0.0, 0.0)
        assert norm(v2) == 1.0


class TestRotations:
    def test_rot3_90_degrees(self):
        # Rotate (1,0,0) by 90 degrees about z-axis -> should give (0,1,0)
        v = (1.0, 0.0, 0.0)
        result = rot3(math.pi / 2, v)
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10
        assert abs(result[2] - 0.0) < 1e-10

    def test_rot3_identity(self):
        # Rotate by 0 should return same vector
        v = (1.0, 2.0, 3.0)
        result = rot3(0.0, v)
        assert result == v

    def test_rot1_90_degrees(self):
        # Rotate (0,1,0) by 90 degrees about x-axis -> should give (0,0,1)
        v = (0.0, 1.0, 0.0)
        result = rot1(math.pi / 2, v)
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10

    def test_rot1_identity(self):
        v = (1.0, 2.0, 3.0)
        result = rot1(0.0, v)
        assert result == v


class TestGeodeticToECEF:
    def test_equator_prime_meridian(self):
        # Lat=0, Lon=0, Alt=0 -> should be at (R_EARTH, 0, 0)
        lat_rad = 0.0
        lon_rad = 0.0
        alt_km = 0.0
        
        result = geodetic_to_ecef_km(lat_rad, lon_rad, alt_km)
        assert abs(result[0] - R_EARTH_KM) < 1e-6
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10

    def test_north_pole(self):
        # Lat=90, Lon=any, Alt=0 -> should be at (0, 0, R_EARTH)
        lat_rad = math.pi / 2
        lon_rad = 0.0
        alt_km = 0.0
        
        result = geodetic_to_ecef_km(lat_rad, lon_rad, alt_km)
        assert abs(result[0]) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2] - R_EARTH_KM) < 1e-6

    def test_with_altitude(self):
        # Altitude should increase radius
        lat_rad = 0.0
        lon_rad = 0.0
        alt_km = 100.0
        
        result = geodetic_to_ecef_km(lat_rad, lon_rad, alt_km)
        expected_r = R_EARTH_KM + 100.0
        assert abs(result[0] - expected_r) < 1e-6


class TestECEFToLatLon:
    def test_prime_meridian_equator(self):
        r_ecef = (R_EARTH_KM, 0.0, 0.0)
        lat, lon = ecef_to_latlon_deg(r_ecef)
        assert abs(lat) < 1e-6
        assert abs(lon) < 1e-6

    def test_north_pole(self):
        r_ecef = (0.0, 0.0, R_EARTH_KM)
        lat, lon = ecef_to_latlon_deg(r_ecef)
        assert abs(lat - 90.0) < 1e-6

    def test_roundtrip_conversion(self):
        # Start with lat/lon/alt, convert to ECEF, then back
        lat_deg = 38.8339
        lon_deg = -104.8214
        alt_km = 1.9
        
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        
        r_ecef = geodetic_to_ecef_km(lat_rad, lon_rad, alt_km)
        lat_result, lon_result = ecef_to_latlon_deg(r_ecef)
        
        assert abs(lat_result - lat_deg) < 1e-6
        assert abs(lon_result - lon_deg) < 1e-6

    def test_zero_vector_raises_error(self):
        r_ecef = (0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="Zero ECEF vector"):
            ecef_to_latlon_deg(r_ecef)


class TestECEFECIConversions:
    def test_eci_to_ecef_at_t0(self):
        # At t=0, ECI and ECEF should be aligned
        r_eci = (7000.0, 0.0, 0.0)
        r_ecef = eci_to_ecef_km(r_eci, 0.0)
        
        assert abs(r_ecef[0] - r_eci[0]) < 1e-10
        assert abs(r_ecef[1] - r_eci[1]) < 1e-10
        assert abs(r_ecef[2] - r_eci[2]) < 1e-10

    def test_ecef_to_eci_at_t0(self):
        # At t=0, conversion should be identity
        r_ecef = (7000.0, 0.0, 0.0)
        r_eci = ecef_to_eci_km(r_ecef, 0.0)
        
        assert abs(r_eci[0] - r_ecef[0]) < 1e-10
        assert abs(r_eci[1] - r_ecef[1]) < 1e-10
        assert abs(r_eci[2] - r_ecef[2]) < 1e-10

    def test_earth_rotation_effect(self):
        # After some time, x-component in ECEF should rotate in ECI
        r_ecef = (R_EARTH_KM, 0.0, 0.0)
        t_s = 3600.0  # 1 hour
        
        r_eci = ecef_to_eci_km(r_ecef, t_s)
        
        # Position should have rotated about z-axis
        # Radius should remain the same
        ecef_radius = norm(r_ecef)
        eci_radius = norm(r_eci)
        assert abs(eci_radius - ecef_radius) < 1e-6
        
        # But x,y components should have changed
        assert r_eci != r_ecef

    def test_roundtrip_ecef_eci(self):
        # ECEF -> ECI -> ECEF should return original
        r_ecef_orig = (7000.0, 1000.0, 500.0)
        t_s = 1234.5
        
        r_eci = ecef_to_eci_km(r_ecef_orig, t_s)
        r_ecef_back = eci_to_ecef_km(r_eci, t_s)
        
        assert abs(r_ecef_back[0] - r_ecef_orig[0]) < 1e-9
        assert abs(r_ecef_back[1] - r_ecef_orig[1]) < 1e-9
        assert abs(r_ecef_back[2] - r_ecef_orig[2]) < 1e-9
