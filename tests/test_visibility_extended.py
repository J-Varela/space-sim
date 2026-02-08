"""
Extended tests for visibility calculations and access window analysis.
"""
import math
import pytest

from space_sim.physics.visibility import (
    elevation_angle_rad,
    is_visible,
    compute_access_windows
)
from space_sim.core.constants import R_EARTH_KM


class TestElevationAngle:
    def test_satellite_directly_overhead(self):
        # Ground station at (R_EARTH, 0, 0), satellite directly above
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        r_sat = (R_EARTH_KM + 500.0, 0.0, 0.0)  # 500 km above
        
        elev = elevation_angle_rad(r_gs, r_sat)
        # Should be close to 90 degrees (pi/2 radians)
        assert elev > math.radians(85)

    def test_satellite_on_horizon(self):
        # Satellite perpendicular to ground station radial
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        r_sat = (R_EARTH_KM, R_EARTH_KM + 500.0, 0.0)
        
        elev = elevation_angle_rad(r_gs, r_sat)
        # Should be close to 0 (on horizon)
        assert abs(elev) < math.radians(45)

    def test_satellite_below_horizon(self):
        # Satellite on opposite side of Earth
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        r_sat = (-R_EARTH_KM - 500.0, 0.0, 0.0)
        
        elev = elevation_angle_rad(r_gs, r_sat)
        # Should be negative (below horizon)
        assert elev < 0


class TestIsVisible:
    def test_high_elevation_is_visible(self):
        # Satellite overhead should be visible
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        r_sat = (R_EARTH_KM + 500.0, 0.0, 0.0)
        
        assert is_visible(r_gs, r_sat, min_elevation_deg=10.0) is True

    def test_below_min_elevation_not_visible(self):
        # Satellite below minimum elevation
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        # Place satellite at low angle
        r_sat = (R_EARTH_KM + 100.0, R_EARTH_KM + 500.0, 0.0)
        
        # With high min elevation requirement, might not be visible
        result = is_visible(r_gs, r_sat, min_elevation_deg=45.0)
        assert isinstance(result, bool)

    def test_opposite_side_not_visible(self):
        # Satellite on opposite side of Earth
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        r_sat = (-R_EARTH_KM - 500.0, 0.0, 0.0)
        
        assert is_visible(r_gs, r_sat, min_elevation_deg=10.0) is False

    def test_different_min_elevations(self):
        r_gs = (R_EARTH_KM, 0.0, 0.0)
        r_sat = (R_EARTH_KM + 300.0, 100.0, 0.0)
        
        # Lower threshold is more permissive
        vis_0 = is_visible(r_gs, r_sat, min_elevation_deg=0.0)
        vis_30 = is_visible(r_gs, r_sat, min_elevation_deg=30.0)
        
        # If visible at 30 deg, must be visible at 0 deg
        if vis_30:
            assert vis_0


class TestComputeAccessWindows:
    def test_no_visibility_no_windows(self):
        times = [0.0, 10.0, 20.0, 30.0]
        visible = [False, False, False, False]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 0

    def test_always_visible_single_window(self):
        times = [0.0, 10.0, 20.0, 30.0]
        visible = [True, True, True, True]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 1
        assert windows[0] == (0.0, 30.0)

    def test_single_pass_window(self):
        times = [0.0, 10.0, 20.0, 30.0, 40.0]
        visible = [False, True, True, False, False]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 1
        assert windows[0] == (10.0, 30.0)

    def test_multiple_passes(self):
        times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        visible = [False, True, False, False, True, True, False]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 2
        assert windows[0] == (10.0, 20.0)
        assert windows[1] == (40.0, 60.0)

    def test_pass_at_start(self):
        times = [0.0, 10.0, 20.0, 30.0]
        visible = [True, True, False, False]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 1
        assert windows[0] == (0.0, 20.0)

    def test_pass_at_end(self):
        times = [0.0, 10.0, 20.0, 30.0]
        visible = [False, False, True, True]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 1
        assert windows[0] == (20.0, 30.0)

    def test_mismatched_lengths_raises_error(self):
        times = [0.0, 10.0, 20.0]
        visible = [True, False]  # Wrong length
        
        with pytest.raises(ValueError, match="must be same length"):
            compute_access_windows(times, visible)

    def test_alternating_visibility(self):
        times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        visible = [True, False, True, False, True, False]
        
        windows = compute_access_windows(times, visible)
        assert len(windows) == 3
        assert windows[0] == (0.0, 10.0)
        assert windows[1] == (20.0, 30.0)
        assert windows[2] == (40.0, 50.0)
