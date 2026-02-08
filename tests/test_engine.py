"""
Tests for simulation engine and scenario components.
"""
import math
import pytest

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements
from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import Engine, SimulationLog


def deg(x):
    return x * math.pi / 180.0


@pytest.fixture
def sample_satellite():
    return Satellite(
        sat_id="SAT-TEST",
        name="TestSat",
        elements=OrbitalElements(
            a_km=7000.0,
            e=0.001,
            inc_rad=deg(51.6),
            raan_rad=deg(30.0),
            argp_rad=deg(40.0),
            M0_rad=0.0,
        )
    )


@pytest.fixture
def sample_ground_station():
    return GroundStation(
        station_id="GS-TEST",
        name="TestGS",
        lat_deg=38.8339,
        lon_deg=-104.8214,
        alt_km=1.9,
        min_elevation_deg=10.0,
    )


class TestScenario:
    def test_scenario_creation(self):
        scenario = Scenario(name="Test Scenario")
        assert scenario.name == "Test Scenario"
        assert len(scenario.satellites) == 0
        assert len(scenario.ground_stations) == 0

    def test_add_satellite(self, sample_satellite):
        scenario = Scenario(name="Test")
        scenario.add_satellite(sample_satellite)
        
        assert len(scenario.satellites) == 1
        assert "SAT-TEST" in scenario.satellites
        assert scenario.satellites["SAT-TEST"] == sample_satellite

    def test_add_ground_station(self, sample_ground_station):
        scenario = Scenario(name="Test")
        scenario.add_ground_station(sample_ground_station)
        
        assert len(scenario.ground_stations) == 1
        assert "GS-TEST" in scenario.ground_stations
        assert scenario.ground_stations["GS-TEST"] == sample_ground_station

    def test_satellite_list(self, sample_satellite):
        scenario = Scenario(name="Test")
        scenario.add_satellite(sample_satellite)
        
        sat_list = scenario.satellite_list()
        assert len(sat_list) == 1
        assert sat_list[0] == sample_satellite

    def test_ground_station_list(self, sample_ground_station):
        scenario = Scenario(name="Test")
        scenario.add_ground_station(sample_ground_station)
        
        gs_list = scenario.ground_station_list()
        assert len(gs_list) == 1
        assert gs_list[0] == sample_ground_station


class TestSimulationLog:
    def test_log_creation(self):
        log = SimulationLog()
        assert len(log.sat_positions_eci_km) == 0
        assert len(log.visibility) == 0
        assert len(log.events) == 0

    def test_record_position(self):
        log = SimulationLog()
        log.record_position("SAT-001", 0.0, (7000.0, 0.0, 0.0))
        log.record_position("SAT-001", 10.0, (6900.0, 100.0, 0.0))
        
        assert "SAT-001" in log.sat_positions_eci_km
        assert len(log.sat_positions_eci_km["SAT-001"]) == 2
        assert log.sat_positions_eci_km["SAT-001"][0] == (0.0, (7000.0, 0.0, 0.0))
        assert log.sat_positions_eci_km["SAT-001"][1] == (10.0, (6900.0, 100.0, 0.0))

    def test_record_visibility(self):
        log = SimulationLog()
        log.record_visibility("GS-001", "SAT-001", 0.0, False)
        log.record_visibility("GS-001", "SAT-001", 10.0, True)
        
        key = ("GS-001", "SAT-001")
        assert key in log.visibility
        assert len(log.visibility[key]) == 2
        assert log.visibility[key][0] == (0.0, False)
        assert log.visibility[key][1] == (10.0, True)


class TestEngine:
    def test_engine_creation(self):
        engine = Engine(dt_s=10.0)
        assert engine.dt_s == 10.0
        assert len(engine.systems) == 0

    def test_engine_validation_negative_dt(self):
        engine = Engine(dt_s=-1.0)
        scenario = Scenario(name="Test")
        
        with pytest.raises(ValueError, match="dt_s must be positive"):
            engine.run(scenario, t_start_s=0.0, t_end_s=100.0)

    def test_engine_validation_end_before_start(self):
        engine = Engine(dt_s=10.0)
        scenario = Scenario(name="Test")
        
        with pytest.raises(ValueError, match="t_end_s must be >= t_start_s"):
            engine.run(scenario, t_start_s=100.0, t_end_s=0.0)

    def test_engine_run_empty_scenario(self):
        engine = Engine(dt_s=10.0)
        scenario = Scenario(name="Test")
        
        log = engine.run(scenario, t_start_s=0.0, t_end_s=50.0)
        assert isinstance(log, SimulationLog)

    def test_engine_with_mock_system(self, sample_satellite):
        """Test that engine calls system on_step method at each timestep."""
        from dataclasses import dataclass, field
        from typing import List
        
        @dataclass
        class MockSystem:
            name: str = "mock"
            call_times: List[float] = field(default_factory=list)
            
            def on_step(self, t_s: float, scenario, log):
                self.call_times.append(t_s)
        
        mock_system = MockSystem()
        engine = Engine(dt_s=10.0, systems=[mock_system])
        scenario = Scenario(name="Test")
        scenario.add_satellite(sample_satellite)
        
        log = engine.run(scenario, t_start_s=0.0, t_end_s=30.0)
        
        # Should be called at t=0, 10, 20, 30
        assert len(mock_system.call_times) == 4
        assert mock_system.call_times[0] == 0.0
        assert mock_system.call_times[1] == 10.0
        assert mock_system.call_times[2] == 20.0
        assert mock_system.call_times[3] == 30.0
