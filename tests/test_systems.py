"""
Tests for simulation systems (state recorder and visibility).
"""
import math

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements
from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import Engine, SimulationLog
from space_sim.simulation.systems.state_recorder import StateRecorderSystem
from space_sim.simulation.systems.visibility_system import VisibilitySystem


def deg(x):
    return x * math.pi / 180.0


class TestStateRecorderSystem:
    def test_system_creation(self):
        system = StateRecorderSystem()
        assert system.name == "state_recorder"

    def test_records_satellite_positions(self):
        # Create scenario with one satellite
        sat = Satellite(
            sat_id="SAT-001",
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
        
        scenario = Scenario(name="Test")
        scenario.add_satellite(sat)
        
        # Run system manually at two timesteps
        log = SimulationLog()
        system = StateRecorderSystem()
        
        system.on_step(0.0, scenario, log)
        system.on_step(10.0, scenario, log)
        
        # Check recorded data
        assert "SAT-001" in log.sat_positions_eci_km
        assert len(log.sat_positions_eci_km["SAT-001"]) == 2
        
        t0, r0 = log.sat_positions_eci_km["SAT-001"][0]
        t1, r1 = log.sat_positions_eci_km["SAT-001"][1]
        
        assert t0 == 0.0
        assert t1 == 10.0
        # Position should be different at different times
        assert r0 != r1

    def test_records_multiple_satellites(self):
        scenario = Scenario(name="Test")
        
        scenario.add_satellite(Satellite(
            sat_id="SAT-001",
            name="Sat1",
            elements=OrbitalElements(
                a_km=7000.0, e=0.001, inc_rad=deg(51.6),
                raan_rad=deg(30.0), argp_rad=deg(40.0), M0_rad=0.0,
            )
        ))
        
        scenario.add_satellite(Satellite(
            sat_id="SAT-002",
            name="Sat2",
            elements=OrbitalElements(
                a_km=7200.0, e=0.01, inc_rad=deg(97.0),
                raan_rad=deg(110.0), argp_rad=deg(10.0), M0_rad=deg(180.0),
            )
        ))
        
        log = SimulationLog()
        system = StateRecorderSystem()
        system.on_step(0.0, scenario, log)
        
        assert "SAT-001" in log.sat_positions_eci_km
        assert "SAT-002" in log.sat_positions_eci_km


class TestVisibilitySystem:
    def test_system_creation(self):
        system = VisibilitySystem()
        assert system.name == "visibility"

    def test_records_visibility(self):
        # Create scenario with satellite and ground station
        sat = Satellite(
            sat_id="SAT-001",
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
        
        gs = GroundStation(
            station_id="GS-001",
            name="TestGS",
            lat_deg=38.8339,
            lon_deg=-104.8214,
            alt_km=1.9,
            min_elevation_deg=10.0,
        )
        
        scenario = Scenario(name="Test")
        scenario.add_satellite(sat)
        scenario.add_ground_station(gs)
        
        log = SimulationLog()
        system = VisibilitySystem()
        
        system.on_step(0.0, scenario, log)
        
        # Check visibility was recorded
        key = ("GS-001", "SAT-001")
        assert key in log.visibility
        assert len(log.visibility[key]) == 1
        
        t, vis = log.visibility[key][0]
        assert t == 0.0
        assert isinstance(vis, bool)

    def test_multiple_ground_stations_and_satellites(self):
        scenario = Scenario(name="Test")
        
        # Add two satellites
        scenario.add_satellite(Satellite(
            sat_id="SAT-001", name="Sat1",
            elements=OrbitalElements(
                a_km=7000.0, e=0.001, inc_rad=deg(51.6),
                raan_rad=deg(30.0), argp_rad=deg(40.0), M0_rad=0.0,
            )
        ))
        
        scenario.add_satellite(Satellite(
            sat_id="SAT-002", name="Sat2",
            elements=OrbitalElements(
                a_km=7200.0, e=0.01, inc_rad=deg(97.0),
                raan_rad=deg(110.0), argp_rad=deg(10.0), M0_rad=deg(180.0),
            )
        ))
        
        # Add two ground stations
        scenario.add_ground_station(GroundStation(
            station_id="GS-001", name="GS1",
            lat_deg=38.8339, lon_deg=-104.8214, alt_km=1.9,
        ))
        
        scenario.add_ground_station(GroundStation(
            station_id="GS-002", name="GS2",
            lat_deg=28.4, lon_deg=-80.6, alt_km=0.0,
        ))
        
        log = SimulationLog()
        system = VisibilitySystem()
        system.on_step(0.0, scenario, log)
        
        # Should have 2 GS × 2 SAT = 4 visibility records
        assert len(log.visibility) == 4
        assert ("GS-001", "SAT-001") in log.visibility
        assert ("GS-001", "SAT-002") in log.visibility
        assert ("GS-002", "SAT-001") in log.visibility
        assert ("GS-002", "SAT-002") in log.visibility


class TestIntegratedSimulation:
    def test_full_simulation_run(self):
        """Integration test: run a complete simulation with both systems."""
        scenario = Scenario(name="Integration Test")
        
        scenario.add_satellite(Satellite(
            sat_id="SAT-001",
            name="TestSat",
            elements=OrbitalElements(
                a_km=7000.0,
                e=0.001,
                inc_rad=deg(51.6),
                raan_rad=deg(30.0),
                argp_rad=deg(40.0),
                M0_rad=0.0,
            )
        ))
        
        scenario.add_ground_station(GroundStation(
            station_id="GS-001",
            name="TestGS",
            lat_deg=38.8339,
            lon_deg=-104.8214,
            alt_km=1.9,
            min_elevation_deg=10.0,
        ))
        
        engine = Engine(
            dt_s=60.0,
            systems=[StateRecorderSystem(), VisibilitySystem()]
        )
        
        log = engine.run(scenario, t_start_s=0.0, t_end_s=300.0)
        
        # Should have 6 timesteps (0, 60, 120, 180, 240, 300)
        assert len(log.sat_positions_eci_km["SAT-001"]) == 6
        assert len(log.visibility[("GS-001", "SAT-001")]) == 6
        
        # Verify times are correct
        times = [t for t, _ in log.sat_positions_eci_km["SAT-001"]]
        assert times == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
