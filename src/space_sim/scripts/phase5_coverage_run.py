import math

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements
from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import Engine
from space_sim.simulation.systems.state_recorder import StateRecorderSystem
from space_sim.simulation.systems.visibility_system import VisibilitySystem

from space_sim.analysis.earth_coverage import (
    CoverageGridSpec,
    compute_earth_coverage_from_log,
    compute_constellation_coverage_from_log,
    save_coverage_json,
)

def deg(x): return x * math.pi / 180.0

scenario = Scenario(name="Phase5 Coverage Demo")

scenario.add_satellite(Satellite(
    sat_id="SAT-001",
    name="DemoSat1",
    elements=OrbitalElements(
        a_km=7000.0,
        e=0.001,
        inc_rad=deg(51.6),
        raan_rad=deg(30.0),
        argp_rad=deg(40.0),
        M0_rad=0.0,
    )
))

scenario.add_satellite(Satellite(
    sat_id="SAT-002",
    name="DemoSat2",
    elements=OrbitalElements(
        a_km=7200.0,
        e=0.01,
        inc_rad=deg(97.0),
        raan_rad=deg(110.0),
        argp_rad=deg(10.0),
        M0_rad=deg(180.0),
    )
))

scenario.add_ground_station(GroundStation(
    station_id="GS-001",
    name="DemoGS",
    lat_deg=38.8339,
    lon_deg=-104.8214,
    alt_km=1.9,
    min_elevation_deg=10.0,
))

dt = 10.0
engine = Engine(dt_s=dt, systems=[StateRecorderSystem(), VisibilitySystem()])

# Run longer to see coverage patterns
log = engine.run(scenario, t_start_s=0.0, t_end_s=6 * 3600.0)

grid = CoverageGridSpec(lat_step_deg=2.0, lon_step_deg=2.0)

# Coverage for each sat
for sat_id in ["SAT-001", "SAT-002"]:
    result = compute_earth_coverage_from_log(
        log=log,
        sat_id=sat_id,
        dt_s=dt,
        grid=grid,
        fov_half_angle_deg=20.0,   # try 10, 20, 30
    )
    out = save_coverage_json(result, out_path=f"out/coverage_{sat_id}.json")
    print("Wrote:", out)

const = compute_constellation_coverage_from_log(
    log=log,
    sat_ids=["SAT-001", "SAT-002"],
    dt_s=dt,
    grid=grid,
    fov_half_angle_deg=20.0
)
out = save_coverage_json(const, out_path="out/coverage_CONSTELLATION.json")
print("Wrote:", out)
