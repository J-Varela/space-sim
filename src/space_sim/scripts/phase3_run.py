import math

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements
from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import Engine
from space_sim.simulation.systems.state_recorder import StateRecorderSystem
from space_sim.simulation.systems.visibility_system import VisibilitySystem
from space_sim.analysis.coverage import access_windows_from_log


def deg(x): return x * math.pi / 180.0

# Build scenario
scenario = Scenario(name="Phase3 Demo")

sat = Satellite(
    sat_id="SAT-001",
    name="DemoSat",
    elements=OrbitalElements(
        a_km=7000.0,
        e=0.001,
        inc_rad=deg(51.6),
        raan_rad=deg(30.0),
        argp_rad=deg(40.0),
        M0_rad=0.0,
    ),
)

gs = GroundStation(
    station_id="GS-001",
    name="DemoGS",
    lat_deg=38.8339,
    lon_deg=-104.8214,
    alt_km=1.9,
    min_elevation_deg=10.0,
)

scenario.add_satellite(sat)
scenario.add_ground_station(gs)

# Run engine
engine = Engine(
    dt_s=10.0,
    systems=[StateRecorderSystem(), VisibilitySystem()],
)

log = engine.run(scenario, t_start_s=0.0, t_end_s=2 * 3600.0)

# Post-process access windows
windows = access_windows_from_log(log)

print("Access windows:")
for (gs_id, sat_id), ws in windows.items():
    print(f"{gs_id} -> {sat_id}: {len(ws)} windows")
    for a, b in ws[:5]:
        print(f"  start={a:8.1f}s end={b:8.1f}s duration={b-a:6.1f}s")

print("\nRecorded positions:", {k: len(v) for k, v in log.sat_positions_eci_km.items()})
print("Recorded visibility series:", {k: len(v) for k, v in log.visibility.items()})
