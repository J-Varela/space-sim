import math

from space_sim.objects.satellite import Satellite
from space_sim.objects.ground_station import GroundStation
from space_sim.physics.orbit import OrbitalElements
from space_sim.simulation.scenario import Scenario
from space_sim.simulation.engine import Engine
from space_sim.simulation.systems.state_recorder import StateRecorderSystem
from space_sim.simulation.systems.visibility_system import VisibilitySystem
from space_sim.visualization.plotly_multisat import render_multisat_playback
from space_sim.visualization.export_log import export_log_to_json
from space_sim.visualization.export_log import export_playback_bundle


def deg(x): return x * math.pi / 180.0

scenario = Scenario(name="Phase4 MultiSat Demo")

# Sat 1
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

# Sat 2 (different RAAN/argp so you can SEE separation)
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

engine = Engine(dt_s=10.0, systems=[StateRecorderSystem(), VisibilitySystem()])
log = engine.run(scenario, t_start_s=0.0, t_end_s=2 * 3600.0)

json_path = export_log_to_json(log, out_path="out/simlog.json")
print("Exported:", json_path)

bundle_path = export_playback_bundle(scenario, log, out_path="out/playback_bundle.json")
print("Exported bundle:", bundle_path)

path = render_multisat_playback(
    log,
    out_html="out/phase4_multisat.html",
    frame_stride=1,   # increase to 2/3/5 if it feels heavy
    trail_len=250
)

print("Wrote:", path)
print("Open out/phase4_multisat.html in your browser.")
